import os
import cv2
import csv
import numpy as np
import json
import argparse
from datetime import datetime, timedelta
import re
from collections import deque
import sys # (เพิ่ม sys เพื่อใช้ sys.exit)

# --- Dependencies ---
from ultralytics import YOLO
from sort import Sort
# --- FIX: ตรวจสอบตำแหน่ง config/model_config ---
try:
    from config import model_config as cfg
except ImportError:
    try:
        import model_config as cfg
    except ImportError:
        class DefaultConfig:
            MAX_AGE_FRAMES = 120; SCORE_THR = 0.35
            STATE_RETENTION_S = 10.0
        cfg = DefaultConfig(); print("Warning: model_config.py not found.")

# --- การตั้งค่าที่สำคัญ ---
CONFIG_FILE = 'config/new_camera_config.json'
BASE_OUTPUT_DIR = "qa_camera_check"
BASE_OUTPUT_RESULT = "qa_camera_check/ai_result"
SIGN_HISTORY_LENGTH = 3

current_run_timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
# --- Tesseract ---
try:
    import pytesseract
except ImportError: pytesseract = None; print("Warn: pytesseract not found.")

# =================== MODEL / TRACKER ====================
print("Loading AI model...")
model_path = "core/yolov8m.pt" if os.path.exists(os.path.join("core", "yolov8m.pt")) else "yolov8m.pt"
if not os.path.exists(model_path):
     model_path_n = "yolov8n.pt"; model_path_n_core = os.path.join("core", "yolov8n.pt")
     if os.path.exists(model_path_n): model_path = model_path_n; print(f"Warn: {model_path} not found.")
     elif os.path.exists(model_path_n_core): model_path = model_path_n_core; print(f"Warn: {model_path} not found.")
     else: 
         print("Error: Could not find yolov8m.pt or yolov8n.pt")
         sys.exit(1) # (เพิ่มการ Exit ถ้าไม่เจอโมเดล)
model = YOLO(model_path, verbose=False)
tracker = Sort(max_age=cfg.MAX_AGE_FRAMES, min_hits=3, iou_threshold=0.2)
print("Model loaded successfully.")

# ====================== HELPERS =========================
def _cross_sign(p, a, b):
    try: p_arr=np.array(p,dtype=np.float64); a_arr=np.array(a,dtype=np.float64); b_arr=np.array(b,dtype=np.float64)
    except: return 0
    val = (b_arr[0]-a_arr[0])*(p_arr[1]-a_arr[1])-(b_arr[1]-a_arr[1])*(p_arr[0]-a_arr[0])
    return 0 if abs(val)<1e-9 else int(np.sign(val))

def make_side_label(a, b):
    a,b=np.array(a),np.array(b); mid_below=(a+b)/2.0+np.array([0,100]); return _cross_sign(mid_below,a,b)<0

def get_timestamp_from_frame(frame, roi):
    # (โค้ด Tesseract เหมือนเดิม)
    if pytesseract is None or roi is None: return None
    try:
        x1,y1,x2,y2=roi; h,w,_=frame.shape; x1,y1=max(0,x1),max(0,y1); x2,y2=min(w,x2),min(h,y2)
        if y2<=y1 or x2<=x1: return None
        ts_img=frame[y1:y2,x1:x2]; gray=cv2.cvtColor(ts_img,cv2.COLOR_BGR2GRAY)
        binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, blockSize=7, C=3)
        text=pytesseract.image_to_string(binary,config=r'--oem 3 --psm 6')
        match = re.search(r'(\d{2})-(\d{2})-(\d{4}).*?(\d{2}:\d{2}:\d{2})', text.replace(" ", ""))
        if match: 
            day, month, year, time_str = match.groups()
            try: return datetime.strptime(f"{day}-{month}-{year} {time_str}", '%d-%m-%Y %H:%M:%S')
            except ValueError: return None
    except Exception as e: return None
    return None

def ensure_dir(dir_path):
    if not os.path.exists(dir_path): os.makedirs(dir_path); print(f"Created directory: {dir_path}")

def format_seconds(seconds, hour_offset=None):
    if seconds is None: return "N/A"
    total_seconds = int(seconds)
    if hour_offset is not None:
        minutes = (total_seconds % 3600) // 60
        seconds_rem = total_seconds % 60
        return f"{hour_offset:02d}:{minutes:02d}:{seconds_rem:02d}"
    else:
        return str(timedelta(seconds=total_seconds))

def is_crossing_line(p1, p2, a, b):
    if p1 is None or p2 is None or a is None or b is None: return False
    s1 = _cross_sign(p1, a, b); s2 = _cross_sign(p2, a, b)
    if s1 * s2 < 0:
        s3 = _cross_sign(a, p1, p2); s4 = _cross_sign(b, p1, p2)
        if s3 * s4 <= 0: return True
    elif s1 == 0 and s2 != 0:
        s3 = _cross_sign(a, p1, p2); s4 = _cross_sign(b, p1, p2)
        if s3 * s4 <= 0: return True
    elif s2 == 0 and s1 != 0:
        s3 = _cross_sign(a, p1, p2); s4 = _cross_sign(b, p1, p2)
        if s3 * s4 <= 0: return True
    return False

# --- ( ✨ 1. เพิ่มฟังก์ชันใหม่นี้ ✨ ) ---
def write_video_clip(frames_list, output_path, width, height, fps):
    """
    ใช้ cv2.VideoWriter เพื่อบันทึก list ของเฟรมเป็นไฟล์วิดีโอ
    """
    if not frames_list:
        print("Warn: No frames in buffer to write video.")
        return
        
    if not fps or fps <= 0:
        fps = 25.0 # (ตั้งค่า FPS เริ่มต้น)
        
    print(f"Saving video clip ({len(frames_list)} frames) to {output_path}...")
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec สำหรับ .mp4
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in frames_list:
            out.write(frame)
        out.release()
        print(f"Video clip saved successfully: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"Error writing video clip: {e}")
# --- ( สิ้นสุดส่วนที่เพิ่ม ) ---

def draw_overlays(frame, config, bbox=None, pid=None, state=None, color=(0, 255, 0)):
    """ฟังก์ชันช่วยวาดเส้นและกรอบคนลงบนเฟรม"""
    # 1. วาดเส้นและโซน (Static)
    pink_zone = config['pink_zone']
    lines = config['lines']
    
    if pink_zone:
        cv2.polylines(frame, [np.array(pink_zone, dtype=np.int32)], isClosed=True, color=(255, 182, 193), thickness=2)
    
    cv2.line(frame, tuple(lines['red'][0]), tuple(lines['red'][1]), (0,0,255), 2)
    cv2.line(frame, tuple(lines['blue'][0]), tuple(lines['blue'][1]), (255,0,0), 2)
    cv2.line(frame, tuple(lines['green'][0]), tuple(lines['green'][1]), (0,255,0), 2)
    cv2.line(frame, tuple(lines['yellow'][0]), tuple(lines['yellow'][1]), (0,255,255), 2)

    # 2. วาดกรอบคน (Dynamic) - ถ้ามีข้อมูลส่งมา
    if bbox is not None and pid is not None:
        # วาดกล่อง
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        # วาดจุด
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int(bbox[1]) # หรือตำแหน่งอื่นตาม logic
        cv2.circle(frame, (center_x, center_y), 5, color, -1)
        # วาด Text
        label = f"PID:{pid} ({state})"
        cv2.putText(frame, label, (bbox[0], max(20, bbox[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    return frame

# ====================== MAIN LOGIC =========================
def main():
    parser = argparse.ArgumentParser(description="Person Counter (Summary Log + Time Range)")
    parser.add_argument("camera_name", help="Name of the camera config.")
    parser.add_argument("--start_min", type=int, default=0, help="Start processing at this minute in the video (default: 0)")
    parser.add_argument("--duration_min", type=int, default=None, help="Process for this many minutes (default: process until end of video)")
    parser.add_argument("--video_hour", type=int, default=None, help="Manual hour (e.g., 18) to use for the Log file")
    parser.add_argument("--run_date", type=str, default=None, help="Date string (YYYYMMDD) for the master log file")
    args = parser.parse_args()
    
    video_hour = None
    if args.video_hour is not None: video_hour = int(args.video_hour)
    print("---------------------------------")
    
    # --- Load Config ---
    try:
        with open(CONFIG_FILE,"r",encoding='utf-8') as f: full_config=json.load(f)
    except: 
        print(f"Config '{CONFIG_FILE}' not found.")
        sys.exit(1) # (Exit ถ้าไม่มี Config)
    if args.camera_name not in full_config: 
        print(f"Camera '{args.camera_name}' not found.")
        sys.exit(1) # (Exit ถ้าไม่มีชื่อกล้อง)
    config = full_config[args.camera_name]

    video_path=config.get('video_path'); display_width=config.get('display_width',1280)
    red_line=tuple(map(tuple,config['lines']['red']))
    blue_line=tuple(map(tuple,config['lines']['blue']))
    green_line=tuple(map(tuple,config['lines']['green']))
    yellow_line=tuple(map(tuple,config['lines']['yellow']))
    pink_zone = config['pink_zone']
    timestamp_roi=config.get('timestamp_roi')
    file_name=config.get('file_name');

    # --- MODIFIED: สร้าง Path สำหรับการรันครั้งนี้ ---
    run_output_dir = os.path.join(BASE_OUTPUT_DIR,"camera", args.camera_name, current_run_timestamp)
    log_dir = os.path.join(run_output_dir, "logs")
    person_snapshot_dir = os.path.join(run_output_dir, "person_snapshots")
    ensure_dir(log_dir); ensure_dir(person_snapshot_dir)
    
    event_log_path = os.path.join(log_dir, f"event_log_{args.camera_name}_{current_run_timestamp}.csv")
    summary_log_path = os.path.join(run_output_dir, f"summary_log_{args.camera_name}_{current_run_timestamp}.csv")
    
    # --- NEW: กำหนด Path สำหรับ Master Log File ---
    today_date_str = args.run_date if args.run_date else datetime.now().strftime('%Y%m%d')
    master_log_filename = f"validation_{today_date_str}.csv"
    master_log_path = os.path.join(BASE_OUTPUT_RESULT, master_log_filename)
    # --- END NEW ---

    print(f"--- Starting Run ---")
    print(f"Event Log: {event_log_path}"); print(f"Snapshots: {person_snapshot_dir}"); print(f"Summary Log: {summary_log_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        print(f"Error: Cannot open video: {video_path}")
        sys.exit(1) # (Exit ถ้าเปิดวิดีโอไม่ได้)
        
    original_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); original_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS) # <-- ( ✨ 2. อ่านค่า FPS ✨ )
    if original_w==0 or original_h==0: 
        print("Error: Could not read video dimensions.")
        sys.exit(1)
    aspect=original_w/max(1,original_h); display_height=int(display_width/aspect)

   
    counts={"inbound":0}; person_states={}; next_pid=1
    tid_to_pid = {}
    
    neg_is_bottom_red=make_side_label(red_line[0],red_line[1])
    bottom_sign=-1 if neg_is_bottom_red else 1; top_sign=-bottom_sign
    
    video_start_time_processed = None

    try:
        with open(event_log_path, "w", newline="", encoding='utf-8') as csv_file:
            csvw = csv.writer(csv_file, delimiter=','); 
            csvw.writerow(["Cam_name","Timestamp","End_Time","TraceID","Status"])

            while True:
                current_time_dt = datetime.now()
                
                # --- ( ✨ 3. ลบ Logic 'paused' และ 'last_frame' ออก ✨ ) ---
                ret, frame = cap.read()
                if not ret: break
                # --- ( สิ้นสุดการแก้ไข ) ---

                current_video_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                current_video_sec = current_video_msec / 1000.0
                
                ocr_timestamp_dt = get_timestamp_from_frame(frame, timestamp_roi)
                display_timestamp_str = ocr_timestamp_dt.strftime('%d-%m-%Y %H:%M:%S') if ocr_timestamp_dt else ""

                # --- Time Range Check (เหมือนเดิม) ---
                process_this_frame = True
                if current_video_sec < (args.start_min * 60):
                    process_this_frame = False
                elif video_start_time_processed is None:
                    video_start_time_processed = current_video_sec
                    print(f"Processing started at video time: {format_seconds(video_start_time_processed)}")
                if args.duration_min is not None and video_start_time_processed is not None and \
                   (current_video_sec - video_start_time_processed) > (args.duration_min * 60):
                    print(f"Processing duration of {args.duration_min} minutes reached. Stopping.")
                    break
                if process_this_frame:
                     video_end_time_processed = current_video_sec

                if process_this_frame:
                    dets=[]; tracks=np.empty((0,5)); valid_results=False
                    results = model(frame, stream=True, conf=cfg.SCORE_THR, verbose=False)
                    for r in results:
                        valid_results=True
                        for box in r.boxes.data:
                            if len(box)>=6 and int(box[5])==0: dets.append([int(b) for b in box[:4]]+[float(box[4])])
                    if valid_results: tracks=tracker.update(np.array(dets) if dets else np.empty((0,5)))
                    live_tids = {int(t[4]) for t in tracks}

                    # --- State Machine & Re-ID Logic ---
                    processed_pids_this_frame = set()
                    for x1, y1, x2, y2, tid in tracks:
                        tid, bbox = int(tid), (int(x1), int(y1), int(x2), int(y2))
                        cur_pos = np.array([(x1 + x2) / 2, y1])
                        pid = tid_to_pid.get(tid)
                        if pid is None or pid not in person_states:
                            pid = next_pid; next_pid += 1
                            tid_to_pid[tid] = pid
                            
                            # --- ( ✨ 4. แก้ไข 'person_states' ✨ ) ---
                            # (คำนวณ maxlen: 5 วินาที @ FPS (ขั้นต่ำ 25fps))
                            pre_buffer_size = int((original_fps or 25.0) * 5) 
                            person_states[pid] = {
                                'state': 'waiting', 'sign_history': deque(maxlen=SIGN_HISTORY_LENGTH), 
                                'last_frame_seen': frame.copy(), 'last_bbox': bbox, 
                                'last_pos': cur_pos, 'last_tid': tid, 
                                'last_seen_time': current_time_dt, 'prev_pos': None,
                                'dot_color': (0, 0, 255),
                                # --- (เพิ่ม Buffer) ---
                                'pre_cross_buffer': deque(maxlen=pre_buffer_size),
                                'post_cross_buffer': [] 
                            }
                            # --- ( สิ้นสุดการแก้ไข ) ---
                            
                        st = person_states[pid]
                        st['tid'] = tid; st['last_bbox'] = bbox; st['last_frame_seen'] = frame.copy()
                        st['last_pos'] = cur_pos; st['last_seen_time'] = current_time_dt
                        processed_pids_this_frame.add(pid)

                        current_color = (0, 0, 255) # แดง (Default)
                        if st['state'] == 'crossed_red':
                            current_color = (0, 255, 0) # เขียว

                        # 2. สร้างเฟรมสำหรับบันทึก (Copy มาวาด)
                        annotated_frame = frame.copy()
                        draw_overlays(annotated_frame, config, bbox, pid, st['state'], current_color)
                        
                        # --- ( ✨ 5. เพิ่ม Logic บันทึก Frame ลง Buffer ✨ ) ---
                        if st['state'] == 'waiting':
                            st['pre_cross_buffer'].append(annotated_frame)
                        elif st['state'] == 'crossed_red':
                            st['post_cross_buffer'].append(annotated_frame)
                        # --- ( สิ้นสุดการเพิ่ม ) ---

                        prev_pos = st.get('prev_pos')
                        if prev_pos is not None:
                            crossed = is_crossing_line(prev_pos, cur_pos, red_line[0], red_line[1])
                            if st['state'] == 'waiting' and crossed and cur_pos[1] > prev_pos[1]:
                                st['state'] = 'crossed_red'
                                st['dot_color'] = (0, 255, 0) 
                                st['cross_time_sec'] = current_video_sec
                                
                                # --- ( ✨ 6. เพิ่ม Logic เท Buffer ✨ ) ---
                                st['post_cross_buffer'] = list(st['pre_cross_buffer'])
                                st['pre_cross_buffer'].clear()
                                # --- ( สิ้นสุดการเพิ่ม ) ---
                                
                            elif st['state'] == 'crossed_red' and crossed and cur_pos[1] < prev_pos[1]:
                                st['state'] = 'waiting'
                                st['dot_color'] = (0, 0, 255)
                        
                        st['prev_pos'] = cur_pos.copy()
                        
                        # (ลบโค้ด UI: cv2.rectangle, cv2.putText, cv2.circle)

                    # --- Process Disappeared People & Cleanup ---
                    pids_to_remove = set()
                    retention_seconds = getattr(cfg, 'STATE_RETENTION_S', 10.0)
                    for pid, st in person_states.items():
                        if pid not in processed_pids_this_frame:
                            if st['state'] == 'crossed_red':
                                 counts['inbound'] += 1
                                 
                                 cross_time_sec = st.get('cross_time_sec', current_video_sec)
                                 exit_time_sec = current_video_sec 

                                 cross_time_str = format_seconds(cross_time_sec, video_hour)
                                 exit_time_str = format_seconds(exit_time_sec, video_hour)

                                 video_time_str = format_seconds(cross_time_sec, video_hour) 
                                 print(f"PID {pid}: Exited -> COUNT = {counts['inbound']} (Video Time: {video_time_str})", flush=True)
                                 csvw.writerow([args.camera_name, cross_time_str, exit_time_str, pid, 'entrance'])
                                 
                                 # --- NEW: บันทึกลง Master Validation Log ---
                                 try:
                                     master_log_dir = os.path.dirname(master_log_path) 
                                     ensure_dir(master_log_dir) 
                                     file_exists = os.path.isfile(master_log_path)
                                     with open(master_log_path, "a", newline="", encoding='utf-8') as master_f:
                                         master_csvw = csv.writer(master_f, delimiter=',')
                                         if not file_exists:
                                             master_csvw.writerow(["Cam_name", "Timestamp", "EndTime", "TraceID", "Status"])
                                         master_csvw.writerow([file_name, cross_time_str, exit_time_str, pid, 'entrance'])
                                 except Exception as e:
                                     print(f"Error writing to Master Log: {e}")
                                 # --- END NEW ---

                                 # --- ( ✨ 7. แทนที่ Snapshot ด้วย Video Clip ✨ ) ---
                                 if st['post_cross_buffer']: # เช็กว่ามีเฟรมให้บันทึกหรือไม่
                                      
                                      video_time_fname = f"{int(cross_time_sec // 3600):02d}h{int((cross_time_sec % 3600) // 60):02d}m{int(cross_time_sec % 60):02d}s"
                                      
                                      # (เปลี่ยนนามสกุลเป็น .mp4)
                                      video_clip_path = os.path.join(person_snapshot_dir, f"inbound_pid{pid}_{video_time_fname}.mp4")

                                      # (เรียกฟังก์ชัน Helper ใหม่ที่เราสร้าง)
                                      write_video_clip(
                                          st['post_cross_buffer'], 
                                          video_clip_path, 
                                          original_w, 
                                          original_h, 
                                          original_fps
                                      )
                                 else: 
                                      print(f"Warn: No video frames in buffer for PID {pid}.")
                                 
                                 # (ล้าง Memory และเปลี่ยนสถานะ)
                                 st['post_cross_buffer'].clear()
                                 st['pre_cross_buffer'].clear()
                                 st['state'] = 'counted'
                                 # --- ( สิ้นสุดการแทนที่ ) ---
                            
                            if current_time_dt - st.get('last_seen_time', datetime.min) > timedelta(seconds=retention_seconds):
                                pids_to_remove.add(pid)
                            elif st['state'] != 'counted':
                                 st['state'] = 'waiting'
                                 last_tid = st.get('last_tid')
                                 if last_tid in tid_to_pid and tid_to_pid[last_tid] == pid: del tid_to_pid[last_tid]
                    for pid in pids_to_remove:
                        if pid in person_states:
                            last_tid = person_states[pid].get('last_tid')
                            if last_tid in tid_to_pid and tid_to_pid[last_tid] == pid: del tid_to_pid[last_tid]
                            del person_states[pid]
                
                # --- ( ✨ 8. ลบ UI Display ทั้งหมด ✨ ) ---
                # (ลบ cv2.polylines, cv2.line, cv2.putText, cv2.imshow, cv2.waitKey)
                #
            
    except KeyboardInterrupt:
        print("\nUser interrupted process (Ctrl+C).")
        try:
             video_end_time_processed = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        except:
             pass 
    finally:
        # (ส่วน Summary Log ยังคงปิดไว้)
        
        cap.release()
        # (ลบ cv2.destroyAllWindows())
        print("Process finished.")

if __name__ == "__main__":
    main()