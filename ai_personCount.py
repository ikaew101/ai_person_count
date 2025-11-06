import os
import cv2
import csv
import numpy as np
import json
import argparse
from datetime import datetime, timedelta # เพิ่ม timedelta
import re
from collections import deque # เพิ่ม deque

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
CONFIG_FILE = 'config/camera_config.json'
BASE_OUTPUT_DIR = "qa_camera_check" # โฟลเดอร์หลัก
SIGN_HISTORY_LENGTH = 3
# --- REMOVED: INTERVAL_MINUTES ---

current_run_timestamp = datetime.now().strftime('%Y%m%d%H%M%S') # เวลาที่เริ่มรันสคริปต์
# --- Tesseract ---
try:
    import pytesseract
    # TESSERACT_PATH = r'...'
    # pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
except ImportError: pytesseract = None; print("Warn: pytesseract not found.")

# =================== MODEL / TRACKER ====================
print("Loading AI model...")
model_path = "core/yolov8m.pt" if os.path.exists(os.path.join("core", "yolov8m.pt")) else "yolov8m.pt"
if not os.path.exists(model_path):
     model_path_n = "yolov8n.pt"; model_path_n_core = os.path.join("core", "yolov8n.pt")
     if os.path.exists(model_path_n): model_path = model_path_n; print(f"Warn: {model_path} not found.")
     elif os.path.exists(model_path_n_core): model_path = model_path_n_core; print(f"Warn: {model_path} not found.")
     else: raise FileNotFoundError("Could not find yolov8m.pt or yolov8n.pt")
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

def get_timestamp_from_frame(frame, roi): # (ยังคงไว้สำหรับ UI Display)
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

# --- NEW: Helper สำหรับแปลงวินาทีเป็น HH:MM:SS ---
def format_seconds(seconds, hour_offset=None):
    """แปลงวินาที (float) เป็น string 'HH:MM:SS'"""
    if seconds is None: return "N/A"
    
    total_seconds = int(seconds)
    
    # ถ้ามี hour_offset ให้ใช้เป็นชั่วโมง
    if hour_offset is not None:
        # คำนวณนาทีและวินาทีที่เหลือ (โดยไม่สนชั่วโมงของ video time)
        minutes = (total_seconds % 3600) // 60
        seconds_rem = total_seconds % 60
        return f"{hour_offset:02d}:{minutes:02d}:{seconds_rem:02d}"
    else:
        # ถ้าไม่มี ให้แปลงตามปกติ
        return str(timedelta(seconds=total_seconds))
# --- END NEW ---

def is_crossing_line(p1, p2, a, b):
    """
    ตรวจสอบว่าเส้นจาก p1 ไป p2 ตัดกับเส้น a ไป b หรือไม่
    (ปรับปรุงให้รองรับกรณีจุดอยู่บนเส้น)
    """
    # ตรวจสอบชนิดข้อมูลก่อนเรียก _cross_sign
    if p1 is None or p2 is None or a is None or b is None:
        return False
        
    s1 = _cross_sign(p1, a, b)
    s2 = _cross_sign(p2, a, b)
    
    # กรณีทั่วไป: ข้ามเส้น (อยู่คนละฝั่ง)
    if s1 * s2 < 0:
        s3 = _cross_sign(a, p1, p2)
        s4 = _cross_sign(b, p1, p2)
        if s3 * s4 <= 0: return True
             
    # กรณีพิเศษ: จุดใดจุดหนึ่งอยู่บนเส้นพอดี
    elif s1 == 0 and s2 != 0:
        s3 = _cross_sign(a, p1, p2)
        s4 = _cross_sign(b, p1, p2)
        if s3 * s4 <= 0: return True
             
    elif s2 == 0 and s1 != 0:
        s3 = _cross_sign(a, p1, p2)
        s4 = _cross_sign(b, p1, p2)
        if s3 * s4 <= 0: return True

    return False

# ====================== MAIN LOGIC =========================
def main():
    # --- MODIFIED: เพิ่ม Arguments สำหรับ Time Range (ใช้ นาที) และ Hour Offset ---
    parser = argparse.ArgumentParser(description="Person Counter (Summary Log + Time Range)")
    parser.add_argument("camera_name", help="Name of the camera config.")
    parser.add_argument("--start_min", type=int, default=0, help="Start processing at this minute in the video (default: 0)")
    parser.add_argument("--duration_min", type=int, default=None, help="Process for this many minutes (default: process until end of video)")
    # --- NEW: เพิ่ม Argument สำหรับ Hour Offset ---
    parser.add_argument("--video_hour", type=int, default=None, help="Manual hour (e.g., 18) to use for the Log file")
    # --- END NEW ---
    args = parser.parse_args()
    # --- END MODIFIED ---

    # --- NEW: รับ Input Hour แบบ Interactive ---
    video_hour_str = input("Enter manual hour offset (e.g., 18) or press Enter to skip: ")
    video_hour = None
    if video_hour_str.isdigit():
        video_hour = int(video_hour_str)
    print("---------------------------------")
    # --- END NEW ---
    
    # --- Load Config ---
    try:
        with open(CONFIG_FILE,"r",encoding='utf-8') as f: full_config=json.load(f)
    except: raise SystemExit(f"Config '{CONFIG_FILE}' not found.")
    if args.camera_name not in full_config: raise SystemExit(f"Camera '{args.camera_name}' not found.")
    config = full_config[args.camera_name]

    video_path=config.get('video_path'); display_width=config.get('display_width',1280)
    red_line=tuple(map(tuple,config['lines']['red']))
    blue_line=tuple(map(tuple,config['lines']['blue']))
    green_line=tuple(map(tuple,config['lines']['green']))
    yellow_line=tuple(map(tuple,config['lines']['yellow']))
    pink_zone = config['pink_zone']
    timestamp_roi=config.get('timestamp_roi')

    # --- MODIFIED: สร้าง Path สำหรับการรันครั้งนี้ ---
    run_output_dir = os.path.join(BASE_OUTPUT_DIR, args.camera_name, current_run_timestamp)
    log_dir = os.path.join(run_output_dir, "logs")
    person_snapshot_dir = os.path.join(run_output_dir, "person_snapshots")
    ensure_dir(log_dir); ensure_dir(person_snapshot_dir)
    
    event_log_path = os.path.join(log_dir, f"event_log_{args.camera_name}_{current_run_timestamp}.csv")
    summary_log_path = os.path.join(run_output_dir, f"summary_log_{args.camera_name}_{current_run_timestamp}.csv") # ไฟล์สรุป
    
    print(f"--- Starting Run ---")
    print(f"Event Log: {event_log_path}"); print(f"Snapshots: {person_snapshot_dir}"); print(f"Summary Log: {summary_log_path}")
    # --- END MODIFIED ---

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise IOError(f"Cannot open video: {video_path}")
    original_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); original_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if original_w==0 or original_h==0: raise IOError("Could not read video dimensions.")
    aspect=original_w/max(1,original_h); display_height=int(display_width/aspect)

    cv2.namedWindow("Video Analysis", cv2.WINDOW_NORMAL)
    paused=False; mouse_pos_raw=(-1,-1)
    counts={"inbound":0}; person_states={}; next_pid=1
    tid_to_pid = {}
    
    neg_is_bottom_red=make_side_label(red_line[0],red_line[1])
    bottom_sign=-1 if neg_is_bottom_red else 1; top_sign=-bottom_sign

    # --- Mouse Callback (FIXED) ---
    CONFIG_HELPER_FILE = "config/config_points.txt"
    def _on_mouse(event, x, y, flags, param):
        nonlocal mouse_pos_raw
        rx = int(x * original_w / display_width)
        ry = int(y * original_h / display_height)
        mouse_pos_raw = (rx, ry)
        if event == cv2.EVENT_LBUTTONDOWN:
            coord_str = f"[{rx}, {ry}],"
            print(f"Clicked Coordinates: ({rx}, {ry}) - Saved to {CONFIG_HELPER_FILE}")
            try:
                with open(CONFIG_HELPER_FILE, "a") as f: f.write(coord_str + "\n")
            except Exception as e: print(f"Error writing to {CONFIG_HELPER_FILE}: {e}")
    cv2.setMouseCallback("Video Analysis", _on_mouse)
    # --- END FIX ---
    
    video_start_time_processed = None
    video_end_time_processed = None
    
    last_frame = None

    try:
        # --- MODIFIED: เปิดไฟล์ Event Log (เพิ่ม Header ใหม่) ---
        with open(event_log_path, "w", newline="", encoding='utf-8') as csv_file:
            csvw = csv.writer(csv_file, delimiter=','); csvw.writerow(["Video Time (HH:MM:SS)","Camera Name","PID","Status", "Hour (Manual)"])
            # --- END MODIFIED ---

            while True:
                current_time_dt = datetime.now()
                
                if not paused:
                    ret, frame = cap.read()
                    if not ret: break
                    last_frame = frame.copy()
                if last_frame is None: continue
                frame = last_frame.copy()

                current_video_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                current_video_sec = current_video_msec / 1000.0
                
                ocr_timestamp_dt = get_timestamp_from_frame(frame, timestamp_roi)
                display_timestamp_str = ocr_timestamp_dt.strftime('%d-%m-%Y %H:%M:%S') if ocr_timestamp_dt else ""

                # --- Time Range Check ---
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
                            # --- MODIFIED: เพิ่ม 'dot_color' ---
                            person_states[pid] = {'state': 'waiting', 'sign_history': deque(maxlen=SIGN_HISTORY_LENGTH), 
                                                  'last_frame_seen': frame.copy(), 'last_bbox': bbox, 
                                                  'last_pos': cur_pos, 'last_tid': tid, 
                                                  'last_seen_time': current_time_dt, 'prev_pos': None,
                                                  'dot_color': (0, 0, 255)} # สีแดง BGR
                        st = person_states[pid]
                        st['tid'] = tid; st['last_bbox'] = bbox; st['last_frame_seen'] = frame.copy()
                        st['last_pos'] = cur_pos; st['last_seen_time'] = current_time_dt
                        processed_pids_this_frame.add(pid)
                        prev_pos = st.get('prev_pos')

                        if prev_pos is not None:
                            crossed = is_crossing_line(prev_pos, cur_pos, red_line[0], red_line[1])
                            if st['state'] == 'waiting' and crossed and cur_pos[1] > prev_pos[1]:
                                st['state'] = 'crossed_red'
                                st['dot_color'] = (0, 255, 0) # --- NEW: เปลี่ยนเป็นสีเขียว ---
                                st['cross_time_sec'] = current_video_sec # <--- **เพิ่มบรรทัดนี้**
                            elif st['state'] == 'crossed_red' and crossed and cur_pos[1] < prev_pos[1]:
                                st['state'] = 'waiting'
                                st['dot_color'] = (0, 0, 255) # --- NEW: เปลี่ยนกลับเป็นสีแดง ---
                        
                        st['prev_pos'] = cur_pos.copy()

                        # --- MODIFIED: ใช้ dot_color จาก state ---
                        dot_color = st.get('dot_color', (0, 0, 255)) # Default สีแดง
                        cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,255,0),2)
                        cv2.putText(frame,f'PID:{pid} ({st["state"]})',(bbox[0],max(20,bbox[1]-5)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                        cv2.circle(frame,(int(cur_pos[0]),int(cur_pos[1])),5, dot_color,-1) # ใช้ dot_color
                        # --- END MODIFIED ---

                    # --- Process Disappeared People & Cleanup ---
                    pids_to_remove = set()
                    retention_seconds = getattr(cfg, 'STATE_RETENTION_S', 10.0)
                    for pid, st in person_states.items():
                        if pid not in processed_pids_this_frame:
                            if st['state'] == 'crossed_red':
                                 counts['inbound'] += 1
                                 
                                 # --- MODIFIED: ดึงเวลาตอน "ข้ามเส้น" มาใช้ ---
                                 log_time_sec = st.get('cross_time_sec', current_video_sec) # 1. ดึงเวลาที่เก็บไว้ (ถ้าไม่มีจริงๆ ค่อยใช้เวลาปัจจุบัน)
                                 video_time_str = format_seconds(log_time_sec, video_hour) 
                                 # --- END MODIFIED ---
                                 
                                 print(f"PID {pid}: Exited -> COUNT = {counts['inbound']} (Video Time: {video_time_str})")
                                 
                                 # --- NEW: เพิ่ม hour_offset ใน Log ---
                                 hour_offset_str = str(args.video_hour) if args.video_hour is not None else ""
                                 csvw.writerow([video_time_str, args.camera_name, pid, 'entrance', hour_offset_str])
                                 # --- END NEW ---
                                 
                                 last_frame_s = st.get('last_frame_seen')
                                 if last_frame_s is not None:
                                      frame_s = last_frame_s.copy()
                                      cv2.polylines(frame_s, [np.array(pink_zone, dtype=np.int32)], isClosed=True, color=(255, 182, 193), thickness=2)
                                      cv2.line(frame_s, red_line[0], red_line[1], (0,0,255), 2)
                                      cv2.line(frame_s, blue_line[0], blue_line[1], (255,0,0), 2)
                                      cv2.line(frame_s, green_line[0], green_line[1], (0,255,0), 2)
                                      cv2.line(frame_s, yellow_line[0], yellow_line[1], (0,255,255), 2)
                                      last_bb = st.get('last_bbox')
                                      if last_bb: cv2.rectangle(frame_s,(last_bb[0],last_bb[1]),(last_bb[2],last_bb[3]),(0,255,0),3)
                                      video_time_fname = f"{int(current_video_sec // 3600):02d}h{int((current_video_sec % 3600) // 60):02d}m{int(current_video_sec % 60):02d}s"
                                      snap_f = os.path.join(person_snapshot_dir, f"inbound_pid{pid}_{video_time_fname}.jpg")
                                      cv2.imwrite(snap_f, frame_s); print(f"Saved snapshot: {os.path.basename(snap_f)}")
                                      st['state'] = 'counted'
                                 else: print(f"Warn: No snapshot for PID {pid}.")
                            
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
                
                # --- UI Display ---
                cv2.polylines(frame, [np.array(pink_zone, dtype=np.int32)], isClosed=True, color=(255, 182, 193), thickness=2)
                cv2.line(frame, red_line[0], red_line[1], (0,0,255), 2)
                cv2.line(frame, blue_line[0], blue_line[1], (255,0,0), 2)
                cv2.line(frame, green_line[0], green_line[1], (0,255,0), 2)
                cv2.line(frame, yellow_line[0], yellow_line[1], (0,255,255), 2)
                
                # --- MODIFIED: เพิ่ม Video Time (HH:MM:SS) ใต้ Inbound ---
                inbound_text = f"Entrance: {counts['inbound']}" # แก้ไขคำว่า "Extrance"
                video_time_text = f"Video Time: {format_seconds(current_video_sec, video_hour)}"
                cv2.putText(frame, inbound_text, (10, original_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, video_time_text, (10, original_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) # แสดงเวลาด้านล่าง
                # --- END MODIFIED ---
                
                if display_timestamp_str:
                     try:
                          font_scale=0.6; thickness=1; font=cv2.FONT_HERSHEY_SIMPLEX; text_x,text_y=10,30
                          (tw,th),bl=cv2.getTextSize(display_timestamp_str,font,font_scale,thickness)
                          pad=5; bx1=max(text_x-pad,0); by1=max(text_y-th-pad-bl,0); bx2=min(text_x+tw+pad,frame.shape[1]); by2=min(text_y+pad,frame.shape[0])
                          if by2>by1 and bx2>bx1: cv2.rectangle(frame,(bx1,by1),(bx2,by2),(0,0,0),-1)
                     except: pass
                cv2.putText(frame, display_timestamp_str, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
                
                if mouse_pos_raw[0]>=0 and paused:
                    cv2.drawMarker(frame,(mouse_pos_raw[0],mouse_pos_raw[1]),(0,255,255), cv2.MARKER_CROSS,20,2)
                    text=f"x:{mouse_pos_raw[0]} y:{mouse_pos_raw[1]}"; cv2.putText(frame,text,(mouse_pos_raw[0]+15,mouse_pos_raw[1]-15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

                cv2.imshow('Video Analysis', cv2.resize(frame, (display_width, display_height)))
                k = cv2.waitKey(10) & 0xFF
                if k == 27: break
                elif k == ord('p'): paused = not paused
            
    except KeyboardInterrupt:
        print("\nUser interrupted process (Ctrl+C).")
        # --- NEW: บันทึกเวลา OCR สุดท้าย แม้จะกด Ctrl+C ---
        try:
             video_end_time_processed = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        except:
             pass # ถ้า cap ปิดไปแล้ว
        # --- END NEW ---
    finally:
        # --- บันทึก Summary Log ---
        print("\n--- Writing Summary Log ---")
        try:
            with open(summary_log_path, "w", newline="", encoding='utf-8') as summary_f:
                summary_csvw = csv.writer(summary_f, delimiter=',')
                # --- NEW: เพิ่ม Header Hour Offset ---
                summary_csvw.writerow(["Camera Name", "Total Inbound", "Video Start Time Processed (HH:MM:SS)", "Video End Time Processed (HH:MM:SS)", "Run Timestamp", "Manual Hour Offset"])
                # --- END NEW ---
                
                start_str = format_seconds(video_start_time_processed, video_hour)
                end_str = format_seconds(video_end_time_processed, video_hour)
                hour_offset_str = str(args.video_hour) if args.video_hour is not None else ""
                
                # --- NEW: เพิ่ม Data Hour Offset ---
                summary_csvw.writerow([args.camera_name, counts["inbound"], start_str, end_str, current_run_timestamp, hour_offset_str])
                # --- END NEW ---
            print(f"Saved summary log to: {summary_log_path}")
        except Exception as e: print(f"Error writing summary log: {e}")
        
        if 'csv_file' in locals() and csv_file is not None and not csv_file.closed:
            csv_file.close()
            print("Closed event log file.")
        cap.release()
        cv2.destroyAllWindows()
        print("Process finished.")

if __name__ == "__main__":
    main()