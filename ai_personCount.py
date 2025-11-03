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
def format_seconds(seconds):
    """แปลงวินาที (float) เป็น string 'HH:MM:SS'"""
    if seconds is None: return "N/A"
    return str(timedelta(seconds=int(seconds)))
# --- END NEW ---

# --- REMOVED: setup_interval_output function ---

# ====================== MAIN LOGIC =========================
# ====================== MAIN LOGIC =========================
def main():
    # --- MODIFIED: เพิ่ม Arguments สำหรับ Time Range (ใช้ นาที) ---
    parser = argparse.ArgumentParser(description="Person Counter (Summary Log + Time Range)")
    parser.add_argument("camera_name", help="Name of the camera config.")
    parser.add_argument("--start_min", type=int, default=0, help="Start processing at this minute in the video (default: 0)")
    parser.add_argument("--duration_min", type=int, default=None, help="Process for this many minutes (default: process until end of video)")
    args = parser.parse_args()
    # --- END MODIFIED ---

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
    
    # --- NEW: ตัวแปรสำหรับ Summary Log ---
    video_start_time_processed = None # เวลา (วิดีโอ) ที่เริ่มประมวลผล
    video_end_time_processed = None   # เวลา (วิดีโอ) ที่สิ้นสุดการประมวลผล
    # --- END NEW ---
    
    last_frame = None

    try:
        with open(event_log_path, "w", newline="", encoding='utf-8') as csv_file:
            csvw = csv.writer(csv_file, delimiter=','); csvw.writerow(["Video Time (HH:MM:SS)","Camera Name","PID","Status"])

            while True:
                current_time_dt = datetime.now() # เวลาปัจจุบันของเครื่อง
                
                # --- MODIFIED: แก้ไข Logic การอ่านเฟรมและ Pause (FIXED) ---
                if not paused:
                    ret, frame = cap.read() # <-- 1. อ่านเฟรมใหม่ "เสมอ" ถ้าไม่ Pause
                    if not ret: break
                    last_frame = frame.copy() # <-- 2. เก็บเฟรมล่าสุดไว้
                
                if last_frame is None: continue # (เผื่อเฟรมแรกอ่านไม่ได้)
                
                frame = last_frame.copy() # <-- 3. ใช้ "frame" เป็นสำเนาของเฟรมล่าสุดเสมอ
                # --- END MODIFIED ---

                # --- Get Video Time ---
                current_video_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                current_video_sec = current_video_msec / 1000.0
                
                ocr_timestamp_dt = get_timestamp_from_frame(frame, timestamp_roi)
                display_timestamp_str = ocr_timestamp_dt.strftime('%d-%m-%Y %H:%M:%S') if ocr_timestamp_dt else ""

                # --- NEW: Time Range Check (ย้ายมาไว้หลังอ่านเฟรม) ---
                process_this_frame = True # ตั้งค่าเริ่มต้น
                
                if current_video_sec < (args.start_min * 60):
                    process_this_frame = False # ยังไม่ถึงเวลา
                
                elif video_start_time_processed is None: # ถ้าเพิ่งเข้าสู่ช่วงเวลาประมวลผล
                    video_start_time_processed = current_video_sec
                    print(f"Processing started at video time: {format_seconds(video_start_time_processed)}")

                # ถ้ากำหนด --duration_min และประมวลผลครบแล้ว
                if args.duration_min is not None and video_start_time_processed is not None and \
                   (current_video_sec - video_start_time_processed) > (args.duration_min * 60):
                    print(f"Processing duration of {args.duration_min} minutes reached. Stopping.")
                    break # หยุด Loop

                if process_this_frame:
                     video_end_time_processed = current_video_sec # อัปเดตเวลาสิ้นสุดที่ประมวลผล (อัปเดตเรื่อยๆ)
                # --- END NEW ---

                # --- Detection, Tracking, State Machine (ต้องอยู่ข้างใน if) ---
                if process_this_frame:
                    dets=[]; tracks=np.empty((0,5)); valid_results=False
                    results = model(frame, stream=True, conf=cfg.SCORE_THR)
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
                            person_states[pid] = {'state': 'waiting', 'sign_history': deque(maxlen=SIGN_HISTORY_LENGTH), 'last_frame_seen': frame.copy(), 'last_bbox': bbox, 'last_pos': cur_pos, 'last_tid': tid, 'last_seen_time': current_time_dt}
                        st = person_states[pid]
                        st['tid'] = tid; st['last_bbox'] = bbox; st['last_frame_seen'] = frame.copy()
                        st['last_pos'] = cur_pos; st['last_seen_time'] = current_time_dt
                        processed_pids_this_frame.add(pid)
                        current_sign = _cross_sign(cur_pos, red_line[0], red_line[1])
                        if current_sign != 0: st['sign_history'].append(current_sign)
                        history = list(st['sign_history'])
                        crossed_top_to_bottom=False; crossed_bottom_to_top=False
                        if len(history)>=2:
                            prev_s, last_s = history[-2], history[-1]
                            if prev_s==top_sign and last_s==bottom_sign: crossed_top_to_bottom=True
                            elif prev_s==bottom_sign and last_s==top_sign: crossed_bottom_to_top=True
                        if st['state'] == 'waiting' and crossed_top_to_bottom: st['state']='crossed_red'
                        elif st['state'] == 'crossed_red' and crossed_bottom_to_top: st['state']='waiting'
                        
                        # --- Drawing ---
                        cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,255,0),2)
                        cv2.putText(frame,f'PID:{pid} ({st["state"]})',(bbox[0],max(20,bbox[1]-5)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                        cv2.circle(frame,(int(cur_pos[0]),int(cur_pos[1])),5,(0,0,255),-1)

                    # --- Process Disappeared People & Cleanup ---
                    pids_to_remove = set()
                    retention_seconds = getattr(cfg, 'STATE_RETENTION_S', 10.0)
                    for pid, st in person_states.items():
                        if pid not in processed_pids_this_frame:
                            if st['state'] == 'crossed_red':
                                 counts['inbound'] += 1
                                 video_time_str = format_seconds(current_video_sec)
                                 print(f"PID {pid}: Exited -> COUNT = {counts['inbound']} (Video Time: {video_time_str})")
                                 csvw.writerow([video_time_str, args.camera_name, pid, 'entrance'])
                                 
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
                
                # --- UI Display (อยู่นอก if process_this_frame) ---
                cv2.polylines(frame, [np.array(pink_zone, dtype=np.int32)], isClosed=True, color=(255, 182, 193), thickness=2)
                cv2.line(frame, red_line[0], red_line[1], (0,0,255), 2)
                cv2.line(frame, blue_line[0], blue_line[1], (255,0,0), 2)
                cv2.line(frame, green_line[0], green_line[1], (0,255,0), 2)
                cv2.line(frame, yellow_line[0], yellow_line[1], (0,255,255), 2)
                cv2.putText(frame, f"Inbound: {counts['inbound']}", (10, original_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
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
                
                # --- MODIFIED: เพิ่มเวลารอให้ UI ---
                k = cv2.waitKey(10) & 0xFF # เพิ่มจาก 1 เป็น 10 (หรือ 25)
                # --- END MODIFIED ---
                
                if k == 27:
                    break
                elif k == ord('p'): 
                    paused = not paused
            
    except KeyboardInterrupt:
        print("\nUser interrupted process (Ctrl+C).")
    finally:
        # --- NEW: บันทึก Summary Log ---
        print("\n--- Writing Summary Log ---")
        try:
            with open(summary_log_path, "w", newline="", encoding='utf-8') as summary_f:
                summary_csvw = csv.writer(summary_f, delimiter=',')
                summary_csvw.writerow(["Camera Name", "Total Inbound", "Video Start Time Processed (HH:MM:SS)", "Video End Time Processed (HH:MM:SS)", "Run Timestamp"])
                
                start_str = format_seconds(video_start_time_processed)
                end_str = format_seconds(video_end_time_processed)
                
                summary_csvw.writerow([args.camera_name, counts["inbound"], start_str, end_str, current_run_timestamp])
            print(f"Saved summary log to: {summary_log_path}")
        except Exception as e:
            print(f"Error writing summary log: {e}")
        # --- END NEW ---
        
        if 'csv_file' in locals() and csv_file is not None and not csv_file.closed:
            csv_file.close()
            print("Closed event log file.")
        cap.release()
        cv2.destroyAllWindows()
        print("Process finished.")

if __name__ == "__main__":
    main()