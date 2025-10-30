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
BASE_OUTPUT_DIR = "qa_camera_check" # เปลี่ยนชื่อโฟลเดอร์หลัก
SIGN_HISTORY_LENGTH = 3
INTERVAL_MINUTES = 5 # กำหนดช่วงเวลาเป็น 5 นาที

# --- Tesseract ---
try:
    import pytesseract
    # TESSERACT_PATH = r'...'
    # pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
except ImportError: pytesseract = None; print("Warn: pytesseract not found.")

# =================== MODEL / TRACKER ====================
# ... (ส่วน Model/Tracker เหมือนเดิม) ...
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
# ... (ฟังก์ชัน _cross_sign, make_side_label, get_timestamp_from_frame, ensure_dir เหมือนเดิม) ...
def _cross_sign(p, a, b):
    try: p_arr=np.array(p,dtype=np.float64); a_arr=np.array(a,dtype=np.float64); b_arr=np.array(b,dtype=np.float64)
    except: return 0
    val = (b_arr[0]-a_arr[0])*(p_arr[1]-a_arr[1])-(b_arr[1]-a_arr[1])*(p_arr[0]-a_arr[0])
    return 0 if abs(val)<1e-9 else int(np.sign(val))

def make_side_label(a, b):
    a,b=np.array(a),np.array(b); mid_below=(a+b)/2.0+np.array([0,100]); return _cross_sign(mid_below,a,b)<0

def get_timestamp_from_frame(frame, roi):
    if pytesseract is None or roi is None: return None
    try:
        x1,y1,x2,y2=roi; h,w,_=frame.shape; x1,y1=max(0,x1),max(0,y1); x2,y2=min(w,x2),min(h,y2)
        if y2<=y1 or x2<=x1: return None
        ts_img=frame[y1:y2,x1:x2]; gray=cv2.cvtColor(ts_img,cv2.COLOR_BGR2GRAY)
        binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,5)
        text=pytesseract.image_to_string(binary,config=r'--oem 3 --psm 6')
        match=re.search(r'(\d{2})-(\d{4}).*?(\d{2}:\d{2}:\d{2})',text.replace(" ",""))
        if match: m,y,t=match.groups(); return datetime.strptime(f"01-{m}-{y} {t}",'%d-%m-%Y %H:%M:%S')
    except: return None
    return None

def ensure_dir(dir_path):
    if not os.path.exists(dir_path): os.makedirs(dir_path); print(f"Created directory: {dir_path}")

# --- NEW: Helper สำหรับจัดการ Output ของแต่ละ Interval ---
def setup_interval_output(base_output_dir, camera_name, interval_idx):
    """สร้าง Directory และเปิดไฟล์ Log สำหรับ Interval ใหม่"""
    interval_minutes = (interval_idx + 1) * INTERVAL_MINUTES
    interval_label = f"{interval_minutes:02d}Min" # e.g., "05Min", "10Min"
    
    interval_dir = os.path.join(base_output_dir, camera_name, interval_label)
    log_dir = os.path.join(interval_dir, "logs")
    snapshot_dir = os.path.join(interval_dir, "person_snapshots")
    
    ensure_dir(log_dir)
    ensure_dir(snapshot_dir)
    
    log_path = os.path.join(log_dir, f"log_{camera_name}_{interval_label}.csv")
    csv_file = open(log_path, "w", newline="", encoding='utf-8')
    csv_writer = csv.writer(csv_file, delimiter=',')
    csv_writer.writerow(["Date", "Time", "Camera Name", "PID", "Status"])
    
    print(f"--- Starting Interval {interval_label} ---")
    print(f"Logging to: {log_path}")
    print(f"Snapshots to: {snapshot_dir}")
    
    return log_path, csv_file, csv_writer, snapshot_dir
# --- END NEW ---

# ====================== MAIN LOGIC =========================
def main():
    parser = argparse.ArgumentParser(description="Person Counter (Interval Logging)")
    parser.add_argument("camera_name", help="Name of the camera config.")
    args = parser.parse_args()

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
    pink_zone=tuple(map(tuple,config['pink_zone']))
    timestamp_roi=config.get('timestamp_roi')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise IOError(f"Cannot open video: {video_path}")
    original_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); original_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if original_w==0 or original_h==0: raise IOError("Could not read video dimensions.")
    aspect=original_w/max(1,original_h); display_height=int(display_width/aspect)

    cv2.namedWindow("Video Analysis", cv2.WINDOW_NORMAL)
    paused=False; mouse_pos_raw=(-1,-1)
    counts={"inbound":0}; person_states={}; next_pid=1 # person_states persists across intervals
    tid_to_pid = {}
    
    neg_is_bottom_red=make_side_label(red_line[0],red_line[1])
    bottom_sign=-1 if neg_is_bottom_red else 1; top_sign=-bottom_sign

    # --- Mouse Callback ---
    def _on_mouse(event, x, y, flags, param):
        nonlocal mouse_pos_raw # <--- ย้ายลงมา + ย่อหน้า
        rx = int(x * original_w / display_width)
        ry = int(y * original_h / display_height)
        mouse_pos_raw = (rx, ry)
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked: ({rx}, {ry})")
    cv2.setMouseCallback("Video Analysis", _on_mouse)

    # --- Interval Management Variables ---
    current_interval_idx = -1
    csv_file = None
    csvw = None
    current_snapshot_dir = None
    # ---

    last_frame = None

    try: # Add try block for proper file closing
        while True:
            current_time_dt = datetime.now() # System time for general use
            
            # --- Get Video Time ---
            current_video_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            current_video_sec = current_video_msec / 1000.0

            # --- Interval Check ---
            interval_duration_sec = INTERVAL_MINUTES * 60
            expected_interval_idx = int(current_video_sec // interval_duration_sec)

            # ถ้าข้าม Interval ใหม่
            if expected_interval_idx != current_interval_idx:
                # ปิดไฟล์ Log เก่า (ถ้ามี)
                if csv_file is not None and not csv_file.closed:
                    csv_file.close()
                    print(f"Closed log file for interval {current_interval_idx * INTERVAL_MINUTES}-{(current_interval_idx + 1) * INTERVAL_MINUTES} Min")
                
                # ตั้งค่า Output สำหรับ Interval ใหม่
                _, csv_file, csvw, current_snapshot_dir = setup_interval_output(
                    BASE_OUTPUT_DIR, args.camera_name, expected_interval_idx
                )
                current_interval_idx = expected_interval_idx

            # ถ้ายังไม่ได้ตั้งค่า Interval แรก (กรณีเริ่มวิดีโอ)
            if current_interval_idx == -1:
                 _, csv_file, csvw, current_snapshot_dir = setup_interval_output(
                    BASE_OUTPUT_DIR, args.camera_name, 0
                )
                 current_interval_idx = 0

            # --- Read Frame (Pause Logic) ---
            if not paused:
                ret, frame = cap.read()
                if not ret: break
                last_frame = frame.copy()
            if last_frame is None: continue
            frame = last_frame.copy()

            ocr_timestamp_dt = get_timestamp_from_frame(frame, timestamp_roi)
            display_timestamp_str = ocr_timestamp_dt.strftime('%d-%m-%Y %H:%M:%S') if ocr_timestamp_dt else ""

            # --- Detection & Tracking ---
            dets=[]; tracks=np.empty((0,5)); valid_results=False
            results = model(frame, stream=True, conf=cfg.SCORE_THR)
            for r in results:
                valid_results=True;
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
                    # print(f"New PID: {pid} (TID {tid})") # Debug

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
                # ไม่ต้องเก็บ prev_pos แล้ว

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
                         # ตรวจสอบว่า csvw ถูกต้องหรือไม่ ก่อนเขียน
                         if csvw is not None and current_snapshot_dir is not None:
                              counts['inbound'] += 1
                              print(f"PID {pid}: Exited -> COUNT = {counts['inbound']}")
                              log_dt = ocr_timestamp_dt if ocr_timestamp_dt else current_time_dt
                              log_date=log_dt.strftime('%Y-%m-%d'); log_time=log_dt.strftime('%H:%M:%S')
                              csvw.writerow([log_date, log_time, args.camera_name, pid, 'entrance'])

                              last_frame_s = st.get('last_frame_seen')
                              if last_frame_s is not None:
                                   frame_s = last_frame_s.copy()
                                   cv2.rectangle(frame_s, pink_zone[0], pink_zone[1], (255,182,193), 2)
                                   # ... (วาดเส้นอื่นๆ) ...
                                   cv2.line(frame_s, red_line[0], red_line[1], (0,0,255), 2)
                                   cv2.line(frame_s, blue_line[0], blue_line[1], (255,0,0), 2)
                                   cv2.line(frame_s, green_line[0], green_line[1], (0,255,0), 2)
                                   cv2.line(frame_s, yellow_line[0], yellow_line[1], (0,255,255), 2)

                                   last_bb = st.get('last_bbox')
                                   if last_bb: cv2.rectangle(frame_s,(last_bb[0],last_bb[1]),(last_bb[2],last_bb[3]),(0,255,0),3)
                                   snap_t = ocr_timestamp_dt if ocr_timestamp_dt else current_time_dt
                                   snap_f = os.path.join(current_snapshot_dir, f"inbound_pid{pid}_{snap_t.strftime('%Y%m%d_%H%M%S')}.jpg")
                                   cv2.imwrite(snap_f, frame_s); print(f"Saved snapshot: {os.path.basename(snap_f)}")
                                   st['state'] = 'counted' # Mark counted only after successful log/snap
                              else: print(f"Warn: No snapshot for PID {pid}.")
                         else: print(f"Warn: Log file writer/Snapshot dir not ready for PID {pid}.")

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
                    # print(f"Removed expired PID: {pid}") # Debug

            # --- UI Display ---
            cv2.rectangle(frame, pink_zone[0], pink_zone[1], (255,182,193), 2)
            # ... (วาดเส้นและ Text อื่นๆ) ...
            cv2.line(frame, red_line[0], red_line[1], (0,0,255), 2)
            cv2.line(frame, blue_line[0], blue_line[1], (255,0,0), 2)
            cv2.line(frame, green_line[0], green_line[1], (0,255,0), 2)
            cv2.line(frame, yellow_line[0], yellow_line[1], (0,255,255), 2)
            cv2.putText(frame, f"Inbound: {counts['inbound']}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            if display_timestamp_str:
                 try:
                      font_scale=0.6; thickness=1; font=cv2.FONT_HERSHEY_SIMPLEX; text_x,text_y=10,30
                      (tw,th),bl=cv2.getTextSize(display_timestamp_str,font,font_scale,thickness)
                      pad=5; bx1=max(text_x-pad,0); by1=max(text_y-th-pad-bl,0); bx2=min(text_x+tw+pad,frame.shape[1]); by2=min(text_y+pad,frame.shape[0])
                      if by2>by1 and bx2>bx1: cv2.rectangle(frame,(bx1,by1),(bx2,by2),(0,0,0),-1)
                 except: pass
            cv2.putText(frame, display_timestamp_str, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)

            if mouse_pos_raw[0]>=0:
                cv2.drawMarker(frame,(mouse_pos_raw[0],mouse_pos_raw[1]),(0,255,255), cv2.MARKER_CROSS,20,2)
                text=f"x:{mouse_pos_raw[0]} y:{mouse_pos_raw[1]}"; cv2.putText(frame,text,(mouse_pos_raw[0]+15,mouse_pos_raw[1]-15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

            cv2.imshow('Video Analysis', cv2.resize(frame, (display_width, display_height)))
            k = cv2.waitKey(1) & 0xFF
            if k == 27: break
            elif k == ord('p'): paused = not paused
            
    finally: # Ensure file is closed even if error occurs
        if csv_file is not None and not csv_file.closed:
            csv_file.close()
            print("Closed final log file.")
        cap.release()
        cv2.destroyAllWindows()
        print("Process finished.")

if __name__ == "__main__":
    main()