import os
import cv2
import csv
import numpy as np
import json
import argparse
from datetime import datetime, timedelta # เพิ่ม timedelta

# --- Dependencies ---
from ultralytics import YOLO
from sort import Sort
# --- FIX: ตรวจสอบตำแหน่ง config/model_config ---
# หาก model_config.py อยู่ในโฟลเดอร์ config ให้ใช้ from config import model_config as cfg
# หาก model_config.py อยู่ที่เดียวกับ final_person_counter.py ให้ใช้ import model_config as cfg
try:
    from config import model_config as cfg
except ImportError:
    try:
        import model_config as cfg
    except ImportError:
        # ถ้าหาไม่เจอจริงๆ ใช้ค่า default ไปก่อน
        class DefaultConfig:
            MAX_AGE_FRAMES = 120
            SCORE_THR = 0.35
        cfg = DefaultConfig()
        print("Warning: model_config.py not found. Using default values.")


# --- การตั้งค่าที่สำคัญ ---
CONFIG_FILE = 'config/camera_config.json' # ตรวจสอบ path นี้ให้ถูกต้อง
BASE_EVENT_DIR = "qa_events"  # โฟลเดอร์หลักสำหรับเก็บผลลัพธ์
PERIODIC_SNAPSHOT_INTERVAL = timedelta(minutes=10) # ช่วงเวลา Snapshot กล้อง

# =================== MODEL / TRACKER (Global) ====================
print("Loading AI model...")
# --- FIX: ตรวจสอบตำแหน่ง model weights ---
# หาก yolov8m.pt อยู่ในโฟลเดอร์ core ให้ใช้ core/yolov8m.pt
# หากอยู่ที่เดียวกับ script ให้ใช้ yolov8m.pt
model_path = "core/yolov8m.pt" if os.path.exists(os.path.join("core", "yolov8m.pt")) else "yolov8m.pt"
if not os.path.exists(model_path):
     model_path_n = "yolov8n.pt" # Fallback ไป nano ถ้าหา m ไม่เจอ
     model_path_n_core = os.path.join("core", "yolov8n.pt")
     if os.path.exists(model_path_n):
         model_path = model_path_n
         print(f"Warning: {model_path} not found. Trying yolov8n.pt")
     elif os.path.exists(model_path_n_core):
          model_path = model_path_n_core
          print(f"Warning: {model_path} not found. Trying core/yolov8n.pt")
     else:
          raise FileNotFoundError("Could not find yolov8m.pt or yolov8n.pt in root or core directory")

model = YOLO(model_path, verbose=False)
tracker = Sort(max_age=cfg.MAX_AGE_FRAMES, min_hits=3, iou_threshold=0.2)
print("Model loaded successfully.")

# ====================== GEOMETRY HELPERS =========================
def _cross_sign(p, a, b):
    # ตรวจสอบชนิดข้อมูลและแปลงเป็น float หากจำเป็น
    try:
        p_arr = np.array(p, dtype=np.float64)
        a_arr = np.array(a, dtype=np.float64)
        b_arr = np.array(b, dtype=np.float64)
    except (ValueError, TypeError):
        return 0 # Return neutral sign if conversion fails

    val = (b_arr[0] - a_arr[0]) * (p_arr[1] - a_arr[1]) - (b_arr[1] - a_arr[1]) * (p_arr[0] - a_arr[0])
    tolerance = 1e-9
    if abs(val) < tolerance: return 0
    return np.sign(val)

def is_crossing_line(p1, p2, a, b):
    """ตรวจสอบว่าเส้นจาก p1 ไป p2 ตัดกับเส้น a ไป b หรือไม่ (รองรับกรณีจุดอยู่บนเส้น)"""
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

# ====================== FILE SYSTEM HELPER =========================
def ensure_dir(dir_path):
    """สร้าง Directory ถ้ายังไม่มี"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

# ====================== MAIN LOGIC =========================
def main():
    parser = argparse.ArgumentParser(description="Final Person Counting Tool with Snapshots")
    parser.add_argument("camera_name", help="Name of the camera config to use.")
    args = parser.parse_args()

    # --- Load Config ---
    try:
        with open(CONFIG_FILE, "r", encoding='utf-8') as f:
            full_config = json.load(f)
    except FileNotFoundError:
         raise SystemExit(f"Error: Config file '{CONFIG_FILE}' not found.")
         
    if args.camera_name not in full_config:
        raise SystemExit(f"Camera '{args.camera_name}' not found in config.")
    config = full_config[args.camera_name]

    video_path = config.get('video_path')
    if not video_path: raise SystemExit("Error: 'video_path' not found in config.")
    
    display_width = config.get('display_width', 1280)
    red_line = tuple(map(tuple, config['lines']['red']))
    blue_line = tuple(map(tuple, config['lines']['blue']))
    green_line = tuple(map(tuple, config['lines']['green']))
    yellow_line = tuple(map(tuple, config['lines']['yellow']))
    pink_zone = tuple(map(tuple, config['pink_zone'])) # โหลด pink_zone

    # --- กำหนด Paths สำหรับ Logs และ Snapshots ---
    camera_event_dir = os.path.join(BASE_EVENT_DIR, args.camera_name)
    log_dir = os.path.join(camera_event_dir, "logs")
    periodic_snapshot_dir = os.path.join(camera_event_dir, "periodic_snapshots")
    person_snapshot_dir = os.path.join(camera_event_dir, "person_snapshots")

    ensure_dir(log_dir)
    ensure_dir(periodic_snapshot_dir)
    ensure_dir(person_snapshot_dir)

    local_log_path = os.path.join(log_dir, f"log_{args.camera_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise IOError(f"Cannot open video file: {video_path}")

    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if original_w == 0 or original_h == 0:
        raise IOError("Could not read video dimensions. Check video file/codecs.")
        
    aspect = original_w / max(1, original_h)
    display_height = int(display_width / aspect)

    cv2.namedWindow("Video Analysis", cv2.WINDOW_NORMAL)

    paused = False
    mouse_pos_raw = (-1, -1)

    counts = {"inbound": 0}
    person_states = {}
    next_pid = 1

    last_periodic_snapshot_time = datetime.min

    # --- Mouse Callback ---
    def _on_mouse(event, x, y, flags, param):
        nonlocal mouse_pos_raw
        rx = int(x * original_w / display_width)
        ry = int(y * original_h / display_height)
        mouse_pos_raw = (rx, ry)
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked Coordinates: ({rx}, {ry})")
    cv2.setMouseCallback("Video Analysis", _on_mouse)

    # --- เปิดไฟล์ Log ---
    with open(local_log_path, "w", newline="", encoding='utf-8') as csv_file:
        csvw = csv.writer(csv_file, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvw.writerow(["Date", "Time", "PID", "Status"])
        
        last_frame = None

        while True:
            current_time = datetime.now()

            # --- Pause Logic ---
            if not paused:
                ret, frame = cap.read()
                if not ret: break
                last_frame = frame.copy()
            if last_frame is None: continue
            frame = last_frame.copy()

            # --- Periodic Snapshot Logic ---
            if current_time - last_periodic_snapshot_time >= PERIODIC_SNAPSHOT_INTERVAL:
                snapshot_filename = os.path.join(periodic_snapshot_dir, f"snapshot_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg")
                cv2.imwrite(snapshot_filename, frame)
                print(f"Saved periodic snapshot: {os.path.basename(snapshot_filename)}")
                last_periodic_snapshot_time = current_time

            timestamp_date_str = current_time.strftime('%Y-%m-%d')
            timestamp_time_str = current_time.strftime('%H:%M:%S')

            # --- Detection & Tracking ---
            dets = []
            for r in model(frame, stream=True, conf=cfg.SCORE_THR):
                for box in r.boxes.data:
                    if len(box) >= 6 and int(box[5]) == 0:
                        dets.append([int(b) for b in box[:4]] + [float(box[4])])
            tracks = tracker.update(np.array(dets) if dets else np.empty((0, 5)))
            live_tids = {int(t[4]) for t in tracks}

            # --- State Machine Logic (ใช้ Top-Center และ is_crossing_line ที่แก้ไขแล้ว) ---
            for x1, y1, x2, y2, tid in tracks:
                tid, bbox = int(tid), (int(x1), int(y1), int(x2), int(y2))
                cur_pos = np.array([(x1 + x2) / 2, y1]) # Top-Center Point

                pid = next((p for p, s in person_states.items() if s.get('tid') == tid), None)
                if pid is None:
                    pid = next_pid; next_pid += 1
                    person_states[pid] = {'tid': tid, 'state': 'waiting', 'prev_pos': None, 'last_bbox': None}

                st = person_states[pid]
                st['tid'] = tid
                st['last_bbox'] = bbox # เก็บ Bbox ล่าสุดเสมอ
                prev_pos = st.get('prev_pos')

                crossed = is_crossing_line(prev_pos, cur_pos, red_line[0], red_line[1])

                if st['state'] == 'waiting':
                    if crossed and prev_pos is not None and cur_pos[1] > prev_pos[1]:
                        st['state'] = 'crossed_red'
                elif st['state'] == 'crossed_red':
                    if crossed and prev_pos is not None and cur_pos[1] < prev_pos[1]:
                        st['state'] = 'waiting'

                st['prev_pos'] = cur_pos.copy()

                # --- Drawing ---
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
                cv2.putText(frame, f'PID:{pid} ({st["state"]})', (bbox[0], max(20, bbox[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.circle(frame, (int(cur_pos[0]), int(cur_pos[1])), 5, (0, 0, 255), -1)

            # --- Check Disappeared People ---
            disappeared_pids = [pid for pid, st in person_states.items() if st.get('tid') not in live_tids]
            for pid in disappeared_pids:
                st = person_states[pid]
                if st['state'] == 'crossed_red':
                    counts['inbound'] += 1
                    print(f"PID {pid}: Exited frame after crossing red. COUNT = {counts['inbound']}")
                    
                    csvw.writerow([timestamp_date_str, timestamp_time_str, pid, 'entrance'])
                    
                    # --- Full Frame Snapshot Logic ---
                    # ไม่ต้องเช็ค last_bbox อีก เพราะเราจะ save frame ปัจจุบันที่หายไป
                    # สร้างสำเนาของเฟรมปัจจุบันเพื่อวาดเส้นทับ
                    # ใช้ last_frame ที่เก็บไว้ เพราะ frame ปัจจุบันคือเฟรมที่คนหายไปแล้ว
                    if last_frame is not None: 
                        frame_with_lines = last_frame.copy() 
                        
                        # วาดเส้น Boundary ต่างๆ ลงบนสำเนาเฟรม
                        cv2.rectangle(frame_with_lines, pink_zone[0], pink_zone[1], (255, 182, 193), 2) # วาดกรอบชมพู
                        cv2.line(frame_with_lines, red_line[0], red_line[1], (0, 0, 255), 2)      # Top = Red
                        cv2.line(frame_with_lines, blue_line[0], blue_line[1], (255, 0, 0), 2)     # Left = Blue
                        cv2.line(frame_with_lines, green_line[0], green_line[1], (0, 255, 0), 2)   # Right = Green
                        cv2.line(frame_with_lines, yellow_line[0], yellow_line[1], (0, 255, 255), 2) # Bottom = Yellow

                        # (Optional) วาด Bbox ของคนที่หายไปด้วย ถ้าต้องการ (ใช้ last_bbox ที่เก็บไว้)
                        last_bbox = st.get('last_bbox')
                        if last_bbox:
                             cv2.rectangle(frame_with_lines, (last_bbox[0], last_bbox[1]), (last_bbox[2], last_bbox[3]), (0, 255, 0), 3) # สีเขียว หนาๆ

                        # สร้างชื่อไฟล์ Snapshot
                        snapshot_filename = os.path.join(person_snapshot_dir, f"inbound_pid{pid}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg")
                        
                        # บันทึกเฟรมเต็มที่มีเส้นแล้ว
                        cv2.imwrite(snapshot_filename, frame_with_lines) 
                        print(f"Saved inbound snapshot: {os.path.basename(snapshot_filename)}")
                    else:
                        print(f"Warning: Could not save snapshot for PID {pid}, last frame not available.")
                    
                    st['state'] = 'counted'
                elif st['state'] != 'counted':
                    st['state'] = 'waiting'

            # --- UI Display ---
            cv2.rectangle(frame, pink_zone[0], pink_zone[1], (255, 182, 193), 2) # วาดกรอบชมพู
            cv2.line(frame, red_line[0], red_line[1], (0, 0, 255), 2)
            cv2.line(frame, blue_line[0], blue_line[1], (255, 0, 0), 2)
            cv2.line(frame, green_line[0], green_line[1], (0, 255, 0), 2)
            cv2.line(frame, yellow_line[0], yellow_line[1], (0, 255, 255), 2)
            cv2.putText(frame, f"Inbound: {counts['inbound']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # --- Mouse Hover Overlay ---
            if mouse_pos_raw[0] >= 0:
                cv2.drawMarker(frame, (mouse_pos_raw[0], mouse_pos_raw[1]), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                text = f"x:{mouse_pos_raw[0]} y:{mouse_pos_raw[1]}"
                cv2.putText(frame, text, (mouse_pos_raw[0] + 15, mouse_pos_raw[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow('Video Analysis', cv2.resize(frame, (display_width, display_height)))
            
            # --- Key Handling ---
            k = cv2.waitKey(1) & 0xFF
            if k == 27: break
            elif k == ord('p'): paused = not paused
            
    cap.release()
    cv2.destroyAllWindows()
    print("Process finished.")

if __name__ == "__main__":
    main()