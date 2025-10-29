import os
import cv2
import csv
import numpy as np
import json
import argparse
from datetime import datetime
from collections import deque
import re # <-- เพิ่ม import re

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
        class DefaultConfig: MAX_AGE_FRAMES = 120; SCORE_THR = 0.35
        cfg = DefaultConfig(); print("Warning: model_config.py not found.")

# --- การตั้งค่าที่สำคัญ ---
CONFIG_FILE = 'config/camera_config.json'
SNAP_DIR = "qa_camera_check"
SIGN_HISTORY_LENGTH = 3

# --- NEW: เพิ่ม Tesseract Path และ Import ---
try:
    import pytesseract
    # แก้ Path ให้ตรงกับที่คุณติดตั้ง Tesseract OCR ไว้ (ถ้าจำเป็น)
    # TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # For Windows
    # pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
except ImportError:
    pytesseract = None
    print("Warning: pytesseract not found. OCR functionality will be disabled.")
# --- END NEW ---


# =================== MODEL / TRACKER (Global) ====================
print("Loading AI model...")
model_path = "core/yolov8m.pt" if os.path.exists(os.path.join("core", "yolov8m.pt")) else "yolov8m.pt"
if not os.path.exists(model_path):
     model_path_n = "yolov8n.pt"; model_path_n_core = os.path.join("core", "yolov8n.pt")
     if os.path.exists(model_path_n): model_path = model_path_n; print(f"Warning: yolov8m.pt not found. Trying yolov8n.pt")
     elif os.path.exists(model_path_n_core): model_path = model_path_n_core; print(f"Warning: yolov8m.pt not found. Trying core/yolov8n.pt")
     else: raise FileNotFoundError("Could not find yolov8m.pt or yolov8n.pt")
model = YOLO(model_path, verbose=False)
tracker = Sort(max_age=cfg.MAX_AGE_FRAMES, min_hits=3, iou_threshold=0.2)
print("Model loaded successfully.")

# ====================== GEOMETRY HELPERS =========================
def _cross_sign(p, a, b):
    try:
        p_arr = np.array(p, dtype=np.float64); a_arr = np.array(a, dtype=np.float64); b_arr = np.array(b, dtype=np.float64)
    except (ValueError, TypeError): return 0
    val = (b_arr[0] - a_arr[0]) * (p_arr[1] - a_arr[1]) - (b_arr[1] - a_arr[1]) * (p_arr[0] - a_arr[0])
    tolerance = 1e-9
    if abs(val) < tolerance: return 0
    return np.sign(val)

def make_side_label(a, b):
    a, b = np.array(a), np.array(b)
    mid_point_below = (a + b) / 2.0 + np.array([0, 100])
    return _cross_sign(mid_point_below, a, b) < 0

# --- NEW: เพิ่มฟังก์ชัน OCR กลับเข้ามา ---
def get_timestamp_from_frame(frame, roi):
    """อ่านค่าเวลาจากพื้นที่ (ROI) ที่กำหนดบนเฟรมวิดีโอ"""
    if pytesseract is None or roi is None: return None # ถ้าไม่มี pytesseract หรือ roi ให้ข้าม
    try:
        x1, y1, x2, y2 = roi
        # ตรวจสอบว่า ROI อยู่ในขอบเขตของ frame หรือไม่
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if y2 <= y1 or x2 <= x1: return None # ROI ไม่ถูกต้อง

        timestamp_img = frame[y1:y2, x1:x2]
        gray_img = cv2.cvtColor(timestamp_img, cv2.COLOR_BGR2GRAY)
        
        # --- ใช้ Adaptive Thresholding ---
        binary_img = cv2.adaptiveThreshold(gray_img, 255,
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV,
                                             blockSize=11, C=5)

        text = pytesseract.image_to_string(binary_img, config=r'--oem 3 --psm 6')
        
        # พยายามจับคู่ format 'MM-YYYY ... HH:MM:SS'
        match = re.search(r'(\d{2})-(\d{4}).*?(\d{2}:\d{2}:\d{2})', text.replace(" ", ""))
        if match:
            month, year, time_str = match.groups()
            try:
                 # ลองสร้าง datetime object (ใช้ day=1 เป็น default)
                 dt_object = datetime.strptime(f"01-{month}-{year} {time_str}", '%d-%m-%Y %H:%M:%S')
                 return dt_object
            except ValueError:
                 return None # Format ไม่ถูกต้อง
    except Exception as e:
        # print(f"OCR Error: {e}") # Uncomment for debugging OCR issues
        return None
    return None
# --- END NEW ---

# ====================== FILE SYSTEM HELPER =========================
def ensure_dir(dir_path):
    if not os.path.exists(dir_path): os.makedirs(dir_path); print(f"Created directory: {dir_path}")

# ====================== MAIN LOGIC =========================
def main():
    parser = argparse.ArgumentParser(description="Person Counter (Sign History Logic)")
    parser.add_argument("camera_name", help="Name of the camera config to use.")
    args = parser.parse_args()

    # --- Load Config ---
    try:
        with open(CONFIG_FILE, "r", encoding='utf-8') as f: full_config = json.load(f)
    except FileNotFoundError: raise SystemExit(f"Config file '{CONFIG_FILE}' not found.")
    if args.camera_name not in full_config: raise SystemExit(f"Camera '{args.camera_name}' not found.")
    config = full_config[args.camera_name]

    video_path = config.get('video_path'); display_width = config.get('display_width', 1280)
    red_line = tuple(map(tuple, config['lines']['red']))
    blue_line = tuple(map(tuple, config['lines']['blue']))
    green_line = tuple(map(tuple, config['lines']['green']))
    yellow_line = tuple(map(tuple, config['lines']['yellow']))
    pink_zone = tuple(map(tuple, config['pink_zone']))
    timestamp_roi = config.get('timestamp_roi') # --- NEW: โหลด timestamp_roi ---

    # --- Paths ---
    camera_event_dir = os.path.join(SNAP_DIR, args.camera_name)
    log_dir = os.path.join(camera_event_dir, "logs")
    person_snapshot_dir = os.path.join(camera_event_dir, "person_snapshots")
    ensure_dir(log_dir); ensure_dir(person_snapshot_dir)
    local_log_path = os.path.join(log_dir, f"log_{args.camera_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise IOError(f"Cannot open video: {video_path}")
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if original_w == 0 or original_h == 0: raise IOError("Could not read video dimensions.")
    aspect = original_w / max(1, original_h); display_height = int(display_width / aspect)

    cv2.namedWindow("Video Analysis", cv2.WINDOW_NORMAL)
    paused = False; mouse_pos_raw = (-1, -1)
    counts = {"inbound": 0}; person_states = {}; next_pid = 1

    neg_is_bottom_red = make_side_label(red_line[0], red_line[1])
    bottom_sign = -1 if neg_is_bottom_red else 1
    top_sign = -bottom_sign

    # --- Mouse Callback ---
    def _on_mouse(event, x, y, flags, param):
        nonlocal mouse_pos_raw
        rx = int(x * original_w / display_width); ry = int(y * original_h / display_height)
        mouse_pos_raw = (rx, ry)
        if event == cv2.EVENT_LBUTTONDOWN: print(f"Clicked: ({rx}, {ry})")
    cv2.setMouseCallback("Video Analysis", _on_mouse)

    with open(local_log_path, "w", newline="", encoding='utf-8') as csv_file:
        csvw = csv.writer(csv_file, delimiter='|'); csvw.writerow(["Date", "Time", "PID", "Status"])
        last_frame = None

        while True:
            current_time_dt = datetime.now()
            if not paused:
                ret, frame = cap.read()
                if not ret: break
                last_frame = frame.copy()
            if last_frame is None: continue
            frame = last_frame.copy()

            # --- NEW: เรียกใช้ OCR ---
            ocr_timestamp_dt = get_timestamp_from_frame(frame, timestamp_roi)
            display_timestamp_str = ocr_timestamp_dt.strftime('%d-%m-%Y %H:%M:%S') if ocr_timestamp_dt else ""
            # --- END NEW ---

            dets = []; tracks = np.empty((0, 5))
            results = model(frame, stream=True, conf=cfg.SCORE_THR)
            valid_results = False
            for r in results:
                valid_results = True
                for box in r.boxes.data:
                    if len(box) >= 6 and int(box[5]) == 0: dets.append([int(b) for b in box[:4]] + [float(box[4])])
            if valid_results:
                 tracks = tracker.update(np.array(dets) if dets else np.empty((0, 5)))

            live_tids = {int(t[4]) for t in tracks}

            # --- State Machine (Using Sign History) ---
            for x1, y1, x2, y2, tid in tracks:
                tid, bbox = int(tid), (int(x1), int(y1), int(x2), int(y2))
                cur_pos = np.array([(x1 + x2) / 2, y1])

                pid = next((p for p, s in person_states.items() if s.get('tid') == tid), None)
                if pid is None:
                    pid = next_pid; next_pid += 1
                    person_states[pid] = {'tid': tid, 'state': 'waiting',
                                          'sign_history': deque(maxlen=SIGN_HISTORY_LENGTH),
                                          'last_frame_seen': frame.copy()}

                st = person_states[pid]
                st['tid'] = tid
                st['last_bbox'] = bbox
                st['last_frame_seen'] = frame.copy()

                current_sign = _cross_sign(cur_pos, red_line[0], red_line[1])
                # Ensure sign is not None before appending
                if current_sign is not None:
                    st['sign_history'].append(current_sign)
                history = list(st['sign_history'])

                crossed_top_to_bottom = False
                crossed_bottom_to_top = False

                if len(history) >= 2:
                    prev_sign = history[-2]
                    last_sign = history[-1]
                    if (prev_sign == top_sign or prev_sign == 0) and (last_sign == bottom_sign or last_sign == 0) and prev_sign != last_sign:
                        crossed_top_to_bottom = True
                    elif (prev_sign == bottom_sign or prev_sign == 0) and (last_sign == top_sign or last_sign == 0) and prev_sign != last_sign:
                        crossed_bottom_to_top = True

                if st['state'] == 'waiting':
                    if crossed_top_to_bottom: st['state'] = 'crossed_red'; print(f"PID {pid}: -> crossed_red")
                elif st['state'] == 'crossed_red':
                    if crossed_bottom_to_top: st['state'] = 'waiting'; print(f"PID {pid}: -> waiting")

                # --- Drawing ---
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
                cv2.putText(frame, f'PID:{pid} ({st["state"]})', (bbox[0], max(20, bbox[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.circle(frame, (int(cur_pos[0]), int(cur_pos[1])), 5, (0, 0, 255), -1)

            # --- Check Disappeared People & Cleanup ---
            pids_to_remove = set()
            for pid, st in person_states.items():
                if st.get('tid') not in live_tids:
                    if st['state'] == 'crossed_red':
                        counts['inbound'] += 1
                        print(f"PID {pid}: Exited frame -> COUNT = {counts['inbound']}")
                        
                        # --- MODIFIED: ใช้เวลาจาก OCR ถ้ามี ---
                        log_dt = ocr_timestamp_dt if ocr_timestamp_dt else current_time_dt
                        log_date_str = log_dt.strftime('%Y-%m-%d')
                        log_time_str = log_dt.strftime('%H:%M:%S')
                        csvw.writerow([log_date_str, log_time_str, pid, 'inbound'])
                        # --- END MODIFIED ---
                        
                        # --- Snapshot Logic ---
                        last_frame_seen = st.get('last_frame_seen')
                        if last_frame_seen is not None:
                            frame_to_save = last_frame_seen.copy()
                            cv2.rectangle(frame_to_save, pink_zone[0], pink_zone[1], (255, 182, 193), 2)
                            cv2.line(frame_to_save, red_line[0], red_line[1], (0, 0, 255), 2)
                            last_bbox = st.get('last_bbox')
                            if last_bbox: cv2.rectangle(frame_to_save, (last_bbox[0], last_bbox[1]), (last_bbox[2], last_bbox[3]), (0, 255, 0), 3)
                            snap_fname = os.path.join(person_snapshot_dir, f"inbound_pid{pid}_{current_time_dt.strftime('%Y%m%d_%H%M%S')}.jpg")
                            cv2.imwrite(snap_fname, frame_to_save); print(f"Saved snapshot: {os.path.basename(snap_fname)}")
                        else: print(f"Warning: No snapshot for PID {pid}.")
                    
                    pids_to_remove.add(pid)

            for pid in pids_to_remove:
                if pid in person_states: del person_states[pid]

            # --- UI Display ---
            cv2.rectangle(frame, pink_zone[0], pink_zone[1], (255, 182, 193), 2)
            cv2.line(frame, red_line[0], red_line[1], (0, 0, 255), 2)
            cv2.line(frame, blue_line[0], blue_line[1], (255, 0, 0), 2)
            cv2.line(frame, green_line[0], green_line[1], (0, 255, 0), 2)
            cv2.line(frame, yellow_line[0], yellow_line[1], (0, 255, 255), 2)
            cv2.putText(frame, f"Inbound: {counts['inbound']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
           
           # --- NEW: วาดพื้นหลังสีดำทึบสำหรับ Timestamp ---
            if display_timestamp_str:
                 try:
                      font_scale = 0.6
                      thickness = 1
                      font = cv2.FONT_HERSHEY_SIMPLEX
                      text_x, text_y = 10, 30

                      (text_width, text_height), baseline = cv2.getTextSize(display_timestamp_str, font, font_scale, thickness)

                      padding = 5
                      bg_x1 = max(text_x - padding, 0)
                      bg_y1 = max(text_y - text_height - padding - baseline, 0) # ปรับ y1 ให้พอดีขึ้น
                      bg_x2 = min(text_x + text_width + padding, frame.shape[1])
                      bg_y2 = min(text_y + padding, frame.shape[0]) # ปรับ y2 ให้พอดีขึ้น

                      if bg_y2 > bg_y1 and bg_x2 > bg_x1:
                           # วาดสี่เหลี่ยมสีดำทึบ (-1 คือ заливка)
                           cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                 except Exception as e:
                      print(f"Error drawing timestamp background: {e}")
            # --- END NEW ---

            # วาด Timestamp ทับพื้นหลังสีดำ (ใช้สีเหลืองเพื่อให้เห็นชัด)
            cv2.putText(frame, display_timestamp_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1) # สีเหลือง (BGR)

            if mouse_pos_raw[0] >= 0:
                cv2.drawMarker(frame, (mouse_pos_raw[0], mouse_pos_raw[1]), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                text = f"x:{mouse_pos_raw[0]} y:{mouse_pos_raw[1]}"; cv2.putText(frame, text, (mouse_pos_raw[0] + 15, mouse_pos_raw[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow('Video Analysis', cv2.resize(frame, (display_width, display_height)))
            k = cv2.waitKey(1) & 0xFF
            if k == 27: break
            elif k == ord('p'): paused = not paused
            
    cap.release()
    cv2.destroyAllWindows()
    print("Process finished.")

if __name__ == "__main__":
    main()