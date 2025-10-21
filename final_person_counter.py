import os
import cv2
import csv
import numpy as np
import json
import argparse
from datetime import datetime

# --- Dependencies ---
from ultralytics import YOLO
from sort import Sort
from config import model_config as cfg

# --- การตั้งค่าที่สำคัญ ---
CONFIG_FILE = 'config/camera_config.json'
SNAP_DIR = "qa_events"  # โฟลเดอร์สำหรับเก็บ Log

# =================== MODEL / TRACKER (Global) ====================
print("Loading AI model...")
model = YOLO("yolov8n.pt", verbose=False)
tracker = Sort(max_age=cfg.MAX_AGE_FRAMES, min_hits=3, iou_threshold=0.2)
print("Model loaded successfully.")

# ====================== GEOMETRY HELPERS =========================
def _cross_sign(p, a, b):
    return np.sign((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))

def is_crossing_line(p1, p2, a, b):
    s1, s2 = _cross_sign(p1, a, b), _cross_sign(p2, a, b)
    s3, s4 = _cross_sign(a, p1, p2), _cross_sign(b, p1, p2)
    return s1 * s2 < 0 and s3 * s4 < 0

def make_side_label(a, b):
    a, b = np.array(a), np.array(b)
    mid_point_below = (a + b) / 2.0 + np.array([0, 100])
    return _cross_sign(mid_point_below, a, b) < 0

def is_top_to_bottom(p1, p2, a, b, neg_is_bottom):
    if not is_crossing_line(p1, p2, a, b): return False
    s1, s2 = _cross_sign(p1, a, b), _cross_sign(p2, a, b)
    if s1 == 0 or s2 == 0: return p2[1] > p1[1]
    bottom_sign = -1 if neg_is_bottom else 1
    return s1 != bottom_sign and s2 == bottom_sign

def is_point_in_zone(point, zone):
    x, y = point
    (zx1, zy1), (zx2, zy2) = zone
    return zx1 <= x <= zx2 and zy1 <= y <= zy2

# ====================== MAIN LOGIC =========================
def main():
    parser = argparse.ArgumentParser(description="Final Person Counting Tool")
    parser.add_argument("camera_name", help="Name of the camera config to use.")
    args = parser.parse_args()

    with open(CONFIG_FILE, "r", encoding='utf-8') as f:
        full_config = json.load(f)
    if args.camera_name not in full_config:
        raise SystemExit(f"Camera '{args.camera_name}' not found in config.")
    config = full_config[args.camera_name]

    video_path = config['video_path']
    display_width = config.get('display_width', 1280)
    pink_zone = tuple(map(tuple, config['pink_zone']))
    red_line = tuple(map(tuple, config['lines']['red']))
    blue_line = tuple(map(tuple, config['lines']['blue']))
    green_line = tuple(map(tuple, config['lines']['green']))
    yellow_line = tuple(map(tuple, config['lines']['yellow']))

    os.makedirs(SNAP_DIR, exist_ok=True)
    local_log_path = os.path.join(SNAP_DIR, f"log_{args.camera_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise IOError("Cannot open video file.")

    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect = original_w / max(1, original_h)
    display_height = int(display_width / aspect)

    cv2.namedWindow("Video Analysis", cv2.WINDOW_NORMAL)

    # --- NEW: เพิ่มตัวแปรสำหรับ Pause และ Mouse ---
    paused = False
    mouse_pos_raw = (-1, -1)
    # --- END NEW ---

    counts = {"inbound": 0}
    person_states = {}
    next_pid = 1
    neg_is_bottom_red = make_side_label(red_line[0], red_line[1])

    # --- NEW: สร้างฟังก์ชัน Mouse Callback และผูกกับหน้าต่าง ---
    def _on_mouse(event, x, y, flags, param):
        """อัปเดตตำแหน่งเมาส์และพิมพ์พิกัดเมื่อคลิก"""
        nonlocal mouse_pos_raw
        # แปลงพิกัดบนจอแสดงผล กลับเป็นพิกัดบนวิดีโอขนาดจริง
        rx = int(x * original_w / display_width)
        ry = int(y * original_h / display_height)
        mouse_pos_raw = (rx, ry)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked Coordinates: ({rx}, {ry})")

    cv2.setMouseCallback("Video Analysis", _on_mouse)
    # --- END NEW ---

    with open(local_log_path, "w", newline="", encoding='utf-8') as csv_file:
        csvw = csv.writer(csv_file)
        csvw.writerow(["Timestamp", "PID", "Event"])
        last_frame = None

        while True:
            ret, frame = cap.read()
            if not ret: break

            # --- MODIFIED: เพิ่ม Logic การ Pause ---
            if not paused:
                ret, frame = cap.read()
                if not ret: break
                last_frame = frame.copy()
            
            if last_frame is None:
                continue
            
            frame = last_frame.copy()
            # --- END MODIFIED ---

            timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            dets = []
            for r in model(frame, stream=True, conf=cfg.SCORE_THR):
                for box in r.boxes.data:
                    if len(box) >= 6 and int(box[5]) == 0:
                        dets.append([int(b) for b in box[:4]] + [float(box[4])])

            tracks = tracker.update(np.array(dets) if dets else np.empty((0, 5)))
            live_tids = {int(t[4]) for t in tracks}

            # --- State Machine Logic ---
            for x1, y1, x2, y2, tid in tracks:
                tid, bbox = int(tid), (int(x1), int(y1), int(x2), int(y2))
                cur_pos = np.array([(x1 + x2) / 2, y2])

                pid = next((p for p, s in person_states.items() if s.get('tid') == tid), None)
                if pid is None:
                    pid = next_pid; next_pid += 1
                    person_states[pid] = {'tid': tid, 'state': 'outside_zone', 'prev_pos': cur_pos}

                st = person_states[pid]
                st['tid'] = tid

                if st['state'] == 'outside_zone':
                    if is_point_in_zone(cur_pos, pink_zone):
                        st['state'] = 'inside_zone'
                elif st['state'] == 'inside_zone':
                    if not is_point_in_zone(cur_pos, pink_zone):
                        st['state'] = 'outside_zone'
                    elif is_top_to_bottom(st['prev_pos'], cur_pos, np.array(red_line[0]), np.array(red_line[1]), neg_is_bottom_red):
                        st['state'] = 'crossed_red'
                elif st['state'] == 'crossed_red':
                    if is_top_to_bottom(cur_pos, st['prev_pos'], np.array(red_line[0]), np.array(red_line[1]), neg_is_bottom_red):
                        st['state'] = 'inside_zone'

                st['prev_pos'] = cur_pos.copy()

                # --- Drawing ---
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
                cv2.putText(frame, f'PID:{pid} ({st["state"]})', (bbox[0], max(20, bbox[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # --- ตรวจสอบคนที่หายไปจากเฟรม ---
            disappeared_pids = [pid for pid, st in person_states.items() if st.get('tid') not in live_tids]
            for pid in disappeared_pids:
                st = person_states[pid]
                if st['state'] == 'crossed_red':
                    counts['inbound'] += 1
                    print(f"PID {pid}: Exited frame after crossing red. COUNT = {counts['inbound']}")
                    csvw.writerow([timestamp_str, pid, 'inbound'])
                    st['state'] = 'counted'
                elif st['state'] != 'counted':
                    st['state'] = 'outside_zone'

            # --- UI Display ---
            cv2.rectangle(frame, pink_zone[0], pink_zone[1], (255, 182, 193), 2)
            cv2.line(frame, red_line[0], red_line[1], (0, 0, 255), 2)
            cv2.line(frame, blue_line[0], blue_line[1], (255, 0, 0), 2)
            cv2.line(frame, green_line[0], green_line[1], (0, 255, 0), 2)
            cv2.line(frame, yellow_line[0], yellow_line[1], (0, 255, 255), 2)
            cv2.putText(frame, f"Inbound: {counts['inbound']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # --- NEW: วาด Mouse Hover Overlay และรับปุ่ม ---
            if mouse_pos_raw[0] >= 0:
                cv2.drawMarker(frame, (mouse_pos_raw[0], mouse_pos_raw[1]), 
                               (0, 255, 255), markerType=cv2.MARKER_CROSS, 
                               markerSize=20, thickness=2)
                text = f"x:{mouse_pos_raw[0]} y:{mouse_pos_raw[1]}"
                cv2.putText(frame, text, (mouse_pos_raw[0] + 15, mouse_pos_raw[1] - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow('Video Analysis', cv2.resize(frame, (display_width, display_height)))
            
            k = cv2.waitKey(1) & 0xFF
            if k == 27: # กด Esc เพื่อออก
                break
            elif k == ord('p'): # กด p เพื่อ Pause/Play
                paused = not paused
            # --- END NEW ---
    
    cap.release()
    cv2.destroyAllWindows()
    print("Process finished.")

if __name__ == "__main__":
    main()