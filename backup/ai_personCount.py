# ai_personCount_v8_hover_coords.py
# Base: v8 (snap only on true miss) + Mouse hover to show raw x,y coordinates
# - Move mouse on the video window to see raw frame coordinates
# - Press 'h' to toggle hover overlay on/off
# - Left-click prints the coordinate to console for easy copy

import os
import cv2
import csv
import numpy as np
import json
import argparse

from math import hypot
from collections import deque
from ultralytics import YOLO
from sort import Sort

parser = argparse.ArgumentParser(description="AI Person Counting QA Tool")
parser.add_argument("camera_name", help="Name of the camera config to use from camera_config.json")
args = parser.parse_args()
# ======================== CONFIG Boundary ========================
video_path      = "5033 North Pattaya (Cam- 61 ) - Entrance2.mp4"

# Load config file
try:
    with open("core/camera_setting.json", "r", encoding='utf-8') as f:
        all_boundary = json.load(f)
except FileNotFoundError:
    raise SystemExit("Error: boundary not found.")

if args.camera_name not in all_boundary:
    raise SystemExit(f"Error: Camera '{args.camera_name}' not found in camera_config.json.")
config = all_boundary[args.camera_name]

# Camera-specific lines

# red_line    = [(616, 275), (1345, 279)]     # entrance
# blue_line   = [(616, 275), (552, 983)]      # right
# green_line  = [(1345, 279), (1537, 956)]    # left
# yellow_line = [(552, 983), (1537, 956)]     # straight

red_line    = tuple(map(tuple, config['lines']['top'])) # entrance
blue_line   = tuple(map(tuple, config['lines']['right'])) # right
green_line  = tuple(map(tuple, config['lines']['left'])) # left
yellow_line = tuple(map(tuple, config['lines']['bottom'])) # straight

# display_width   = config.get('display_width', 1280)
# display_height  = config.get('display_height', 1024)

# ======================== Camera CONSTANTS ========================
# UI
SHOW_TRAIL            = True
TRAIL_LEN             = 20
DRAW_EVENT_LIFETIME_S = 1.6

# --- Mouse hover overlay ---
HOVER_ENABLED = True  # press 'h' to toggle
mouse_pos_disp = (-1, -1)
mouse_pos_raw  = (-1, -1)

# Snapshots: ONLY for misses
SAVE_SNAPSHOTS_MISS   = True
SNAP_DIR              = "qa_events"

# Tracking / Timing
INBOUND_TIMEOUT_S   = 30.0
STATE_RETENTION_S   = 25.0
REID_MAX_GAP_S      = 8.0
REID_DIST_PX        = 300
REID_IOU_THRESH     = 0.01
MAX_AGE_FRAMES      = 120

# Detection
SCORE_THR           = 0.35

# Anti-double-count (RED)
RED_DEBOUNCE_S         = 0.6
REARM_DIST_FROM_RED_PX = 35
MIN_V_SHIFT_PX         = 2

# Cross confirmation
CONSEC_NEEDED_ON_RED  = 1
CONSEC_NEEDED_ON_DEST = 1

# Global dedupe window along red
GLOBAL_TIME_WIN  = 0.30
GLOBAL_X_TOL     = 18

# Hot re-ID guard (to keep PID consistent right after IN)
HOT_REID_DIST_PX      = 140
HOT_REID_IOU          = 0.02
HOT_REID_TIMELOCK_S   = 2.0

# Direction gating
DIR_MIN_TRAVEL_PX     = 30

# ====================== HELPERS =========================

def _cross_sign(p, a, b):
    return np.sign((b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]))

def _bb_overlap(p, q, a, b):
    minx1, maxx1 = min(p[0], q[0]), max(p[0], q[0])
    miny1, maxy1 = min(p[1], q[1]), max(p[1], q[1])
    minx2, maxx2 = min(a[0], b[0]), max(a[0], b[0])
    miny2, maxy2 = min(a[1], b[1]), max(a[1], b[1])
    return not (maxx1 < minx2 or maxx2 < minx1 or maxy1 < miny2 or maxy2 < miny1)

def is_crossing_line(p1, p2, a, b, allow_touch=True):
    if not _bb_overlap(p1, p2, a, b):
        return False
    s1 = _cross_sign(p1, a, b)
    s2 = _cross_sign(p2, a, b)
    if allow_touch:
        return (s1 == 0 and s2 != 0) or (s2 == 0 and s1 != 0) or (s1 * s2 < 0)
    return s1 * s2 < 0

def make_side_label(a,b):
    a, b = np.array(a), np.array(b)
    mid_point_below = (a + b) / 2.0 + np.array([0, 100])
    return _cross_sign(mid_point_below, a, b) < 0

def is_top_to_bottom(p1, p2, a, b, neg_is_bottom):
    if not is_crossing_line(p1, p2, a, b):
        return False
    
    s1 = _cross_sign(p1, a, b)
    s2 = _cross_sign(p2, a, b)
    
    # If starting or ending on the line, a simple vertical check is sufficient
    if s1 == 0 or s2 == 0:
        return p2[1] > p1[1]
    
    # Define which sign represents the "bottom" side
    bottom_sign = -1 if neg_is_bottom else 1
    
    # The start point (s1) must NOT be on the bottom side,
    # and the end point (s2) MUST be on the bottom side.
    return s1 != bottom_sign and s2 == bottom_sign

def bbox_iou(b1, b2):
    xA = max(b1[0], b2[0]); yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2]); yB = min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = max(0, b1[2]-b1[0]) * max(0, b1[3]-b1[1])
    area2 = max(0, b2[2]-b2[0]) * max(0, b2[3]-b2[1])
    denom = area1 + area2 - inter
    return inter / denom if denom > 0 else 0.0

def point_to_line_distance(px, py, a, b):
    ax, ay = a; bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return np.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax)*dx + (py - ay)*dy) / float(dx*dx + dy*dy)))
    projx, projy = ax + t*dx, ay + t*dy
    return np.hypot(px - projx, py - projy)

def proj_x_on_segment(p, a, b):
    ax, ay = a; bx, by = b
    dx, dy = bx - ax, by - ay
    denom = dx*dx + dy*dy + 1e-6
    t = max(0.0, min(1.0, ((p[0]-ax)*dx + (p[1]-ay)*dy) / denom))
    return ax + t*dx

# =================== MODEL / TRACKER ====================
model = YOLO("yolov8n.pt", verbose=False)
try:
    tracker = Sort(max_age=MAX_AGE_FRAMES, min_hits=3, iou_threshold=0.2)
except TypeError:
    tracker = Sort(max_age=MAX_AGE_FRAMES, min_hits=3)

# =================== STATE & LOGGING ====================
counts = {"right": 0, "left": 0, "straight": 0, "inbound": 0}
person_states = {}
tid_to_pid    = {}
next_pid      = 1

neg_is_bottom_red = make_side_label(red_line[0], red_line[1])

trails = {}
event_overlays = []
recent_in_events = deque(maxlen=20)  # (time, x_on_red)

os.makedirs(SNAP_DIR, exist_ok=True)
csv_path  = os.path.join(SNAP_DIR, "results_v8_hover.csv")
csv_file  = open(csv_path, "w", newline="")
csvw      = csv.writer(csv_file)
csvw.writerow(["time", "pid", "event", "total"])

# --- dedupe helpers ---

def is_dup_global(now, p_on_screen):
    x = proj_x_on_segment(p_on_screen, red_line[0], red_line[1])
    for (t, x0) in recent_in_events:
        if abs(now - t) <= GLOBAL_TIME_WIN and abs(x - x0) <= GLOBAL_X_TOL:
            return True
    return False

def remember_global(now, p_on_screen):
    recent_in_events.append((now, proj_x_on_segment(p_on_screen, red_line[0], red_line[1])))

# --- IO helper (MISS only) ---

def save_miss_snapshot(frame, now, pid, tag):
    if not SAVE_SNAPSHOTS_MISS:
        return
    name = f"{int(now*1000):010d}_PID{pid}_{tag}.jpg"
    cv2.imwrite(os.path.join(SNAP_DIR, name), frame)

# ======================== VIDEO =========================
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise SystemExit("Error: Could not open video file.")

# Auto-scaling logic: คำนวณขนาดแสดงผลโดยรักษาสัดส่วนภาพ
max_w = config.get("max_display_width", 1280) # ดึงค่าความกว้างสูงสุดจาก config
original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
# W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

aspect_ratio = original_h / original_w
display_width = max_w
display_height = int(display_width * aspect_ratio)

# keep aspect for display size already chosen
# aspect = W / max(1, H)
# if display_width / display_height > aspect:
#     display_width = int(display_height * aspect)
# else:
#     display_height = int(display_width / aspect)

cv2.namedWindow("Video Analysis", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video Analysis", display_width, display_height)

# mouse callback (map display coords -> raw frame coords)

def _on_mouse(event, x, y, flags, param):
    global mouse_pos_disp, mouse_pos_raw
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pos_disp = (x, y)
        rx = int(x * original_w / display_width)
        ry = int(y * original_h / display_height)
        mouse_pos_raw = (rx, ry)
    elif event == cv2.EVENT_LBUTTONDOWN:
        rx = int(x * original_w / display_width)
        ry = int(y * original_h / display_height)
        print(f"[CLICK] raw=({rx},{ry})  disp=({x},{y})")

cv2.setMouseCallback("Video Analysis", _on_mouse)

frame_idx = 0
paused = False

# ---- draw helpers ----

def draw_lines(img):
    cv2.line(img, red_line[0], red_line[1], (0, 0, 255), 1)
    cv2.line(img, blue_line[0], blue_line[1], (255, 0, 0), 1)
    cv2.line(img, green_line[0], green_line[1], (0, 255, 0), 1)
    cv2.line(img, yellow_line[0], yellow_line[1], (0, 255, 255), 1)

def put_text(img, text, xy, color=(255,255,255)):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

# ------------- Main loop -------------
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
    now = frame_idx / fps

    # detection
    dets = []
    for r in model(frame, stream=True, conf=SCORE_THR):
        for box in r.boxes.data:
            if len(box) >= 6:
                x1, y1, x2, y2, score, cls = box
                if int(cls) == 0 and float(score) >= SCORE_THR:
                    dets.append([int(x1), int(y1), int(x2), int(y2), float(score)])
    tracks = tracker.update(np.array(dets)) if dets else tracker.update(np.empty((0,5)))

    draw_lines(frame)
    live_tids = set(int(t[4]) for t in tracks)
    
    # เพิ่มตัวแปรสำหรับเก็บ PID ที่หายไปจากเฟรมชั่วคราว
    pids_disappeared = set(person_states.keys())

    # ----- per track -----
    for x1, y1, x2, y2, tid in tracks:
        tid = int(tid)
        cx_c, cy_c = int((x1 + x2) / 2), int((y1 + y2) / 2)  # center (RED)
        cx_b, cy_b = int((x1 + x2) / 2), int(y2)             # bottom-center (DEST)
        bbox = (int(x1), int(y1), int(x2), int(y2))

        # re-id with HOT guard
        pid = tid_to_pid.get(tid)
        if pid is None:
            best_pid, best_cost = None, 1e9
            for p, st in person_states.items():
                if now - st['last_seen'] > REID_MAX_GAP_S:
                    continue
                d = hypot(cx_b - st['last_pos_bottom'][0], cy_b - st['last_pos_bottom'][1])
                iou = bbox_iou(bbox, st['last_bbox']) if st['last_bbox'] else 0.0

                if st.get('has_entered') and (now - st.get('start_time', -1)) <= INBOUND_TIMEOUT_S:
                    if (now - st['start_time']) <= HOT_REID_TIMELOCK_S and st.get('enter_tid') is not None and st['enter_tid'] != tid:
                        continue
                    if d > HOT_REID_DIST_PX or iou < HOT_REID_IOU:
                        continue
                else:
                    if d > REID_DIST_PX or iou < REID_IOU_THRESH:
                        continue

                cost = d + (1.0 - iou) * 50.0
                if cost < best_cost:
                    best_cost, best_pid = cost, p

            pid = best_pid if best_pid is not None else next_pid
            if best_pid is None:
                next_pid += 1
            tid_to_pid[tid] = pid

        st = person_states.get(pid)
        if st is None:
            st = person_states[pid] = {
                'has_entered': False,
                'start_time': 0.0,
                'destination': None,
                'last_pos_center': (cx_c, cy_c),
                'last_pos_bottom': (cx_b, cy_b),
                'last_bbox': bbox,
                'last_seen': now,
                'age_frames': 0,
                'red_cooldown_until': 0.0,
                'need_rearm_from_red': False,
                'red_consec': 0,
                'blue_consec': 0,
                'green_consec': 0,
                'yellow_consec': 0,
                'red_cross_point': None,
                'enter_tid': None,
                'enter_token': 0,
                'active_token': 0,
                'last_in_time': -1.0,
                'last_in_x_on_red': None,
                'saw_red_cross': False,
                'miss_in_logged': False,
                'miss_dir_logged': {'right': False, 'left': False, 'straight': False},
            }

        last_c = st['last_pos_center']
        last_b = st['last_pos_bottom']
        cur_c  = (cx_c, cy_c)
        cur_b  = (cx_b, cy_b)
        vshift = abs(cur_c[1] - last_c[1])
        
        st['last_pos_bottom'] = cur_b
        st['last_seen'] = now
        pids_disappeared.discard(pid) # คนนี้ยังอยู่ในเฟรม

        # re-arm from RED zone
        dist_red = point_to_line_distance(cur_c[0], cur_c[1], red_line[0], red_line[1])
        if st['need_rearm_from_red'] and dist_red > REARM_DIST_FROM_RED_PX:
            st['need_rearm_from_red'] = False

        # -------- RED entry (with MISS detection) --------
        red_cross = False
        red_in_counted = False

        if (not st['need_rearm_from_red']) and (now >= st['red_cooldown_until']):
            if vshift >= MIN_V_SHIFT_PX and is_crossing_line(last_c, cur_c, red_line[0], red_line[1]):
                st['red_consec'] += 1
                if st['red_consec'] >= CONSEC_NEEDED_ON_RED:
                    if is_top_to_bottom(last_c, cur_c, red_line[0], red_line[1], neg_is_bottom_red):
                        red_cross = True
                        st['saw_red_cross'] = True
                        x_on_red = proj_x_on_segment(cur_c, red_line[0], red_line[1])
                        pid_dup = (st['last_in_time'] > 0 and (now - st['last_in_time'] <= GLOBAL_TIME_WIN)
                                   and st['last_in_x_on_red'] is not None and abs(x_on_red - st['last_in_x_on_red']) <= GLOBAL_X_TOL)
                        if (not st['has_entered']) and (not pid_dup) and (not is_dup_global(now, cur_c)):
                            st['has_entered'] = True
                            st['start_time'] = now
                            st['destination'] = None
                            counts['inbound'] += 1
                            st['last_in_time'] = now
                            st['last_in_x_on_red'] = x_on_red
                            st['enter_tid'] = tid
                            st['enter_token'] += 1
                            st['active_token'] = st['enter_token']
                            st['red_cross_point'] = cur_c
                            remember_global(now, cur_c)
                            st['red_cooldown_until'] = now + RED_DEBOUNCE_S
                            st['need_rearm_from_red'] = True
                            csvw.writerow([round(now,2), pid, 'inbound', counts['inbound']]); csv_file.flush()
                            red_in_counted = True
                    st['red_cooldown_until'] = now + RED_DEBOUNCE_S
                    st['need_rearm_from_red'] = True
                    st['red_consec'] = 0
            else:
                st['red_consec'] = 0

        # MISS: crossed RED top->bottom but NOT counted as IN
        if red_cross and (not red_in_counted) and (not st['has_entered']) and (not st['miss_in_logged']):
            st['miss_in_logged'] = True
            csvw.writerow([round(now,2), pid, 'miss_in_cross', counts['inbound']]); csv_file.flush()
            save_miss_snapshot(frame, now, pid, 'MISS_IN_RED')

        # -------- DESTINATION (with MISS detection) --------
        crossed_raw = None
        if is_crossing_line(last_b, cur_b, blue_line[0], blue_line[1]):
            crossed_raw = 'right'
        elif is_crossing_line(last_b, cur_b, green_line[0], green_line[1]):
            crossed_raw = 'left'
        elif is_crossing_line(last_b, cur_b, yellow_line[0], yellow_line[1]):
            crossed_raw = 'straight'

        dir_counted = False
        if st['has_entered'] and st['destination'] is None and st['active_token'] == st['enter_token'] and st['red_cross_point'] is not None:
            if now - st['start_time'] <= INBOUND_TIMEOUT_S:
                crossed = None
                if is_crossing_line(last_b, cur_b, blue_line[0], blue_line[1]):
                    crossed = 'right'
                elif is_crossing_line(last_b, cur_b, green_line[0], green_line[1]):
                    crossed = 'left'
                elif is_crossing_line(last_b, cur_b, yellow_line[0], yellow_line[1]):
                    crossed = 'straight'

                if crossed is not None:
                    dx = cur_b[0] - st['red_cross_point'][0]
                    dy = cur_b[1] - st['red_cross_point'][1]
                    if (dx*dx + dy*dy) ** 0.5 >= DIR_MIN_TRAVEL_PX:
                        st['destination'] = crossed
                        counts[crossed] += 1
                        st['has_entered'] = False
                        st['need_rearm_from_red'] = True
                        st['red_cooldown_until'] = now + RED_DEBOUNCE_S
                        st['active_token'] = 0
                        csvw.writerow([round(now,2), pid, crossed, counts[crossed]]); csv_file.flush()
                        dir_counted = True

        if crossed_raw is not None and (not dir_counted):
            if st['saw_red_cross'] and not st['miss_dir_logged'][crossed_raw]:
                st['miss_dir_logged'][crossed_raw] = True
                csvw.writerow([round(now,2), pid, f'miss_dir_cross_{crossed_raw}', counts['inbound']]); csv_file.flush()
                save_miss_snapshot(frame, now, pid, f'MISS_DIR_{crossed_raw.upper()}')

        # update state
        st['last_pos_center'] = cur_c
        st['last_pos_bottom'] = cur_b
        st['last_bbox'] = bbox
        st['last_seen'] = now
        st['age_frames'] += 1

        # draw bbox & anchors
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,255,0), 2)
        cv2.putText(frame, f'PID:{pid}', (bbox[0], max(20, bbox[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.circle(frame, cur_c, 5, (0,165,255), -1)
        cv2.circle(frame, cur_b, 5, (0,255,255), -1)

    # cleanup
    for tid in list(tid_to_pid.keys()):
        if tid not in live_tids:
            del tid_to_pid[tid]
    for pid in list(person_states.keys()):
        if now - person_states[pid]['last_seen'] > STATE_RETENTION_S:
            person_states.pop(pid, None)
            trails.pop(pid, None)

    # HUD
    # put_text(frame, f"Right: {counts['right']}", (10, 30))
    # put_text(frame, f"Left: {counts['left']}", (10, 65))
    put_text(frame, f"Straight: {counts['straight']}", (10, 100))
    put_text(frame, f"Inbound: {counts['inbound']}", (10, 135))
    # put_text(frame, "Rule: RED top->bottom then Blue/Green/Yellow", (10, 170), (0,255,255))
    # put_text(frame, "Key: Esc=quit, p=pause, space=step, h=toggle hover", (10, 205), (200,200,200))

    # resize for display
    disp = cv2.resize(frame, (display_width, display_height))

    # mouse hover overlay drawn on *disp* (display coords)
    if HOVER_ENABLED and mouse_pos_disp[0] >= 0:
        x, y = mouse_pos_disp
        rx, ry = mouse_pos_raw
        # crosshair
        cv2.drawMarker(disp, (x, y), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=1)
        # label (stay inside frame)
        tx = min(x + 12, display_width - 220)
        ty = max(y - 10, 20)
        cv2.putText(disp, f"x:{rx}  y:{ry}", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video Analysis', disp)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == ord('p'):
        paused = not paused
    elif k == 32 and paused:
        ret, frame = cap.read()
        if ret:
            frame_idx += 1
    elif k == ord('h'):
        HOVER_ENABLED = not HOVER_ENABLED

cap.release()
csv_file.close()
cv2.destroyAllWindows()
