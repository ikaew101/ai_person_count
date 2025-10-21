import os
import cv2
import csv
import numpy as np
import json
import argparse
import re
import shutil
from datetime import datetime
from math import hypot
from collections import deque

# --- Dependencies ---
import pytesseract
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from ultralytics import YOLO
from sort import Sort

# --- การตั้งค่าที่สำคัญ ---
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
CONFIG_FILE = 'core/final_config.json'
SNAP_DIR = "qa_events" # โฟลเดอร์สำหรับเก็บ Log และ Snapshot

# =================== MODEL / TRACKER (Global) ====================
print("Loading AI model...")
model = YOLO("yolov8n.pt", verbose=False)
tracker = Sort(max_age=120, min_hits=3, iou_threshold=0.2)
print("Model loaded successfully.")

# ====================== GOOGLE DRIVE, OCR, GEOMETRY HELPERS =========================
def authenticate_gdrive():
    print("Connecting to Google Drive...")
    # ตรวจสอบว่ามี core/client_secrets.json หรือไม่ ถ้าไม่มีให้ใช้ที่ root
    secrets_path = os.path.join("core", "client_secrets.json") if os.path.exists("core") else "client_secrets.json"
    creds_path = os.path.join("core", "credentials.json") if os.path.exists("core") else "credentials.json"
    
    settings = {
        "client_config_file": secrets_path,
        "save_credentials": True, "get_refresh_token": True,
        "save_credentials_backend": "file",
        "save_credentials_file": creds_path
    }
    gauth = GoogleAuth(settings=settings)
    gauth.LocalWebserverAuth()
    print("Authentication successful.")
    return GoogleDrive(gauth)

def download_video_from_gdrive(drive, filename, download_path):
    print(f"Searching for '{filename}' on Google Drive...")
    file_list = drive.ListFile({'q': f"title='{filename}' and trashed=false"}).GetList()
    if not file_list: raise FileNotFoundError(f"Error: Video file '{filename}' not found.")
    gdrive_file = file_list[0]
    print(f"File found (ID: {gdrive_file['id']}). Downloading...")
    gdrive_file.GetContentFile(download_path)
    print("Download complete.")
    return download_path

def upload_log_to_gdrive(drive, local_log_path, folder_id):
    if not os.path.exists(local_log_path): return
    print(f"Uploading log file '{os.path.basename(local_log_path)}'...")
    file_metadata = {'title': os.path.basename(local_log_path), 'parents': [{'id': folder_id}]}
    gfile = drive.CreateFile(file_metadata)
    gfile.SetContentFile(local_log_path)
    gfile.Upload()
    print("Upload complete.")

def get_timestamp_from_frame(frame, roi):
    try:
        x1, y1, x2, y2 = roi
        timestamp_img = frame[y1:y2, x1:x2]
        gray_img = cv2.cvtColor(timestamp_img, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(gray_img, 170, 255, cv2.THRESH_BINARY_INV)
        text = pytesseract.image_to_string(binary_img, config=r'--oem 3 --psm 6')
        match = re.search(r'(\d{2})-(\d{4}).*?(\d{2}:\d{2}:\d{2})', text.replace(" ", ""))
        if match:
            month, year, time_str = match.groups()
            return datetime.strptime(f"01-{month}-{year} {time_str}", '%d-%m-%Y %H:%M:%S')
    except Exception: return None
    return None

def _cross_sign(p, a, b): return np.sign((b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0]))
def is_crossing_line(p1,p2,a,b):
    if max(p1[0],p2[0])<min(a[0],b[0]) or min(p1[0],p2[0])>max(a[0],b[0]): return False
    if max(p1[1],p2[1])<min(a[1],b[1]) or min(p1[1],p2[1])>max(a[1],b[1]): return False
    s1,s2=_cross_sign(p1,a,b),_cross_sign(p2,a,b)
    s3,s4=_cross_sign(a,p1,p2),_cross_sign(b,p1,p2)
    return s1*s2<0 and s3*s4<0
def make_side_label(a,b):
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

    with open(CONFIG_FILE, "r", encoding='utf-8') as f: full_config = json.load(f)
    if args.camera_name not in full_config['cameras']: raise SystemExit(f"Camera '{args.camera_name}' not found in config.")
    config = full_config['cameras'][args.camera_name]
    gdrive_log_folder_id = full_config.get("gdrive_log_folder_id")
    
    video_filename = config['video_filename_gdrive']
    timestamp_roi = config['timestamp_roi']
    max_w = config.get("max_display_width", 1280)
    pink_zone = tuple(map(tuple, config['pink_zone']))
    red_line = tuple(map(tuple, config['lines']['red']))
    
    local_video_path = f"temp_{video_filename}"
    os.makedirs(SNAP_DIR, exist_ok=True)
    local_log_path = os.path.join(SNAP_DIR, f"log_{args.camera_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    drive = None
    try:
        drive = authenticate_gdrive()
        download_video_from_gdrive(drive, video_filename, local_video_path)

        cap = cv2.VideoCapture(local_video_path)
        if not cap.isOpened(): raise IOError("Cannot open video file.")
        
        original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        display_width = max_w
        display_height = int(display_width * (original_h / original_w))

        cv2.namedWindow("Video Analysis", cv2.WINDOW_NORMAL)
        
        counts = {"inbound": 0}
        person_states = {} # Stores state for each tracked person (PID)
        next_pid = 1
        neg_is_bottom_red = make_side_label(red_line[0], red_line[1])
        
        with open(local_log_path, "w", newline="", encoding='utf-8') as csv_file:
            csvw = csv.writer(csv_file)
            csvw.writerow(["Timestamp", "PID", "Event"])

            while True:
                ret, frame = cap.read()
                if not ret: break

                ocr_timestamp = get_timestamp_from_frame(frame, timestamp_roi)
                timestamp_str = ocr_timestamp.strftime('%d-%m-%Y %H:%M:%S') if ocr_timestamp else "N/A"

                dets = []
                for r in model(frame, stream=True, conf=0.35):
                    for box in r.boxes.data:
                        if len(box) >= 6 and int(box[5]) == 0:
                            dets.append([int(b) for b in box[:4]] + [float(box[4])])
                
                tracks = tracker.update(np.array(dets) if dets else np.empty((0, 5)))
                
                live_tids = {int(t[4]) for t in tracks}
                
                # --- State Machine Logic ---
                for x1, y1, x2, y2, tid in tracks:
                    tid, bbox = int(tid), (int(x1), int(y1), int(x2), int(y2))
                    cur_pos = np.array([(x1+x2)/2, y2]) # Bottom-center position

                    # Assign a persistent PID to each tracker ID (tid)
                    pid = next((p for p,s in person_states.items() if s.get('tid') == tid), None)
                    if pid is None:
                        pid = next_pid; next_pid += 1
                        person_states[pid] = {'tid': tid, 'state': 'outside_zone', 'prev_pos': cur_pos}
                    
                    st = person_states[pid]
                    st['tid'] = tid # Update tracker ID in case of re-identification
                    
                    # State 1: 'outside_zone' -> Waiting to enter the pink zone
                    if st['state'] == 'outside_zone':
                        if is_point_in_zone(cur_pos, pink_zone):
                            st['state'] = 'inside_zone'
                            print(f"PID {pid}: Entered Pink Zone")
                    
                    # State 2: 'inside_zone' -> Waiting to cross the red line
                    elif st['state'] == 'inside_zone':
                        if not is_point_in_zone(cur_pos, pink_zone): # Exited without crossing red? Reset.
                            st['state'] = 'outside_zone'
                        elif is_top_to_bottom(st['prev_pos'], cur_pos, np.array(red_line[0]), np.array(red_line[1]), neg_is_bottom_red):
                            st['state'] = 'crossed_red'
                            print(f"PID {pid}: Crossed Red Line (Top->Bottom)")

                    # State 3: 'crossed_red' -> Waiting to exit the frame
                    elif st['state'] == 'crossed_red':
                        # Crossed back (bottom->top)? Reset to inside_zone.
                        if is_top_to_bottom(cur_pos, st['prev_pos'], np.array(red_line[0]), np.array(red_line[1]), neg_is_bottom_red):
                             st['state'] = 'inside_zone'
                             print(f"PID {pid}: Crossed Red Line back (Bottom->Top)")

                    st['prev_pos'] = cur_pos.copy() # Update previous position for next frame
                    
                    # --- Drawing ---
                    cv2.rectangle(frame, (bbox[0],bbox[1]),(bbox[2],bbox[3]), (255,255,0), 2)
                    cv2.putText(frame,f'PID:{pid} ({st["state"]})',(bbox[0],max(20,bbox[1]-5)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

                # --- Check for people who have disappeared from the frame ---
                disappeared_pids = [pid for pid, st in person_states.items() if st.get('tid') not in live_tids]
                for pid in disappeared_pids:
                    st = person_states[pid]
                    # If the person was in the 'crossed_red' state when they disappeared, it's a valid count.
                    if st['state'] == 'crossed_red':
                        counts['inbound'] += 1
                        print(f"PID {pid}: Exited frame after crossing red. COUNT = {counts['inbound']}")
                        csvw.writerow([timestamp_str, pid, 'inbound'])
                        st['state'] = 'counted' # Mark as counted to prevent re-counting
                    
                    # If they disappear in any other state, reset them.
                    elif st['state'] != 'counted':
                        st['state'] = 'outside_zone'
                
                # --- UI Display ---
                cv2.rectangle(frame, pink_zone[0], pink_zone[1], (255, 182, 193), 2) # Draw Pink Zone
                cv2.line(frame, red_line[0], red_line[1], (0, 0, 255), 2) # Draw Red Line
                cv2.putText(frame, timestamp_str, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
                cv2.putText(frame, f"Inbound: {counts['inbound']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
                
                cv2.imshow('Video Analysis', cv2.resize(frame, (display_width, display_height)))
                if cv2.waitKey(1) & 0xFF == 27: break
        
        upload_log_to_gdrive(drive, local_log_path, gdrive_log_folder_id)

    finally:
        print("Cleaning up temporary files...")
        if 'cap' in locals() and cap.isOpened(): cap.release()
        cv2.destroyAllWindows()
        if os.path.exists(local_video_path): os.remove(local_video_path)
        if os.path.exists(SNAP_DIR): shutil.rmtree(SNAP_DIR)
        print("Process finished.")

if __name__ == "__main__":
    main()