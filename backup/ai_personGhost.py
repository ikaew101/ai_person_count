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
CONFIG_FILE = 'core/config_log_generator.json'

# =================== MODEL / TRACKER (Global) ====================
print("Loading AI model...")
model = YOLO("yolov8n.pt", verbose=False)
try:
    tracker = Sort(max_age=120, min_hits=3, iou_threshold=0.2)
except TypeError:
    tracker = Sort(max_age=120, min_hits=3)
print("Model loaded successfully.")

# ====================== GOOGLE DRIVE HELPERS =========================
# ... (คัดลอกฟังก์ชัน authenticate_gdrive, download_video_from_gdrive, upload_log_to_gdrive มาวางตรงนี้ได้เลย) ...
def authenticate_gdrive():
    print("Connecting to Google Drive...")
    settings = {
        "client_config_file": os.path.join("core", "client_secrets.json"),
        "save_credentials": True, "get_refresh_token": True,
        "save_credentials_backend": "file",
        "save_credentials_file": os.path.join("core", "credentials.json")
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

# ====================== OCR HELPER =========================
# ... (คัดลอกฟังก์ชัน get_timestamp_from_frame มาวางตรงนี้) ...
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

# ====================== HELPERS (from ai_personCount) =========================
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

def point_to_line_distance(p,a,b):
    d=b-a; t=max(0.,min(1.,(p-a).dot(d)/d.dot(d))); proj=a+t*d; return np.linalg.norm(p-proj)

# --- NEW: ฟังก์ชันสำหรับสร้างไฟล์ Log ใหม่ ---
def create_new_log_file(log_dir, camera_name):
    """ปิดไฟล์ Log เก่า (ถ้ามี) และสร้างไฟล์ใหม่พร้อม Header"""
    local_log_path = os.path.join(log_dir, f"log_{camera_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    csv_file = open(local_log_path, "w", newline="", encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Timestamp", "PID", "Event", "Total"])
    print(f"Created new log file: {os.path.basename(local_log_path)}")
    return local_log_path, csv_file, csv_writer
# --- END NEW ---

# ====================== MAIN LOGIC =========================
def main():
    parser = argparse.ArgumentParser(description="Merged Person Counting Tool")
    parser.add_argument("camera_name", help="Name of the camera config to use.")
    args = parser.parse_args()

    with open(CONFIG_FILE, "r", encoding='utf-8') as f: full_config = json.load(f)
    if args.camera_name not in full_config['cameras']: raise SystemExit(f"Camera not found.")
    config = full_config['cameras'][args.camera_name]
    gdrive_log_folder_id = full_config.get("gdrive_log_folder_id")
    
    video_filename = config['video_filename_gdrive']
    timestamp_roi = config['timestamp_roi']
    max_w = config.get("max_display_width", 1280)
    
    # --- MODIFIED: แยกเส้นกรอบ (สีขาว) และเส้นนับ (สีแดง) ---
    # สร้างเส้นกรอบ 4 เส้นจากพิกัด top, left, right, bottom
    top_left = tuple(config['lines']['left'][0])
    top_right = tuple(config['lines']['right'][0])
    bottom_left = tuple(config['lines']['left'][1])
    bottom_right = tuple(config['lines']['right'][1])

    white_lines = [
        (top_left, top_right),
        (top_right, bottom_right),
        (bottom_right, bottom_left),
        (bottom_left, top_left)
    ]

    red_line = tuple(map(tuple, config['lines']['red']))
    
    local_video_path = f"temp_{video_filename}"
    log_dir = "temp_logs"
    os.makedirs(log_dir, exist_ok=True)
    local_log_path = os.path.join(log_dir, f"log_{args.camera_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

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
        
        counts = {"right": 0, "left": 0, "straight": 0, "inbound": 0}
        person_states = {}
        next_pid = 1
        neg_is_bottom_red = make_side_label(red_line[0], red_line[1])
        
        # --- NEW: เพิ่มตัวแปรสำหรับ Mouse Hover ---
        global mouse_pos_raw
        mouse_pos_raw = (-1, -1)

        def _on_mouse(event, x, y, flags, param):
            """ฟังก์ชันที่จะถูกเรียกเมื่อมี event ของเมาส์เกิดขึ้น"""
            global mouse_pos_raw
            # คำนวณพิกัดบนวิดีโอขนาดจริง
            rx = int(x * original_w / display_width)
            ry = int(y * original_h / display_height)
            mouse_pos_raw = (rx, ry)
            
            # ถ้ามีการคลิกซ้าย ให้พิมพ์พิกัดออกทาง console
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"[CLICK] raw_coords=({rx},{ry})")

        cv2.setMouseCallback("Video Analysis", _on_mouse)
        # --- END NEW ---

        paused = False

        with open(local_log_path, "w", newline="", encoding='utf-8') as csv_file:
            csvw = csv.writer(csv_file)
            csvw.writerow(["Timestamp", "PID", "Event", "Total"])

            while True:
                # --- MODIFIED: อ่านเฟรมใหม่เมื่อไม่ได้ Pause ---
                if not paused:
                    ret, frame = cap.read()
                    if not ret: break
                    last_frame = frame.copy() # เก็บสำเนาของเฟรมล่าสุดไว้
                
                # ถ้าไม่มีเฟรมล่าสุด (วิดีโอยังไม่เริ่ม) ให้ข้ามไป
                if last_frame is None:
                    continue

                frame = last_frame.copy()
                # --- END MODIFIED ---

                ocr_timestamp = get_timestamp_from_frame(frame, timestamp_roi)
                timestamp_str = ocr_timestamp.strftime('%d-%m-%Y %H:%M:%S') if ocr_timestamp else "N/A"

                dets = []
                for r in model(frame, stream=True, conf=0.35):
                    for box in r.boxes.data:
                        if len(box) >= 6 and int(box[5]) == 0:
                            dets.append([int(b) for b in box[:4]] + [float(box[4])])
                
                tracks = tracker.update(np.array(dets) if dets else np.empty((0, 5)))
                
                live_tids = {int(t[4]) for t in tracks}
                
                # --- ใช้ Logic การนับจาก ai_personCount เดิม ---
                for x1, y1, x2, y2, tid in tracks:
                    tid, bbox = int(tid), (int(x1), int(y1), int(x2), int(y2))
                    cur_c = np.array([(x1+x2)/2, (y1+y2)/2])
                    cur_b = np.array([(x1+x2)/2, y2])
                    
                    pid = next((p for p,s in person_states.items() if s.get('tid') == tid), None)
                    if pid is None:
                        pid = next_pid; next_pid += 1
                        person_states[pid] = {'tid': tid, 'last_seen': datetime.now()}
                    
                    st = person_states[pid]
                    st['last_pos_center'] = cur_c; st['last_pos_bottom'] = cur_b; st['tid'] = tid
                    
                    if 'prev_pos_center' not in st: st['prev_pos_center'] = cur_c
                    if 'prev_pos_bottom' not in st: st['prev_pos_bottom'] = cur_b
                    
                    # Logic การข้ามเส้นสีแดง (Inbound)
                    if is_top_to_bottom(st['prev_pos_center'], cur_c, np.array(red_line[0]), np.array(red_line[1]), neg_is_bottom_red):
                        if not st.get('has_entered', False):
                            st['has_entered'] = True; st['destination'] = None
                            counts['inbound'] += 1
                            csvw.writerow([timestamp_str, pid, 'inbound', counts['inbound']])
                    
                    # Logic การหาปลายทาง
                    if st.get('has_entered', False) and st['destination'] is None:
                        crossed = None
                        # if is_crossing_line(st['prev_pos_bottom'], cur_b, np.array(right_line[0]), np.array(right_line[1])): crossed = 'right'
                        # elif is_crossing_line(st['prev_pos_bottom'], cur_b, np.array(left_line[0]), np.array(left_line[1])): crossed = 'left'
                        # elif is_crossing_line(st['prev_pos_bottom'], cur_b, np.array(bottom_line[0]), np.array(bottom_line[1])): crossed = 'straight'
                        
                        if crossed:
                            st['destination'] = crossed; counts[crossed] += 1; st['has_entered'] = False
                            csvw.writerow([timestamp_str, pid, crossed, counts[crossed]])
                    
                    st['prev_pos_center'] = cur_c.copy(); st['prev_pos_bottom'] = cur_b.copy()
                    
                    # วาด Bbox + PID
                    cv2.rectangle(frame, (bbox[0],bbox[1]),(bbox[2],bbox[3]), (255,255,0), 2)
                    cv2.putText(frame,f'PID:{pid}',(bbox[0],max(20,bbox[1]-5)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

                # --- MODIFIED: แสดงผล UI (วาดกรอบขาวและเส้นแดง) ---
                # วาดกรอบสีขาว
                for line in white_lines:
                    cv2.line(frame, line[0], line[1], (255, 255, 255), 2)
                
                # วาดเส้นสีแดง
                cv2.line(frame, red_line[0], red_line[1], (0, 0, 255), 2) # ทำให้หนาขึ้นเล็กน้อย
                # --- END MODIFIED ---

                # --- NEW: เพิ่ม Text Overlay ---
                cv2.putText(frame, f"Inbound: {counts['inbound']}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                # --- END NEW ---

                # --- NEW: วาด Mouse Hover Overlay ---
                if mouse_pos_raw[0] >= 0:
                    # แปลงพิกัดจริงกลับมาเป็นพิกัดบนหน้าจอแสดงผล
                    disp_x = int(mouse_pos_raw[0] * display_width / original_w)
                    disp_y = int(mouse_pos_raw[1] * display_height / original_h)
                    
                    # วาดกากบาท
                    cv2.drawMarker(frame, (mouse_pos_raw[0], mouse_pos_raw[1]), 
                                   (0, 255, 255), markerType=cv2.MARKER_CROSS, 
                                   markerSize=20, thickness=2)
                    
                    # วาดข้อความพิกัด
                    text = f"x:{mouse_pos_raw[0]} y:{mouse_pos_raw[1]}"
                    cv2.putText(frame, text, (mouse_pos_raw[0] + 15, mouse_pos_raw[1] - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                # --- END NEW ---
                
                cv2.imshow('Video Analysis', cv2.resize(frame, (display_width, display_height)))
                # --- MODIFIED: เพิ่ม Key ลัด ---
                k = cv2.waitKey(1) & 0xFF
                if k == 27: # กด Esc เพื่อออก
                    break
                elif k == ord('p'): # กด p เพื่อ Pause/Play
                    paused = not paused
                    
                    # --- NEW: ถ้ากด Pause ให้ทำการอัปโหลด Log ---
                    if paused:
                        print("\n--- PAUSED ---")
                        # 1. ปิดไฟล์ Log ปัจจุบัน
                        csv_file.close()
                        # 2. อัปโหลดไฟล์ Log นั้นขึ้น Google Drive
                        upload_log_to_gdrive(drive, local_log_path, gdrive_log_folder_id)
                        # 3. สร้างไฟล์ Log ใหม่สำหรับรอบต่อไป
                        local_log_path, csv_file, csvw = create_new_log_file(log_dir, args.camera_name)
                        print("--- RESUME by pressing 'p' again ---\n")
                    # --- END NEW ---
        
        upload_log_to_gdrive(drive, local_log_path, gdrive_log_folder_id)

    finally:
        print("Cleaning up temporary files...")
        if 'cap' in locals() and cap.isOpened(): cap.release()
        cv2.destroyAllWindows()
        if os.path.exists(local_video_path): os.remove(local_video_path)
        if os.path.exists(log_dir): shutil.rmtree(log_dir)
        print("Process finished.")

if __name__ == "__main__":
    main()