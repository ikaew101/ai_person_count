import os
import json
from pathlib import Path

# --- การตั้งค่า ---
# 1. โฟลเดอร์ที่เก็บวิดีโอ
VIDEO_DIR = "ss_data/vdo" 

# 2. ไฟล์ Config "หลัก" (สำหรับอ่านอย่างเดียว)
MASTER_CONFIG_FILE = "config/camera_config.json"

# 3. ไฟล์ Config "ผลลัพธ์" (สำหรับเขียนกล้องใหม่เท่านั้น)
# (ไฟล์นี้จะถูกเขียนทับทุกครั้งที่รัน)
NEW_CAMERAS_OUTPUT_FILE = "config/new_cameras_to_add.json"

# 4. นามสกุลวิดีโอที่รองรับ
SUPPORTED_EXTENSIONS = ['.mp4', '.avi', '.mkv'] 
# --- จบการตั้งค่า ---

def load_existing_config():
    """โหลด config ที่มีอยู่ ถ้าไม่มีก็คืนค่า dict ว่าง"""
    try:
        with open(MASTER_CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Master '{MASTER_CONFIG_FILE}' not found. Assuming all cameras are new.")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: '{MASTER_CONFIG_FILE}' is corrupted. Please fix it first.")
        return {} # คืนค่าว่างเพื่อป้องกันการทำงานต่อที่ผิดพลาด

def scan_video_folder():
    """สแกนโฟลเดอร์ VIDEO_DIR เพื่อหาไฟล์วิดีโอ"""
    video_files = []
    print(f"Scanning for videos in '{VIDEO_DIR}'...")
    try:
        for file_name in os.listdir(VIDEO_DIR):
            file_ext = Path(file_name).suffix.lower()
            if file_ext in SUPPORTED_EXTENSIONS:
                video_files.append(file_name)
    except FileNotFoundError:
        print(f"Error: Video directory '{VIDEO_DIR}' not found.")
        return []
        
    print(f"Found {len(video_files)} video file(s).")
    return video_files

def create_default_entry(video_filename):
    """สร้างโครงสร้าง JSON เริ่มต้นสำหรับกล้องใหม่"""
    
    camera_key = Path(video_filename).stem
    video_path = f"{VIDEO_DIR.replace(os.sep, '/')}/{video_filename}"
    
    return {
        "file_name": camera_key,
        "video_path": video_path,
        "display_width": 1280,
        "timestamp_roi": [0, 0, 0, 0],
        "start_min": 0,
        "duration_min": 0, 
        "pink_zone": [],
        "lines": {
            "red": [],
            "blue": [],
            "green": [],
            "yellow": []
        }
    }

def main():
    # 1. โหลด Config หลัก
    config_data = load_existing_config()
    
    # 2. สแกนไฟล์วิดีโอ
    video_files = scan_video_folder()

    if not video_files:
        print("No videos found to process.")
        return

    # 3. สร้าง Set ของ "file_name" ที่มีอยู่แล้วใน Config หลัก
    existing_file_names = set()
    for camera_key, camera_info in config_data.items():
        file_name_in_config = camera_info.get("file_name") 
        if file_name_in_config:
            existing_file_names.add(file_name_in_config)

    # --- ( ✨ นี่คือส่วนที่แก้ไข ✨ ) ---
    # 4. สร้าง Dictionary "ใหม่" สำหรับเก็บกล้องที่ขาดหายไปเท่านั้น
    new_cameras_dict = {} 
    
    cameras_already_configured = 0
    total_cameras_found = len(video_files)

    # 5. วนลูปเช็กไฟล์วิดีโอ
    for video_file in video_files:
        video_file_stem = Path(video_file).stem
        
        # 6. เปรียบเทียบ "ชื่อไฟล์" กับ "Set ของ file_name"
        if video_file_stem not in existing_file_names:
            # ถ้าไม่เจอ = เป็นกล้องใหม่
            print(f"[+] Found new camera: '{video_file_stem}'")
            
            new_camera_key = video_file_stem
            new_entry = create_default_entry(video_file)
            
            # เพิ่มลงใน Dictionary ใหม่
            new_cameras_dict[new_camera_key] = new_entry
        else:
            # ถ้าเจอ = มีกล้องนี้ใน config แล้ว
            cameras_already_configured += 1
    
    new_cameras_added = len(new_cameras_dict)
    
    # 7. แสดงผลสรุป
    print("\n" + "="*30)
    print("📊 Bootstrap Summary")
    print("="*30)
    print(f"Total Videos Found:      {total_cameras_found}")
    print(f"Already Configured:    {cameras_already_configured}")
    print(f"New Cameras to Add:    {new_cameras_added}")
    print("="*30)

    # 8. บันทึก "เฉพาะ" กล้องใหม่ ลงในไฟล์ Output
    if new_cameras_added > 0:
        print(f"\nWriting {new_cameras_added} new camera(s) to '{NEW_CAMERAS_OUTPUT_FILE}'...")
        try:
            # สร้างโฟลเดอร์ config ถ้ายังไม่มี
            os.makedirs(os.path.dirname(NEW_CAMERAS_OUTPUT_FILE), exist_ok=True)
            
            # บันทึกไฟล์ JSON (indent=4 เพื่อให้อ่านง่าย)
            # ใช้ "w" (write mode) เพื่อให้ไฟล์นี้ถูก "เขียนทับ" ทุกครั้งที่รัน
            with open(NEW_CAMERAS_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(new_cameras_dict, f, indent=4)
                
            print(f"Successfully created '{NEW_CAMERAS_OUTPUT_FILE}'")
            print("You can now copy-paste these entries into your main config file.")
        except Exception as e:
            print(f"Error writing to '{NEW_CAMERAS_OUTPUT_FILE}': {e}")
    else:
        print("\nAll video files are already in the main config. No new file created.")

if __name__ == "__main__":
    main()