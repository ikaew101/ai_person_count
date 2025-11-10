import os
import csv
import json

CONFIG_FILE = 'config/camera_config.json'
MASTER_LOG_FILE = 'qa_camera_check/master_video_log.csv'

def create_master_log():
    print(f"Reading config file: {CONFIG_FILE}...")
    try:
        with open(CONFIG_FILE, "r", encoding='utf-8') as f:
            full_config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{CONFIG_FILE}' not found.")
        return

    # รายชื่อกล้องทั้งหมด (เช่น "5033_entrance2", "5129_Banglen-Yao")
    camera_names = full_config.keys()
    
    tasks = []
    for cam_name in camera_names:
        video_path = full_config[cam_name].get('video_path')
        if video_path:
            tasks.append({
                'camera_name': cam_name,
                'video_path': video_path,
                'status': 'pending' # สถานะเริ่มต้น
            })
        else:
            print(f"Warning: Skipping '{cam_name}', 'video_path' not found.")

    if not tasks:
        print("No cameras with video_path found in config.")
        return

    # เขียนไฟล์ Master Log
    try:
        with open(MASTER_LOG_FILE, "w", newline="", encoding='utf-8') as f:
            # ใช้ DictWriter เพื่อให้จัดการง่ายขึ้น
            writer = csv.DictWriter(f, fieldnames=['camera_name', 'video_path', 'status'])
            writer.writeheader() # เขียน Header
            writer.writerows(tasks) # เขียนข้อมูลทั้งหมด
        
        print(f"\nSuccessfully generated '{MASTER_LOG_FILE}' with {len(tasks)} tasks.")
        print("You can now run 'run_processor.py' to start processing.")
        
    except Exception as e:
        print(f"Error writing master log file: {e}")

if __name__ == "__main__":
    create_master_log()