import os
import csv
import subprocess
import sys
import time
import json # (เพิ่ม)

MASTER_LOG_FILE = 'qa_camera_check/master_video_log.csv'
CONFIG_FILE = 'config/camera_config.json'
PYTHON_COMMAND = sys.executable # ใช้ Python ตัวเดียวกับที่รันสคริปต์นี้ (เช่น python.exe)

def read_all_tasks():
    """อ่าน CSV ทั้งหมดมาเก็บใน List of Dictionaries"""
    if not os.path.exists(MASTER_LOG_FILE):
        return None, None
    with open(MASTER_LOG_FILE, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames if reader.fieldnames else ['camera_name', 'video_path', 'status']
        return [row for row in reader], fieldnames

def write_all_tasks(tasks, fieldnames):
    """เขียน List of Dictionaries ทับ CSV ทั้งไฟล์"""
    with open(MASTER_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tasks)

def find_next_task(tasks):
    """หา task 'failed' ก่อน, ถ้าไม่เจอก็ค่อยหา 'pending'"""
    for task in tasks:
        if task['status'] == 'failed':
            return task
    for task in tasks:
        if task['status'] == 'pending':
            return task
    return None # ไม่เหลือ task

def update_task_status(tasks, camera_name, new_status):
    """อัปเดต status ใน list (in-memory)"""
    for task in tasks:
        if task['camera_name'] == camera_name:
            task['status'] = new_status
            return

def main_processor():
    # --- โหลด Config ของกล้อง (สำหรับส่ง Arguments) ---
    try:
        with open(CONFIG_FILE,"r",encoding='utf-8') as f: 
            camera_configs = json.load(f)
    except: 
        print(f"Warning: '{CONFIG_FILE}' not found. Cannot pass arguments like --start_min.")
        camera_configs = {}

    while True:
        tasks, fieldnames = read_all_tasks()
        if tasks is None:
            print(f"Error: '{MASTER_LOG_FILE}' not found.")
            print("Please run 'python generate_master_log.py' first.")
            break
        
        task_to_run = find_next_task(tasks)
        
        if task_to_run is None:
            print("All tasks completed. Exiting.")
            break
            
        task_camera_name = task_to_run['camera_name']
        print(f"\n===========================================")
        print(f"Found task: '{task_camera_name}' (Status: {task_to_run['status']})")
        
        # 1. อัปเดตสถานะเป็น 'running' และเขียนลง CSV
        update_task_status(tasks, task_camera_name, 'running')
        write_all_tasks(tasks, fieldnames)
        print(f"Status set to 'running'. Executing 'ai_personCount.py'...")

        new_status = 'failed' # ตั้งค่าเริ่มต้นว่าล้มเหลว
        try:
            # 2. รันสคริปต์หลัก (ai_personCount.py)
            
            # --- (สำคัญ) สร้าง List คำสั่ง ---
            command = [
                PYTHON_COMMAND, 
                'ai_personCount.py', # (หรือ final_person_counter.py ถ้าคุณใช้ชื่อนั้น)
                task_camera_name,
            ]
            
            # --- (ตัวอย่าง) การเพิ่ม Arguments ถ้าคุณเก็บไว้ใน Config ---
            # cam_config = camera_configs.get(task_camera_name, {})
            # if cam_config.get("start_min"):
            #     command.extend(["--start_min", str(cam_config["start_min"])])
            # if cam_config.get("duration_min"):
            #     command.extend(["--duration_min", str(cam_config["duration_min"])])

            
            # รันและรอจนจบ
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
            stdout, stderr = process.communicate() # รอจนจบ
            
            if process.returncode == 0:
                print(f"Successfully processed '{task_camera_name}'.")
                print("---------- Output (from ai_personCount.py) ----------")
                print(stdout)
                print("-----------------------------------------------------")
                new_status = 'completed'
            else:
                print(f"!!! FAILED to process '{task_camera_name}' !!!")
                print("---------- Error (from ai_personCount.py) ----------")
                print(stderr)
                print("----------------------------------------------------")
                new_status = 'failed'

        except KeyboardInterrupt:
            print("\nBatch processing interrupted by user.")
            print("Setting current task status back to 'pending'.")
            new_status = 'pending'
            # ฆ่า process ที่กำลังรัน (ถ้ายังอยู่)
            process.terminate()
            time.sleep(1) # รอ process ปิด
            # อัปเดตสถานะทันที
            tasks, fieldnames = read_all_tasks() # อ่านใหม่
            update_task_status(tasks, task_camera_name, new_status)
            write_all_tasks(tasks, fieldnames)
            print(f"Status for '{task_camera_name}' set to 'pending'. Exiting.")
            break # ออกจาก while loop
        
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            new_status = 'failed'
        
        # 6. อัปเดตสถานะสุดท้าย (completed หรือ failed)
        tasks, fieldnames = read_all_tasks() # อ่านใหม่
        update_task_status(tasks, task_camera_name, new_status)
        write_all_tasks(tasks, fieldnames)
        print(f"Status for '{task_camera_name}' set to '{new_status}'.")
        
        time.sleep(1) # พัก 1 วิ

if __name__ == "__main__":
    if not os.path.exists(MASTER_LOG_FILE):
        print(f"Error: '{MASTER_LOG_FILE}' not found.")
        print("Please run 'python generate_master_log.py' first to create the task list.")
    else:
        main_processor()