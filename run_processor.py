import os
import csv
import subprocess
import sys
import time
import json
import csv_validator
import generate_master_log
import google_auth

from googleapiclient.http import MediaFileUpload

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

# === (ฟังก์ชัน Helpers สำหรับ Google Drive Upload) ===
def find_or_create_folder(service, folder_name, parent_id=None):
    """ค้นหาโฟลเดอร์ ถ้าไม่เจอให้สร้างใหม่"""
    # 1. ค้นหาก่อน
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    else:
        query += " and 'root' in parents"

    response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    files = response.get('files', [])

    if files:
        # ถ้าเจอ
        return files[0].get('id')
    else:
        # ถ้าไม่เจอ, สร้างใหม่
        print(f"Folder '{folder_name}' not found, creating...")
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_id] if parent_id else []
        }
        folder = service.files().create(body=file_metadata, fields='id').execute()
        print(f"Created folder '{folder_name}' (ID: {folder.get('id')})")
        return folder.get('id')

def upload_file(service, local_file_path, remote_folder_id):
    """อัปโหลดไฟล์ 1 ไฟล์"""
    file_name = os.path.basename(local_file_path)
    print(f"Uploading '{file_name}' to Drive...")
    try:
        media = MediaFileUpload(local_file_path, resumable=True)

        # ตรวจสอบว่ามีไฟล์ชื่อนี้อยู่แล้วหรือไม่ (เพื่ออัปเดตแทนการสร้างซ้ำ)
        query = f"name='{file_name}' and '{remote_folder_id}' in parents"
        response = service.files().list(q=query, spaces='drive', fields='files(id)').execute()
        files = response.get('files', [])

        if files:
            # --- (แก้ไข) ---
            # ไฟล์มีอยู่แล้ว -> อัปเดต
            # เราจะส่ง Body ที่มีแค่ 'name' (ห้ามส่ง 'parents')
            update_metadata = {'name': file_name}
            file_id = files[0].get('id')
            
            service.files().update(
                fileId=file_id,
                body=update_metadata, # <--- ใช้ metadata ที่ไม่มี 'parents'
                media_body=media
            ).execute()
            print(f"Updated '{file_name}' in Drive.")
            # --- (สิ้นสุดการแก้ไข) ---
            
        else:
            # --- (เหมือนเดิม) ---
            # ไฟล์ไม่มี -> สร้างใหม่
            # เราต้องส่ง 'parents' ใน Body
            create_metadata = {
                'name': file_name,
                'parents': [remote_folder_id]
            }
            service.files().create(
                body=create_metadata, # <--- ใช้ metadata ที่มี 'parents'
                media_body=media
            ).execute()
            print(f"Created '{file_name}' in Drive.")
            # --- (สิ้นสุดส่วนเหมือนเดิม) ---

    except Exception as e:
        print(f"Error uploading {file_name}: {e}")
        
def upload_folder_recursive(service, local_folder, remote_parent_folder_id):
    """อัปโหลดทุกอย่างในโฟลเดอร์ (รวมถึงโฟลเดอร์ย่อย)"""
    print(f"\nUploading contents of '{local_folder}'...")
    folder_name = os.path.basename(local_folder)

    # 1. สร้างโฟลเดอร์ปลายทาง (เช่น 'Camera', 'Output')
    remote_folder_id = find_or_create_folder(service, folder_name, remote_parent_folder_id)

    # 2. วนลูปไฟล์/โฟลเดอร์ ในเครื่อง
    for item_name in os.listdir(local_folder):
        local_item_path = os.path.join(local_folder, item_name)

        if os.path.isdir(local_item_path):
            # ถ้าเป็นโฟลเดอร์ -> เรียกตัวเองซ้ำ (Recursive)
            upload_folder_recursive(service, local_item_path, remote_folder_id)
        elif os.path.isfile(local_item_path):
            # ถ้าเป็นไฟล์ -> อัปโหลด
            upload_file(service, local_item_path, remote_folder_id)

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
            print("All AI tasks completed.")

            print("\n===========================================")
            print("🚀 Starting Data Validation step...")
            print("===========================================")
            try:
                csv_validator.process_data_validation()
                print("✅ Validation step completed successfully.")
            except Exception as e:
                print(f"!!! ERROR during validation step: {e}")

            # --- 🔽 บล็อกนี้ทั้งหมดต้อง "ย่อหน้า" เข้ามา ---
            print("\n===========================================")
            print("🚀 Starting Google Drive Upload step...")
            print("===========================================")
            try:
                service = google_auth.get_drive_service()
                if service:
                    # 1. หา ID โฟลเดอร์หลัก
                    base_id = find_or_create_folder(service, "TDG-QA Zonemall")

                    # 2. หา ID โฟลเดอร์ QA Camera
                    qa_camera_id = find_or_create_folder(service, "QA Camera", base_id)

                    # 3. หา ID โฟลเดอร์ย่อย
                    output_id = find_or_create_folder(service, "Output", qa_camera_id)
                    camera_id = find_or_create_folder(service, "Camera", qa_camera_id)
                    ai_result_id = find_or_create_folder(service, "AI Result", qa_camera_id)

                    # 4. อัปโหลดไฟล์และโฟลเดอร์

                    # 4.1 อัปโหลด master_video_log.csv
                    upload_file(service, MASTER_LOG_FILE, qa_camera_id) #

                    # 4.2 อัปโหลดโฟลเดอร์ AI Result (หาไฟล์ validation_{date}.csv)
                    ai_result_path = "qa_camera_check/ai_result" #
                    for f_name in os.listdir(ai_result_path):
                        if "validation_" in f_name and f_name.endswith(".csv"):
                            upload_file(service, os.path.join(ai_result_path, f_name), ai_result_id)

                    # 4.3 อัปโหลดโฟลเดอร์ Output (แบบไม่ recursive เพราะมีแต่ไฟล์)
                    output_path = "qa_camera_check/output" #
                    for f_name in os.listdir(output_path):
                        f_path = os.path.join(output_path, f_name)
                        if os.path.isfile(f_path):
                            upload_file(service, f_path, output_id)

                    # 4.4 อัปโหลดโฟลเดอร์ Camera (แบบ Recursive)
                    # (เราจะอัปโหลดทั้งโฟลเดอร์ 'camera' ไปไว้ใน 'Camera')
                    
                    # upload_folder_recursive(service, "qa_camera_check/camera", qa_camera_id)

                    print("✅ Google Drive Upload completed.")
                else:
                    print("!!! ERROR: Could not connect to Google Drive for upload.")
            except Exception as e:
                print(f"!!! ERROR during Google Drive Upload step: {e}")

            print("All processes finished. Exiting.")
            break # 
            
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
            cam_config = camera_configs.get(task_camera_name, {})
            if cam_config.get("start_min"):
                command.extend(["--start_min", str(cam_config["start_min"])])
            if cam_config.get("duration_min"):
                command.extend(["--duration_min", str(cam_config["duration_min"])])

            
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
        print(f"Warning: '{MASTER_LOG_FILE}' not found.")
        print("Attempting to run 'generate_master_log.py' automatically...")
        
        try:
            # 1. เรียกฟังก์ชัน create_master_log จากไฟล์ที่ import มา
            generate_master_log.create_master_log()
            
            # 2. ตรวจสอบอีกครั้งว่าไฟล์ถูกสร้างสำเร็จหรือไม่
            if os.path.exists(MASTER_LOG_FILE):
                print(f"Successfully generated '{MASTER_LOG_FILE}'.")
                print("Proceeding with processor...")
                main_processor() # 3. ถ้าสำเร็จ ก็เริ่มทำงานหลักต่อ
            else:
                print(f"!!! ERROR: 'generate_master_log.py' ran but failed to create the file.")
                print("Please check 'config/camera_config.json' and permissions.")

        except Exception as e:
            print(f"!!! FAILED to run 'generate_master_log.py': {e}")
            print("Please fix 'generate_master_log.py' or 'config/camera_config.json' and try again.")
    else:
        print("Master log found. Starting processor...")
        main_processor()