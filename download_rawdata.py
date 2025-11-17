import os
import io
from pathlib import Path # <-- 1. เพิ่ม Import นี้
from googleapiclient.http import MediaIoBaseDownload
import google_auth 

# === การตั้งค่า Path ===
LOCAL_VDO_PATH = "ss_data/vdo"
LOCAL_RAW_DATA_PATH = "ss_data/raw_data"

REMOTE_BASE_FOLDER = "TDG-QA Zonemall"
REMOTE_VIDEO_FOLDER = "SS Video"
REMOTE_DATA_FOLDER = "SS Raw Data"

# === (ฟังก์ชัน Helpers: find_folder_id, list_remote_files, list_local_files, download_file - เหมือนเดิม) ===

def find_folder_id(service, folder_name, parent_id=None):
    """ค้นหา ID ของโฟลเดอร์จากชื่อ"""
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    else:
        query += " and 'root' in parents"
    response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    files = response.get('files', [])
    if not files:
        print(f"Error: Folder '{folder_name}' not found.")
        return None
    return files[0].get('id')

def list_remote_files(service, folder_id):
    """ดึงรายการไฟล์ทั้งหมดในโฟลเดอร์ (ชื่อ, ID)"""
    query = f"'{folder_id}' in parents and trashed = false"
    response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    return response.get('files', [])

def list_local_files(local_path):
    """ดึงรายการไฟล์ทั้งหมดในเครื่อง"""
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    return set(os.listdir(local_path))

def download_file(service, file_id, file_name, local_dest_path):
    """ดาวน์โหลดไฟล์ 1 ไฟล์"""
    print(f"Downloading '{file_name}' to '{local_dest_path}'...")
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(local_dest_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")
        print(f"Successfully downloaded '{file_name}'.")
    except Exception as e:
        print(f"Error downloading {file_name}: {e}")

# --- ( ✨ 2. แก้ไขฟังก์ชันนี้ ✨ ) ---
def sync_folder(service, remote_folder_id, local_folder_path):
    """
    ฟังก์ชันหลัก: ดาวน์โหลดเฉพาะไฟล์ที่ยังไม่มีในเครื่อง
    (แก้ไข: ให้คืนค่า Set ของไฟล์ทั้งหมดในเครื่องหลัง Sync)
    """
    print(f"\n--- Starting Sync for: {local_folder_path} ---")
    
    remote_files = list_remote_files(service, remote_folder_id)
    if not remote_files:
        print("No remote files found.")
        return set() # คืนค่า Set ว่าง

    local_files = list_local_files(local_folder_path)
    final_local_file_set = local_files.copy() # <-- (เพิ่ม) สร้าง Set สำหรับผลลัพธ์
    
    download_count = 0
    for item in remote_files:
        file_name = item['name']
        file_id = item['id']
        
        if file_name not in local_files:
            download_count += 1
            local_dest_path = os.path.join(local_folder_path, file_name)
            download_file(service, file_id, file_name, local_dest_path)
            final_local_file_set.add(file_name) # <-- (เพิ่ม) Add ไฟล์ใหม่ลง Set
        else:
            print(f"Skipping '{file_name}' (already exists).")
            
    print(f"Sync complete. Downloaded {download_count} new file(s).")
    return final_local_file_set # <-- (แก้ไข) คืนค่า Set ผลลัพธ์

# --- ( ✨ 3. เพิ่มฟังก์ชันใหม่นี้ ✨ ) ---
def check_file_matching(video_file_set, data_file_set):
    """
    เปรียบเทียบ Set ของไฟล์วิดีโอและไฟล์ Excel
    โดยใช้ชื่อไฟล์ (ไม่รวมนามสกุล)
    """
    print("\n" + "="*40)
    print("📊 Starting File Match Verification")
    print("="*40)

    # 1. สกัด "Stem" (ชื่อไฟล์ไม่รวมนามสกุล)
    video_stems = set(Path(f).stem for f in video_file_set if f.endswith(('.mp4', '.avi', '.mkv')))
    data_stems = set(Path(f).stem for f in data_file_set if f.endswith('.xlsx'))
    
    print(f"Found {len(video_stems)} video stems (e.g., .mp4)")
    print(f"Found {len(data_stems)} data stems (e.g., .xlsx)")

    # 2. เปรียบเทียบ Set
    missing_in_data = video_stems - data_stems  # วิดีโอที่ไม่มี Excel
    missing_in_video = data_stems - video_stems  # Excel ที่ไม่มีวิดีโอ

    all_good = True

    # 3. รายงานผล
    if missing_in_data:
        all_good = False
        print("\n--- 🔴 WARNING: Mismatches Found ---")
        print("The following VIDEO files are MISSING a matching .xlsx file:")
        for file_stem in sorted(missing_in_data):
            print(f"  - {file_stem}") # (แสดงแค่ชื่อ)
    
    if missing_in_video:
        all_good = False
        if not missing_in_data: # พิมพ์ Header แค่ครั้งเดียว
             print("\n--- 🔴 WARNING: Mismatches Found ---")
        print("The following .xlsx files are MISSING a matching video file:")
        for file_stem in sorted(missing_in_video):
            print(f"  - {file_stem}") # (แสดงแค่ชื่อ)

    if all_good:
        print("\n--- ✅ SUCCESS ---")
        print("All video files have a matching .xlsx file.")
    
    print("="*40)
    print("Match verification complete.")
# --- (สิ้นสุดการเพิ่ม) ---


# === (สคริปต์หลัก) ===
def main():
    print("Connecting to Google Drive...")
    service = google_auth.get_drive_service()
    if not service:
        print("Failed to connect to Google Drive. Exiting.")
        return

    print("Finding remote folder IDs...")
    base_folder_id = find_folder_id(service, REMOTE_BASE_FOLDER)
    if not base_folder_id:
        return

    video_folder_id = find_folder_id(service, REMOTE_VIDEO_FOLDER, base_folder_id)
    data_folder_id = find_folder_id(service, REMOTE_DATA_FOLDER, base_folder_id)

    # --- ( ✨ 4. แก้ไขส่วน Sync ✨ ) ---
    video_files = set() # สร้าง Set ว่างไว้ก่อน
    data_files = set()  # สร้าง Set ว่างไว้ก่อน

    # 3. เริ่ม Sync
    if video_folder_id:
        # (แก้ไข) รับค่า Set ที่ Sync เสร็จแล้ว
        video_files = sync_folder(service, video_folder_id, LOCAL_VDO_PATH)
    else:
        print(f"Could not sync '{REMOTE_VIDEO_FOLDER}' (ID not found).")
        
    if data_folder_id:
        # (แก้ไข) รับค่า Set ที่ Sync เสร็จแล้ว
        data_files = sync_folder(service, data_folder_id, LOCAL_RAW_DATA_PATH)
    else:
        print(f"Could not sync '{REMOTE_DATA_FOLDER}' (ID not found).")

    # 4. (เพิ่ม) เรียกใช้ฟังก์ชันตรวจสอบไฟล์
    check_file_matching(video_files, data_files)
    # --- (สิ้นสุดการแก้ไข) ---

if __name__ == '__main__':
    main()