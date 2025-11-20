import os
import io
import json # <-- 1. เพิ่ม import json
from pathlib import Path
from googleapiclient.http import MediaIoBaseDownload
import google_auth 

# === การตั้งค่า Path ===
LOCAL_VDO_PATH = "ss_data/vdo"
LOCAL_RAW_DATA_PATH = "ss_data/raw_data"
CONFIG_FILE = 'config/camera_config.json' # <-- 2. เพิ่ม Path Config

REMOTE_BASE_FOLDER = "TDG-QA Zonemall"
REMOTE_VIDEO_FOLDER = "SS Video"
REMOTE_DATA_FOLDER = "SS Raw Data"

# --- ( ✨ 3. เพิ่มฟังก์ชันโหลด Config เพื่อหาไฟล์ที่ Active ✨ ) ---
def get_active_filenames():
    """
    อ่าน Config และคืนค่า Set ของชื่อไฟล์ (Video & Excel) ที่ Active
    """
    print(f"Reading config: {CONFIG_FILE}...")
    allowed_files = set()
    
    try:
        with open(CONFIG_FILE, "r", encoding='utf-8') as f:
            full_config = json.load(f)
            
        for cam_name, data in full_config.items():
            # เช็คว่า Active หรือไม่ (Default = True)
            if data.get('active', True):
                # 1. ดึงชื่อไฟล์วิดีโอจาก video_path
                video_path = data.get('video_path')
                if video_path:
                    video_filename = os.path.basename(video_path)
                    allowed_files.add(video_filename)
                    
                    # 2. (แถม) ดึงชื่อไฟล์ Excel ด้วย (สมมติว่าชื่อเหมือนกันเปลี่ยนแค่นามสกุล)
                    # หรือถ้าคุณมี field 'excel_path' ก็ดึงจากตรงนั้นได้
                    # แต่ปกติระบบคุณใช้ชื่อไฟล์เดียวกัน (stem)
                    stem = Path(video_filename).stem
                    allowed_files.add(f"{stem}.xlsx")
                    
        print(f"Found {len(allowed_files)} active files in config whitelist.")
        return allowed_files
        
    except Exception as e:
        print(f"Warning: Could not read config for filtering ({e}). Downloading ALL files.")
        return None # คืนค่า None แปลว่า "เอาหมด"
# -----------------------------------------------------------

# === (ฟังก์ชัน Helpers เดิม: find_folder_id, list_remote_files, list_local_files, download_file) ===
# (คงเดิมไว้ ไม่ต้องแก้)

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
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    return set(os.listdir(local_path))

def download_file(service, file_id, file_name, local_dest_path):
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

# --- ( ✨ 4. แก้ไขฟังก์ชัน Sync ให้รองรับ Whitelist ✨ ) ---
def sync_folder(service, remote_folder_id, local_folder_path, whitelist=None):
    """
    ดาวน์โหลดไฟล์ (ถ้า whitelist ไม่เป็น None จะโหลดเฉพาะที่มีชื่อใน whitelist)
    """
    print(f"\n--- Starting Sync for: {local_folder_path} ---")
    
    remote_files = list_remote_files(service, remote_folder_id)
    if not remote_files:
        print("No remote files found.")
        return set()

    local_files = list_local_files(local_folder_path)
    final_local_file_set = local_files.copy()
    
    download_count = 0
    skipped_count = 0
    
    for item in remote_files:
        file_name = item['name']
        file_id = item['id']
        
        # --- Logic การกรอง ---
        if whitelist is not None:
            if file_name not in whitelist:
                # ถ้าไม่อยู่ในรายการที่ Active -> ข้าม
                # print(f"Skipping '{file_name}' (Not active in config).") 
                continue 
        # --------------------

        if file_name not in local_files:
            download_count += 1
            local_dest_path = os.path.join(local_folder_path, file_name)
            download_file(service, file_id, file_name, local_dest_path)
            final_local_file_set.add(file_name)
        else:
            print(f"Skipping '{file_name}' (already exists).")
            skipped_count += 1
            
    print(f"Sync complete. Downloaded: {download_count}, Skipped: {skipped_count}")
    return final_local_file_set

def check_file_matching(video_file_set, data_file_set):
    # (ฟังก์ชันเดิม ไม่ต้องแก้)
    print("\n" + "="*40)
    print("📊 Starting File Match Verification")
    print("="*40)
    video_stems = set(Path(f).stem for f in video_file_set if f.endswith(('.mp4', '.avi', '.mkv')))
    data_stems = set(Path(f).stem for f in data_file_set if f.endswith('.xlsx'))
    
    print(f"Found {len(video_stems)} video stems")
    print(f"Found {len(data_stems)} data stems")

    missing_in_data = video_stems - data_stems
    missing_in_video = data_stems - video_stems
    all_good = True

    if missing_in_data:
        all_good = False
        print("\n--- 🔴 WARNING: Mismatches Found ---")
        print("The following VIDEO files are MISSING a matching .xlsx file:")
        for file_stem in sorted(missing_in_data):
            print(f"  - {file_stem}")
    
    if missing_in_video:
        all_good = False
        if not missing_in_data: print("\n--- 🔴 WARNING: Mismatches Found ---")
        print("The following .xlsx files are MISSING a matching video file:")
        for file_stem in sorted(missing_in_video):
            print(f"  - {file_stem}")

    if all_good:
        print("\n--- ✅ SUCCESS ---")
        print("All video files have a matching .xlsx file.")
    print("="*40)

# === (สคริปต์หลัก) ===
def main():
    # 1. เตรียม Whitelist
    active_files_whitelist = get_active_filenames() # <-- เรียกฟังก์ชันใหม่

    print("Connecting to Google Drive...")
    service = google_auth.get_drive_service()
    if not service:
        print("Failed to connect to Google Drive. Exiting.")
        return

    print("Finding remote folder IDs...")
    base_folder_id = find_folder_id(service, REMOTE_BASE_FOLDER)
    if not base_folder_id: return

    video_folder_id = find_folder_id(service, REMOTE_VIDEO_FOLDER, base_folder_id)
    data_folder_id = find_folder_id(service, REMOTE_DATA_FOLDER, base_folder_id)

    video_files = set()
    data_files = set()

    # 3. เริ่ม Sync (ส่ง whitelist ไปด้วย)
    if video_folder_id:
        video_files = sync_folder(service, video_folder_id, LOCAL_VDO_PATH, active_files_whitelist)
    else:
        print(f"Could not sync '{REMOTE_VIDEO_FOLDER}' (ID not found).")
        
    if data_folder_id:
        data_files = sync_folder(service, data_folder_id, LOCAL_RAW_DATA_PATH, active_files_whitelist)
    else:
        print(f"Could not sync '{REMOTE_DATA_FOLDER}' (ID not found).")

    # 4. ตรวจสอบ
    check_file_matching(video_files, data_files)

if __name__ == '__main__':
    main()