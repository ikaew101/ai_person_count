import os
import io
from googleapiclient.http import MediaIoBaseDownload
import google_auth # Import ไฟล์ที่เราเพิ่งสร้าง

# === การตั้งค่า Path ===
# (ตรวจสอบให้แน่ใจว่า Path เหล่านี้ตรงกับโครงสร้างของคุณ)
LOCAL_VDO_PATH = "ss_data/vdo"
LOCAL_RAW_DATA_PATH = "ss_data/raw_data"

REMOTE_BASE_FOLDER = "TDG-QA Zonemall"
REMOTE_VIDEO_FOLDER = "SS Video"
REMOTE_DATA_FOLDER = "SS Raw Data"

# === (ฟังก์ชัน Helpers) ===

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
    query = f"'{folder_id}' in parents"
    response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    return response.get('files', [])

def list_local_files(local_path):
    """ดึงรายการไฟล์ทั้งหมดในเครื่อง"""
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    return set(os.listdir(local_path)) # ใช้ set เพื่อให้ค้นหาเร็ว

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

def sync_folder(service, remote_folder_id, local_folder_path):
    """
    ฟังก์ชันหลัก: ดาวน์โหลดเฉพาะไฟล์ที่ยังไม่มีในเครื่อง
    """
    print(f"\n--- Starting Sync for: {local_folder_path} ---")
    
    # 1. ดึงรายการไฟล์ใน Drive
    remote_files = list_remote_files(service, remote_folder_id)
    if not remote_files:
        print("No remote files found.")
        return

    # 2. ดึงรายการไฟล์ในเครื่อง
    local_files = list_local_files(local_folder_path)
    
    # 3. เปรียบเทียบและดาวน์โหลด
    download_count = 0
    for item in remote_files:
        file_name = item['name']
        file_id = item['id']
        
        if file_name not in local_files:
            # ถ้าไฟล์นี้ยังไม่มีในเครื่อง -> ดาวน์โหลด
            download_count += 1
            local_dest_path = os.path.join(local_folder_path, file_name)
            download_file(service, file_id, file_name, local_dest_path)
        else:
            # ถ้ามีอยู่แล้ว -> ข้าม
            print(f"Skipping '{file_name}' (already exists).")
            
    print(f"Sync complete. Downloaded {download_count} new file(s).")

# === (สคริปต์หลัก) ===
def main():
    print("Connecting to Google Drive...")
    service = google_auth.get_drive_service()
    if not service:
        print("Failed to connect to Google Drive. Exiting.")
        return

    print("Finding remote folder IDs...")
    # 1. หา ID โฟลเดอร์หลัก
    base_folder_id = find_folder_id(service, REMOTE_BASE_FOLDER)
    if not base_folder_id:
        return

    # 2. หา ID โฟลเดอร์ย่อย
    video_folder_id = find_folder_id(service, REMOTE_VIDEO_FOLDER, base_folder_id)
    data_folder_id = find_folder_id(service, REMOTE_DATA_FOLDER, base_folder_id)

    # 3. เริ่ม Sync
    if video_folder_id:
        sync_folder(service, video_folder_id, LOCAL_VDO_PATH)
    else:
        print(f"Could not sync '{REMOTE_VIDEO_FOLDER}' (ID not found).")
        
    if data_folder_id:
        sync_folder(service, data_folder_id, LOCAL_RAW_DATA_PATH)
    else:
        print(f"Could not sync '{REMOTE_DATA_FOLDER}' (ID not found).")

if __name__ == '__main__':
    main()