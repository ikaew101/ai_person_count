import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# === การตั้งค่า ===
# นี่คือ "ขอบเขต" การอนุญาตที่เราขอ (แก้ไข/อ่าน/เขียน ไฟล์ใน Drive)
SCOPES = ['https://www.googleapis.com/auth/drive']
CREDENTIALS_FILE = './config/credentials.json' # ไฟล์ที่คุณดาวน์โหลดจาก Google
TOKEN_FILE = './config/token.json'             # ไฟล์นี้จะถูกสร้างขึ้นอัตโนมัติ

def get_drive_service():
    """
    Verify your identity with Google Drive and return the 'service' object.
    """
    creds = None
    # ตรวจสอบว่ามี token.json (ไฟล์เก็บการล็อกอิน) หรือยัง
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    
    # ถ้า token ไม่มี หรือหมดอายุ
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # นี่คือส่วนที่ต้องรันครั้งแรก
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0) # จะเปิดเบราว์เซอร์ให้คุณล็อกอิน
        
        # บันทึก token ไว้ใช้ครั้งต่อไป
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('drive', 'v3', credentials=creds)
        print("Google Drive service created successfully.")
        return service
    except HttpError as error:
        print(f'An error occurred: {error}')
        return None

if __name__ == '__main__':
    # รันไฟล์นี้โดยตรง 1 ครั้งเพื่อสร้าง token.json
    print("Attempting to authenticate with Google Drive...")
    print("Your browser will open for authentication.")
    get_drive_service()
    print("Authentication successful. 'token.json' created.")
    print("You can now run 'download_rawdata.py' and 'run_processor.py'.")