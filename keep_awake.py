import os
import ctypes
import time
import sys

# ตรวจสอบว่าเป็น Windows เท่านั้น
if os.name != 'nt':
    print("This script is designed for Windows only. Exiting.")
    sys.exit(0) # ออกจากโปรแกรมโดยไม่ Error (เพื่อให้ Batch file ทำงานต่อได้)

# (จาก Windows API)
# ES_CONTINUOUS: แจ้งว่า Process นี้ทำงานต่อเนื่อง
# ES_SYSTEM_REQUIRED: "บังคับ" ไม่ให้ระบบเข้าสู่โหมด Sleep (Idle)
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

def set_keep_awake():
    """สั่ง Windows ว่า "ห้ามหลับ" """
    try:
        print("[KeepAwake] Setting execution state to prevent system sleep/idle...")
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
        print("[KeepAwake] System sleep prevention ACTIVATED.")
    except Exception as e:
        print(f"[KeepAwake] Warning: Could not set execution state. {e}")

def reset_sleep_mode():
    """สั่ง Windows ว่า "กลับไปโหมดปกติได้" """
    try:
        print("[KeepAwake] Resetting execution state to normal...")
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        print("[KeepAwake] System sleep prevention DEACTIVATED.")
    except Exception as e:
        print(f"[KeepAwake] Warning: Could not reset execution state. {e}")

if __name__ == "__main__":
    try:
        # 1. สั่ง "ห้ามหลับ"
        set_keep_awake()
        
        # 2. วนลูปไปเรื่อยๆ เพื่อให้สคริปต์นี้ทำงานค้างไว้
        while True:
            # (ไม่ต้องทำอะไรเลย แค่รอ)
            time.sleep(60) # ตรวจสอบทุก 60 วินาที

    except KeyboardInterrupt:
        # (เมื่อผู้ใช้กด Ctrl+C หรือปิดหน้าต่างนี้)
        print("[KeepAwake] Interrupted. Releasing sleep lock...")
        
    finally:
        # 3. (สำคัญมาก) คืนค่าให้ Windows กลับไปหลับได้
        # (เพื่อให้เครื่อง Shutdown ได้ตามปกติหลังจากงานเสร็จ)
        reset_sleep_mode()