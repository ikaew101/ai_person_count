import os
import cv2
import numpy as np
import json
import argparse
from datetime import datetime

# --- การตั้งค่าที่สำคัญ ---
CONFIG_FILE = 'config/camera_config.json'
CONFIG_HELPER_FILE = "config/config_points.txt" # ไฟล์ที่เราจะบันทึกพิกัด

# ====================== MAIN LOGIC =========================
def main():
    parser = argparse.ArgumentParser(description="Boundary Drawing Tool")
    parser.add_argument("camera_name", help="Name of the camera config to load.")
    args = parser.parse_args()
    
    # --- Load Config ---
    try:
        with open(CONFIG_FILE,"r",encoding='utf-8') as f: full_config=json.load(f)
    except: raise SystemExit(f"Config '{CONFIG_FILE}' not found.")
    if args.camera_name not in full_config: raise SystemExit(f"Camera '{args.camera_name}' not found.")
    config = full_config[args.camera_name]

    # --- โหลดค่าตั้งต้นจาก Config (เพื่อแสดงผล) ---
    video_path=config.get('video_path'); display_width=config.get('display_width',1280)
    red_line=tuple(map(tuple,config['lines']['red']))
    blue_line=tuple(map(tuple,config['lines']['blue']))
    green_line=tuple(map(tuple,config['lines']['green']))
    yellow_line=tuple(map(tuple,config['lines']['yellow']))
    pink_zone = config['pink_zone']

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise IOError(f"Cannot open video: {video_path}")
    original_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); original_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if original_w==0 or original_h==0: raise IOError("Could not read video dimensions.")
    aspect=original_w/max(1,original_h); display_height=int(display_width/aspect)

    cv2.namedWindow("Boundary Drawer", cv2.WINDOW_NORMAL)
    paused=True # <-- 1. เริ่มแบบ Paused เลย
    mouse_pos_raw=(-1,-1)
    last_frame = None

    # --- ตัวแปรสถานะสำหรับวาดเส้น ---
    current_drawing_key = None 
    temp_point_1 = None
    temp_points_list = [] 

    # --- ฟังก์ชัน Helper สำหรับบันทึกลง .txt ---
    def save_to_helper_file(shape_key, data_list):
        try:
            with open(CONFIG_HELPER_FILE, "a", encoding='utf-8') as f:
                f.write(f"\n# --- {shape_key.upper()} coordinates (at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n")
                if shape_key == 'pink_zone':
                    f.write(f"\"pink_zone\": [\n")
                    for point in data_list:
                        f.write(f"    {point},\n")
                    f.write("],\n")
                else: 
                    f.write(f"\"{shape_key}\": {data_list},\n")
                print(f"Coordinates for '{shape_key}' saved to {CONFIG_HELPER_FILE}")
        except Exception as e:
            print(f"Error writing to {CONFIG_HELPER_FILE}: {e}")

    # --- Mouse Callback (เกือบเหมือนเดิม) ---
    def _on_mouse(event, x, y, flags, param):
        nonlocal mouse_pos_raw, temp_point_1, current_drawing_key, temp_points_list
        nonlocal red_line, blue_line, green_line, yellow_line, pink_zone
        
        rx = int(x * original_w / display_width)
        ry = int(y * original_h / display_height)
        mouse_pos_raw = (rx, ry)

        if event == cv2.EVENT_LBUTTONDOWN:
            if current_drawing_key is None:
                pass # โหมดปกติ ไม่ต้องทำอะไร
            
            elif current_drawing_key in ['red', 'blue', 'green', 'yellow']:
                if temp_point_1 is None:
                    temp_point_1 = (rx, ry)
                    print(f"Set {current_drawing_key} line - Point 1: {temp_point_1}")
                else:
                    temp_point_2 = (rx, ry)
                    line_tuple = (temp_point_1, temp_point_2)
                    line_list = [list(temp_point_1), list(temp_point_2)]

                    if current_drawing_key == 'red': red_line = line_tuple
                    elif current_drawing_key == 'blue': blue_line = line_tuple
                    elif current_drawing_key == 'green': green_line = line_tuple
                    elif current_drawing_key == 'yellow': yellow_line = line_tuple
                    
                    save_to_helper_file(current_drawing_key, line_list)
                    temp_point_1 = None
                    current_drawing_key = None
                    
            elif current_drawing_key == 'pink_zone':
                new_point = [rx, ry]
                temp_points_list.append(new_point)
                print(f"Set pink_zone - Point {len(temp_points_list)}: {new_point}")
    
    cv2.setMouseCallback("Boundary Drawer", _on_mouse)

    try:
        while True:
            if not paused or last_frame is None: # <-- 2. อ่านเฟรมเฉพาะเมื่อ Play หรือเป็นเฟรมแรก
                ret, frame = cap.read()
                if not ret: 
                    print("End of video. Resetting to first frame.")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                last_frame = frame.copy()
            
            frame = last_frame.copy() # ทำงานบนเฟรมที่หยุดนิ่ง

            # --- UI Display (วาดเส้น/โซน ที่อัปเดตแล้วจากใน memory) ---
            cv2.polylines(frame, [np.array(pink_zone, dtype=np.int32)], isClosed=True, color=(255, 182, 193), thickness=2)
            cv2.line(frame, red_line[0], red_line[1], (0,0,255), 2)
            cv2.line(frame, blue_line[0], blue_line[1], (255,0,0), 2)
            cv2.line(frame, green_line[0], green_line[1], (0,255,0), 2)
            cv2.line(frame, yellow_line[0], yellow_line[1], (0,255,255), 2)
            
            # --- UI สำหรับวาดเส้น (เหมือนเดิม) ---
            if mouse_pos_raw[0]>=0:
                cv2.drawMarker(frame,(mouse_pos_raw[0],mouse_pos_raw[1]),(0,255,255), cv2.MARKER_CROSS,20,2)
                text=f"x:{mouse_pos_raw[0]} y:{mouse_pos_raw[1]}"; cv2.putText(frame,text,(mouse_pos_raw[0]+15,mouse_pos_raw[1]-15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

            prompt_text = ""
            if current_drawing_key in ['red', 'blue', 'green', 'yellow']:
                if temp_point_1 is None:
                    prompt_text = f"Click to set Point 1 for {current_drawing_key} line"
                else:
                    prompt_text = f"Click to set Point 2 for {current_drawing_key} line"
                    cv2.line(frame, temp_point_1, mouse_pos_raw, (0, 255, 255), 1) 
            
            elif current_drawing_key == 'pink_zone':
                prompt_text = "Click to add point. ENTER=Apply, ESC=Cancel"
                if len(temp_points_list) > 0:
                    cv2.line(frame, tuple(temp_points_list[-1]), mouse_pos_raw, (255, 182, 193), 1)
                if len(temp_points_list) > 1:
                    cv2.line(frame, tuple(temp_points_list[0]), mouse_pos_raw, (255, 182, 193), 1, cv2.LINE_AA)
                for pt in temp_points_list:
                    cv2.circle(frame, tuple(pt), 5, (255, 182, 193), -1)
            else:
                 prompt_text = "Paused: (P)lay. Draw: (R)ed (B)lue (G)reen (Y)ellow (K)Pink. (ESC)uit"
            
            cv2.putText(frame, prompt_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow('Boundary Drawer', cv2.resize(frame, (display_width, display_height)))
            
            # --- Key Handler (เหมือนเดิม) ---
            k = cv2.waitKey(20) & 0xFF # <-- 3. เพิ่ม delay เล็กน้อย
            
            if k == 27: # ESC 
                if temp_point_1 is not None or current_drawing_key is not None or len(temp_points_list) > 0:
                    temp_point_1 = None
                    current_drawing_key = None
                    temp_points_list = [] 
                    print("Drawing mode cancelled.")
                else:
                    break 
            
            elif k == 13 and current_drawing_key == 'pink_zone': # ENTER
                if len(temp_points_list) >= 3:
                    pink_zone = temp_points_list
                    save_to_helper_file('pink_zone', temp_points_list)
                else:
                    print("Zone requires at least 3 points. Cancelling.")
                temp_point_1 = None
                current_drawing_key = None
                temp_points_list = []
                
            elif k == ord('p'): 
                paused = not paused
                print(f"Paused: {paused}")
            
            elif k == ord('r'):
                current_drawing_key = 'red'; temp_point_1 = None; print("DRAWING MODE: RED")
            elif k == ord('b'):
                current_drawing_key = 'blue'; temp_point_1 = None; print("DRAWING MODE: BLUE")
            elif k == ord('g'):
                current_drawing_key = 'green'; temp_point_1 = None; print("DRAWING MODE: GREEN")
            elif k == ord('y'):
                current_drawing_key = 'yellow'; temp_point_1 = None; print("DRAWING MODE: YELLOW")
            elif k == ord('k'): 
                current_drawing_key = 'pink_zone'; temp_points_list = []; print("DRAWING MODE: PINK ZONE")
            
    except KeyboardInterrupt:
        print("\nUser interrupted process.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Boundary Drawer closed.")

if __name__ == "__main__":
    main()