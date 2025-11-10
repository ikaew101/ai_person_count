# import pandas as pd
# import os
# import glob
# import datetime

# def process_data_validation():
    
#     # ======================================================================
#     # !! ตัวแปรตั้งค่า: !!
#     # ======================================================================
    
#     # 1. กำหนดจำนวนนาทีที่จะตรวจสอบ (จากต้นชั่วโมง)
#     MINUTES_TO_CHECK = 10
    
#     # 2. กำหนดชื่อคอลัมน์ "เวลา" ในไฟล์ TDG (AI Model for data validation.csv)
#     TDG_TIMESTAMP_COLUMN = 'Timestamp' 
    
#     # 3. !! ใหม่: กำหนด % Accuracy ขั้นต่ำสำหรับ 'Y'
#     ACCURACY_CHECK_THRESHOLD = 50.0 
    
#     # ======================================================================
    
#     print(f"Starting Data Validation Process (v6 - Accuracy Logic)...")
#     print(f"Config: Checking first {MINUTES_TO_CHECK} minutes of the hour.")
#     print(f"Config: Accuracy Threshold set to {ACCURACY_CHECK_THRESHOLD}%.")
    
#     # --- 1. กำหนด Path ---
#     folder1_path = r"C:\Users\sukit\AI Model\SS RAW\TDG QA"
#     folder2_path = r"C:\Users\sukit\AI Model\SS RAW"
#     output_folder_path = r"C:\Users\sukit\AI Model\SS RAW\TDG QA"
    
#     tdg_file_name = "AI Model for data validation.csv"
#     tdg_file_path = os.path.join(folder1_path, tdg_file_name)
    
#     # --- 2. สร้างชื่อไฟล์ Output ---
#     today_str = datetime.datetime.now().strftime("%Y%m%d")
#     output_filename = f"QA Data validation_{today_str}.csv"
#     output_file_path = os.path.join(output_folder_path, output_filename)
    
#     print(f"TDG (Reference) file: {tdg_file_path}")
#     print(f"SS (Source) folder: {folder2_path}")

#     # --- 3. อ่านและเตรียมไฟล์หลัก (TDG) ---
#     try:
#         print(f"Reading TDG file...")
#         df_tdg_main = pd.read_csv(tdg_file_path)
        
#         if 'Cam_name' not in df_tdg_main.columns or TDG_TIMESTAMP_COLUMN not in df_tdg_main.columns:
#             print(f"Error: TDG file must contain 'Cam_name' and '{TDG_TIMESTAMP_COLUMN}' columns.")
#             return
            
#         print(f"TDG file read successfully. Found {len(df_tdg_main)} total rows.")
        
#         df_tdg_main['Cam_name_stripped'] = df_tdg_main['Cam_name'].astype(str).str.strip()
#         df_tdg_main[TDG_TIMESTAMP_COLUMN] = pd.to_datetime(df_tdg_main[TDG_TIMESTAMP_COLUMN], errors='coerce')
#         df_tdg_main = df_tdg_main.dropna(subset=[TDG_TIMESTAMP_COLUMN])
#         print(f"TDG file pre-processed.")
        
#     except FileNotFoundError:
#         print(f"Error: TDG file not found at {tdg_file_path}.")
#         return
#     except Exception as e:
#         print(f"Error reading TDG file {tdg_file_path}: {e}")
#         return

#     # --- 4. เตรียม List สำหรับเก็บผลลัพธ์ ---
#     results = []
    
#     # --- 5. ค้นหาไฟล์ Excel ทั้งหมดใน Folder 2 ---
#     search_pattern = os.path.join(folder2_path, "*.xlsx")
#     excel_files_in_folder2 = glob.glob(search_pattern)
    
#     excel_files_to_process = [
#         f for f in excel_files_in_folder2 
#         if not f.endswith(tdg_file_name)
#     ]

#     if not excel_files_to_process:
#         print(f"No Excel (.xlsx) files found in {folder2_path} to process.")
#         return
        
#     print(f"Found {len(excel_files_to_process)} Excel files. Starting processing...")

#     # --- 6. วนลูปประมวลผลทีละไฟล์ ---
#     for i, xlsx_file_path in enumerate(excel_files_to_process):
        
#         cam_name = ""
#         ss_count = 0
#         tdg_count = 0 
#         check_val = 'N'
        
#         try:
#             base_name = os.path.basename(xlsx_file_path)
#             cam_name, _ = os.path.splitext(base_name)
#             cam_name_stripped = cam_name.strip()

#             print(f"\n--- Processing file {i+1}/{len(excel_files_to_process)}: {cam_name} ---")
            
#             # 
#             # !! ========================================================== !!
#             # !! ส่วนที่ 1: คำนวณ SS Count (จากไฟล์ Excel)
#             # !! ========================================================== !!
#             print(f"--- DEBUG (SS count) ---")
#             df_cam = pd.read_excel(xlsx_file_path)
            
#             if 'action' not in df_cam.columns or 'start_time' not in df_cam.columns:
#                 print(f"Warning: File {base_name} is missing 'action' or 'start_time' column. SS count = 0.")
#                 ss_count = 0
#             else:
#                 df_filtered = df_cam[df_cam['action'] == 1].copy()
#                 print(f"Found {len(df_filtered)} rows with 'action == 1'.")
                
#                 if df_filtered.empty:
#                     print(f"No 'action == 1'. SS count = 0.")
#                     ss_count = 0
#                 else:
#                     df_filtered['start_time'] = pd.to_datetime(df_filtered['start_time'], errors='coerce')
#                     df_filtered = df_filtered.dropna(subset=['start_time'])
                    
#                     if df_filtered.empty:
#                         print(f"No valid 'start_time'. SS count = 0.")
#                         ss_count = 0
#                     else:
#                         df_sorted = df_filtered.sort_values(by='start_time')
#                         earliest_ss_time = df_sorted['start_time'].iloc[0]
#                         ss_min_time = earliest_ss_time.replace(minute=0, second=0, microsecond=0)
#                         ss_time_limit = ss_min_time + pd.Timedelta(minutes=MINUTES_TO_CHECK)
                        
#                         print(f"(SS) Earliest timestamp: {earliest_ss_time}")
#                         print(f"(SS) Calculated window: {ss_min_time} TO {ss_time_limit}")

#                         df_ss_in_window = df_sorted[
#                             (df_sorted['start_time'] >= ss_min_time) & 
#                             (df_sorted['start_time'] < ss_time_limit)
#                         ]
#                         ss_count = len(df_ss_in_window)
#                         print(f"Result SS count (in window): {ss_count}")

#             # 
#             # !! ========================================================== !!
#             # !! ส่วนที่ 2: คำนวณ TDG Count (จากไฟล์ CSV)
#             # !! ========================================================== !!
#             print(f"--- DEBUG (TDG count) ---")
#             print(f"Matching Cam_name '{cam_name_stripped}' in TDG file...")
            
#             df_tdg_cam_specific = df_tdg_main[df_tdg_main['Cam_name_stripped'] == cam_name_stripped].copy()
            
#             if df_tdg_cam_specific.empty:
#                 print(f"No matching Cam_name found in TDG file. TDG count = 0.")
#                 tdg_count = 0
#             else:
#                 print(f"Found {len(df_tdg_cam_specific)} total rows for this Cam in TDG.")
#                 df_tdg_cam_sorted = df_tdg_cam_specific.sort_values(by=TDG_TIMESTAMP_COLUMN)
                
#                 earliest_tdg_time = df_tdg_cam_sorted[TDG_TIMESTAMP_COLUMN].iloc[0]
#                 tdg_min_time = earliest_tdg_time.replace(minute=0, second=0, microsecond=0)
#                 tdg_time_limit = tdg_min_time + pd.Timedelta(minutes=MINUTES_TO_CHECK)
                
#                 print(f"(TDG) Earliest timestamp: {earliest_tdg_time}")
#                 print(f"(TDG) Calculated window: {tdg_min_time} TO {tdg_time_limit}")

#                 df_tdg_in_window = df_tdg_cam_sorted[
#                     (df_tdg_cam_sorted[TDG_TIMESTAMP_COLUMN] >= tdg_min_time) &
#                     (df_tdg_cam_sorted[TDG_TIMESTAMP_COLUMN] < tdg_time_limit)
#                 ]
#                 tdg_count = len(df_tdg_in_window)
#                 print(f"Result TDG count (in window): {tdg_count}")
            

#             # 
#             # !! ========================================================== !!
#             # !! Step 3: เทียบ 2 column (New Accuracy Logic)
#             # !! ========================================================== !!
#             print(f"--- RESULT ---")
#             difference = abs(ss_count - tdg_count)
#             accuracy = 0.0 # ตั้งต้น
            
#             if tdg_count == 0:
#                 if ss_count == 0:
#                     accuracy = 100.0 # ตรงกัน (0 ทั้งคู่)
#                 else:
#                     accuracy = 0.0 # ผิด (TDG คาดหวัง 0 แต่ SS ได้ {ss_count})
#             else:
#                 # คำนวณ accuracy ตามสูตร
#                 accuracy = (1 - (difference / tdg_count)) * 100
            
#             # ถ้า SS มากกว่า TDG เยอะๆ accuracy อาจติดลบได้ ให้ปัดเป็น 0
#             if accuracy < 0:
#                 accuracy = 0.0

#             # เปรียบเทียบกับ Threshold
#             check_val = 'Y' if accuracy >= ACCURACY_CHECK_THRESHOLD else 'N'
            
#             print(f"SS count: {ss_count}, TDG count: {tdg_count}, Diff: {difference}")
#             # ใช้ :.2f เพื่อแสดงทศนิยม 2 ตำแหน่ง
#             print(f"Accuracy: {accuracy:.2f}% (Threshold: {ACCURACY_CHECK_THRESHOLD}%) -> Check: {check_val}")
#             # !! ========================================================== !!


#             results.append({
#                 'Cam_name': cam_name, 
#                 'SS count': ss_count, 
#                 'TDG count': tdg_count, 
#                 'Passed': check_val
#             })

#         except Exception as e:
#             print(f"!! Error processing file {xlsx_file_path}: {e}")
#             results.append({
#                 'Cam_name': cam_name if cam_name else os.path.basename(xlsx_file_path), 
#                 'SS count': 'Error', 
#                 'TDG count': 'Error', 
#                 'Passed': 'Error'
#             })

#     # --- 7. สร้าง DataFrame ผลลัพธ์ และ บันทึกเป็น CSV ---
#     if not results:
#         print("\nNo results to save.")
#     else:
#         try:
#             print("\nCreating output DataFrame...")
#             df_output = pd.DataFrame(results)
#             df_output = df_output[['Cam_name', 'SS count', 'TDG count', 'Passed']]
            
#             df_output.to_csv(output_file_path, index=False, encoding='utf-8-sig')
            
#             print(f"\n=======================================================")
#             print(f"Successfully created output file at:")
#             print(output_file_path)
#             print("=======================================================")
#             print("\nFinal Data Head (Top 5 rows):")
#             print(df_output.head())
            
#         except Exception as e:
#             print(f"Error saving output file: {e}")

# # --- เรียกใช้งาน function ---
# if __name__ == "__main__":
#     process_data_validation()