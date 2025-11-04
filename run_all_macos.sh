#!/bin/bash
echo "Starting batch processing..."

# --- ----------------------------------------------------
# ---           กำหนดคำสั่งรันสำหรับแต่ละกล้องตรงนี้          ---
# --- ----------------------------------------------------

# --- รันกล้องที่ 1: (Banglen-Yao) เริ่มนาทีที่ 10 รัน 20 นาที ---
echo ""
echo "==========================================="
echo "Processing Camera: 5129_Banglen-Yao (Start: 10 min, Duration: 20 min)"
echo "==========================================="
python3 ai_personCount.py 5129_Banglen-Yao --start_min 10 --duration_min 20

# --- รันกล้องที่ 2: (5033_entrance2) เริ่มนาทีที่ 5 รัน 15 นาที ---
echo ""
echo "==========================================="
echo "Processing Camera: 5033_entrance2 (Start: 5 min, Duration: 15 min)"
echo "==========================================="
python3 ai_personCount.py 5033_entrance2 --start_min 5 --duration_min 15

# --- รันกล้องที่ 3: (สมมติ) รันตั้งแต่ต้นจนจบ (Default) ---
# echo ""
# echo "==========================================="
# echo "Processing Camera: Another_Camera (Full Video)"
# echo "==========================================="
# python3 ai_personCount.py Another_Camera

echo ""
echo "==========================================="
echo "All processing complete."
echo "==========================================="