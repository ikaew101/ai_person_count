@echo off
echo Starting batch processing...

REM --- 5129_Banglen-Yao ---
echo.
echo ===========================================
echo Processing Camera: 5129_Banglen-Yao (Full Video)
echo ===========================================
python ai_personCount.py 5129_Banglen-Yao

REM --- 5033_entrance2 ---
echo.
echo ===========================================
echo Processing Camera: 5033_entrance2 (Start: 0 min, Duration: 30 min)
echo ===========================================
python ai_personCount.py 5033_entrance2 --start_min 0 --duration_min 30

REM --- Rawai2-CAM24 ---
echo.
echo ===========================================
echo Processing Camera: Rawai2-CAM24 (Start: 30 min, Duration: 30 min)
echo ===========================================
python ai_personCount.py 166_rawai2-cam24 --start_min 30 --duration_min 30

REM --- lampun-19_old ---
echo.
echo ===========================================
echo Processing Camera: lampun-19_old (Full Video)
echo ===========================================
python ai_personCount.py lampun-19_old

REM --- lampun-19_new ---
echo.
echo ===========================================
echo Processing Camera: lampun-19_new (Full Video)
echo ===========================================
python ai_personCount.py lampun-19_new

echo.
echo ===========================================
echo All processing complete.
echo ===========================================
pause