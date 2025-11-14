@echo off
echo Starting batch processing...

REM --- Download Raw Data (File Video, Camera raw excel) ---
echo.
echo ===========================================
echo Processing Download Raw Data (File Video, Camera raw excel)
echo ===========================================
python download_rawdata.py

REM --- Check Status File ---
echo.
echo ===========================================
echo Processing Check Status File
echo ===========================================
python generate_master_log.py

REM --- Run Processing AI QA Camera ---
echo.
echo ===========================================
echo Run Processing AI QA Camera
echo ===========================================
python run_processor.py

echo.
echo ===========================================
echo All processing complete.
echo ===========================================
pause