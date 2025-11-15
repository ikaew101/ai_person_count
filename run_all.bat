@echo off
echo Starting batch processing...

REM --- Download Raw Data (File Video, Camera raw excel) ---
echo.
echo ===========================================
echo Processing Download Raw Data (File Video, Camera raw excel)
echo ===========================================
py download_rawdata.py

REM --- Run Processing AI QA Camera ---
echo.
echo ===========================================
echo Run Processing AI QA Camera
echo ===========================================
py run_processor.py

echo.
echo ===========================================
echo All processing complete.
echo ===========================================
pause