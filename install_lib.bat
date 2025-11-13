@echo off
REM This batch file will install all necessary Python libraries for the AI Person Count project.

echo =======================================================
echo Upgrading pip...
echo =======================================================
REM Use 'python -m pip' for consistency
python -m pip install --upgrade pip

echo =======================================================
echo Installing Python Dependencies...
echo =======================================================
REM Use '%~dp0' to make the path relative to the batch file itself
python -m pip install -r "%~dp0config\requirements.txt"

echo =======================================================
echo Installing Google Drive...
echo =======================================================
python -m pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

echo =======================================================
echo All Python libraries are installed!
echo =======================================================
echo.
echo !!! IMPORTANT NOTE !!!
echo This script CANNOT install Tesseract OCR (the application)
echo.
echo After installing, you may need to update the TESSERACT_PATH in 'ai_personCount.py'.
echo.

REM Pause to let the user read the output
pause