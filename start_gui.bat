@echo off
cd /d "%~dp0"
echo ========================================
echo   MIMII Baseline GUI - PyTorch
echo ========================================
echo.

:: Start Flask server in a separate window
start "MIMII GUI" cmd /c python gui.py

:: Wait for server to start
echo Waiting for server...
ping -n 3 127.0.0.1 >nul

:: Open browser
start http://127.0.0.1:8080

echo.
echo GUI is running at http://127.0.0.1:8080
echo Close this window to keep server running in background,
echo or close the "MIMII GUI" window to stop the server.
echo.
pause
