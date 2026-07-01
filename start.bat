@echo off
cd /d "%~dp0"
echo Starting AudioAE server...
python -u app.py
pause
