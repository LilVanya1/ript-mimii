@echo off
chcp 65001 >nul 2>&1
python -m pip install --upgrade pip
pip install -r requirements.txt
pause
