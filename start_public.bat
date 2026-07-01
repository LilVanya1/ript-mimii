@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo [1/2] Git auto-deploy watcher (friend push = auto pull + restart)
start "" /B python -u scripts\git_watch_deploy.py

echo [2/2] Starting server...
echo Friend only needs: git push
echo You do NOT need to restart manually.
python -u app.py --public
pause
