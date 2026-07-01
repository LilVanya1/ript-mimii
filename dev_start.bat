@echo off
cd /d "%~dp0"
echo [AudioAE] Starting dev server with auto-reload...

python -m pip show watchfiles >nul 2>&1
if errorlevel 1 (
  echo [AudioAE] Installing watchfiles...
  python -m pip install watchfiles
)

echo [AudioAE] Auto-reload is ON. Save files to restart automatically.
python -m watchfiles --filter python --ignore-paths ".git,__pycache__,data,models,results,.ipynb_checkpoints" "python app.py" app.py src templates

pause
