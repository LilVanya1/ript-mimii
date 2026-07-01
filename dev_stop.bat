@echo off
echo [AudioAE] Stopping server on port 228...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :228 ^| findstr LISTENING') do (
  taskkill /PID %%a /F >nul 2>&1
)
echo [AudioAE] Done.
pause
