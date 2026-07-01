@echo off
cd /d "%~dp0"
call dev_stop.bat
start "" cmd /k dev_start.bat
