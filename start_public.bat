@echo off
chcp 65001 >nul 2>&1
echo Запускаю сервер с публичным доступом через ngrok...
python app.py --public
pause
