@echo off
chcp 65001 >nul
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

echo Starting server at http://127.0.0.1:5002
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:5002"

python "AI-DDL Writing(2).py"
pause
