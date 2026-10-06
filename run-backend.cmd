@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python virtual environment...
    python -m venv .venv
    if errorlevel 1 exit /b 1
    echo Installing backend dependencies...
    ".venv\Scripts\python.exe" -m pip install -r backend\requirements.txt
    if errorlevel 1 exit /b 1
)

if not exist "backend\.env" (
    copy /y "backend\.env.example" "backend\.env" >nul
)

echo Starting backend at http://localhost:8000
".venv\Scripts\python.exe" -m uvicorn app.main:app --app-dir backend --reload --port 8000
