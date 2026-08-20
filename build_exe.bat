@echo off
setlocal
cd /d "%~dp0"

python -m pip install -r requirements.txt

python -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onefile ^
    --console ^
    --name RACZoneMonitor ^
    --collect-all matplotlib ^
    app.py

if errorlevel 1 (
    echo BUILD FAILED
    pause
    exit /b 1
)

mkdir "dist\sql" 2>nul
copy /Y "config.ini" "dist\config.ini"
copy /Y "zones.xml" "dist\zones.xml"
copy /Y "sql\rac_events.sql" "dist\sql\rac_events.sql"
copy /Y "sql\lanes.sql" "dist\sql\lanes.sql"

echo Build complete: dist\RACZoneMonitor.exe
pause
