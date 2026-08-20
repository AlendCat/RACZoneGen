@echo off
setlocal
cd /d "%~dp0"

python -m pip install -r requirements.txt
if errorlevel 1 goto fail

python -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onefile ^
    --console ^
    --name RACZoneGen ^
    --collect-all matplotlib ^
    app.py
if errorlevel 1 goto fail

mkdir "dist\sql" 2>nul
copy /Y "config.ini" "dist\config.ini"
copy /Y "zones.xml" "dist\zones.xml"
copy /Y "sql\rac_events.sql" "dist\sql\rac_events.sql"
copy /Y "sql\lanes.sql" "dist\sql\lanes.sql"
copy /Y "sql\zones.sql" "dist\sql\zones.sql"

echo Build complete: dist\RACZoneGen.exe
pause
exit /b 0

:fail
echo BUILD FAILED
pause
exit /b 1