@echo off
setlocal
cd /d "%~dp0"

python -m pip install -r requirements.txt
if errorlevel 1 goto fail

python -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onefile ^
    --windowed ^
    --optimize 2 ^
    --noupx ^
    --name RACZoneGen ^
    --exclude-module PyQt5 ^
    --exclude-module PyQt6 ^
    --exclude-module PySide2 ^
    --exclude-module PySide6 ^
    --exclude-module scipy ^
    --exclude-module sklearn ^
    --exclude-module IPython ^
    --exclude-module notebook ^
    --collect-all matplotlib ^
    --collect-all pyodbc ^
    --collect-all sqlalchemy ^
    --hidden-import pyodbc ^
    --hidden-import sqlalchemy ^
    app.py
if errorlevel 1 goto fail

mkdir "dist\sql" 2>nul
copy /Y "sql\rac_events.sql" "dist\sql\rac_events.sql"
copy /Y "sql\lanes.sql" "dist\sql\lanes.sql"
copy /Y "sql\zones.sql" "dist\sql\zones.sql"

if exist "dist\HVC_sql" (
    if exist "sql_HVC\rac_events.sql" copy /Y "sql_HVC\rac_events.sql" "dist\HVC_sql\rac_events.sql"
    if exist "sql_HVC\lanes.sql" copy /Y "sql_HVC\lanes.sql" "dist\HVC_sql\lanes.sql"
    if exist "sql_HVC\zones.sql" copy /Y "sql_HVC\zones.sql" "dist\HVC_sql\zones.sql"
)
mkdir "dist\sql_HVC" 2>nul
copy /Y "sql_HVC\rac_events.sql" "dist\sql_HVC\rac_events.sql"
copy /Y "sql_HVC\lanes.sql" "dist\sql_HVC\lanes.sql"
copy /Y "sql_HVC\zones.sql" "dist\sql_HVC\zones.sql"
copy /Y "zones.xml" "dist\zones.xml"
if not exist "dist\config.ini" copy /Y "config.ini" "dist\config.ini"

echo Build complete: dist\RACZoneGen.exe
pause
exit /b 0

:fail
echo BUILD FAILED
pause
exit /b 1