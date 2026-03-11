@echo off
chcp 65001 >nul 2>&1
title Deal Agent - Stop Service

echo ========================================
echo   Deal Agent - Stop Service
echo ========================================
echo.

echo [1/2] Finding and stopping processes on port 7860...
set found=0
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :7860 ^| findstr LISTENING') do (
    echo Found process on port 7860: PID %%a
    taskkill /F /PID %%a >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] Failed to kill process %%a (may require admin rights)
    ) else (
        echo [OK] Process %%a stopped
        set found=1
    )
)

if %found%==0 (
    echo [INFO] No process found listening on port 7860
)

echo.
echo [2/2] Stopping Python processes running app_with_logs.py...
set python_found=0
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST 2^>nul ^| find /I "PID:"') do (
    set pid=%%a
    for /f "tokens=*" %%b in ('wmic process where "ProcessId=!pid!" get CommandLine /format:list 2^>nul ^| find /I "app_with_logs.py"') do (
        echo Found Python process: PID !pid!
        taskkill /F /PID !pid! >nul 2>&1
        if errorlevel 1 (
            echo [WARNING] Failed to kill Python process !pid!
        ) else (
            echo [OK] Python process !pid! stopped
            set python_found=1
        )
    )
)

if %python_found%==0 (
    echo [INFO] No Python processes running app_with_logs.py found
)

echo.
echo ========================================
echo   Stop Operation Complete
echo ========================================
echo.
echo All Deal Agent processes have been stopped.
echo You can now safely run "start.bat" again.
echo.
timeout /t 3 >nul
