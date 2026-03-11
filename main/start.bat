@echo off
chcp 65001 >nul 2>&1
title Deal Agent - Auto Setup & Start
setlocal enabledelayedexpansion

echo ========================================
echo   Deal Agent - Auto Setup and Start
echo ========================================
echo.

cd /d "%~dp0"

REM ========================================
REM Step 1: Check Python Installation
REM ========================================
echo [1/6] Checking Python environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo.
    echo Please install Python 3.11 or later from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python !PYTHON_VERSION! found
echo.

REM ========================================
REM Step 2: Check/Create Virtual Environment
REM ========================================
echo [2/6] Setting up virtual environment...
if exist "venv\Scripts\python.exe" (
    echo [OK] Virtual environment found
    set PYTHON_CMD=venv\Scripts\python.exe
    set PIP_CMD=venv\Scripts\pip.exe
    set USE_VENV=1
) else (
    echo [INFO] No virtual environment found, creating one...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        echo.
        echo Please ensure you have Python 3.11+ installed.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
    set PYTHON_CMD=venv\Scripts\python.exe
    set PIP_CMD=venv\Scripts\pip.exe
    set USE_VENV=1
)
echo.

REM ========================================
REM Step 3: Upgrade pip
REM ========================================
echo [3/6] Upgrading pip...
%PYTHON_CMD% -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [WARNING] Failed to upgrade pip, continuing anyway...
) else (
    echo [OK] pip upgraded
)
echo.

REM ========================================
REM Step 4: Install Dependencies
REM ========================================
echo [4/6] Installing dependencies...
echo This may take a few minutes on first run...
echo.

if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found!
    echo Please ensure you're running this script from the project directory.
    pause
    exit /b 1
)

%PIP_CMD% install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies!
    echo.
    echo Trying with verbose output...
    %PIP_CMD% install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [ERROR] Installation failed. Please check the error messages above.
        pause
        exit /b 1
    )
) else (
    echo [OK] Dependencies installed
)
echo.

REM ========================================
REM Step 5: Check Port Availability
REM ========================================
echo [5/6] Checking if port 7860 is available...
netstat -ano | findstr :7860 >nul 2>&1
if not errorlevel 1 (
    echo [WARNING] Port 7860 is already in use!
    echo.
    echo Options:
    echo   1. Stop existing service using "stop.bat"
    echo   2. Close the application using port 7860
    echo   3. Press any key to continue anyway (may cause conflicts)
    pause
    echo.
    echo Continuing...
)
echo [OK] Port check complete
echo.

REM ========================================
REM Step 6: Start Application
REM ========================================
echo [6/6] Starting application...
echo.
echo ========================================
echo   Starting Deal Agent...
echo   This may take 20-60 seconds (first time may be longer)
echo   Loading ML models and starting Gradio server...
echo ========================================
echo.

REM Start the application in a new window
start "Deal Agent" %PYTHON_CMD% app_with_logs.py

REM Give the process a moment to start
timeout /t 3 /nobreak >nul

REM Check if Python process is running
tasklist /FI "IMAGENAME eq python.exe" 2>nul | find /I "python.exe" >nul
if errorlevel 1 (
    echo [ERROR] Python process did not start!
    echo.
    echo Troubleshooting:
    echo   1. Check logs.txt for error messages
    echo   2. Try running manually: %PYTHON_CMD% app_with_logs.py
    echo   3. Check if all dependencies are installed correctly
    pause
    exit /b 1
)
echo [OK] Application process started
echo.

REM ========================================
REM Wait for Service to be Ready
REM ========================================
echo Waiting for service to initialize...
echo Please wait, this may take 20-60 seconds...
echo.

set service_ready=0
set max_wait=90
for /L %%i in (1,1,%max_wait%) do (
    timeout /t 1 /nobreak >nul
    
    REM Check if port is listening
    netstat -ano | findstr ":7860" | findstr "LISTENING" >nul 2>&1
    if not errorlevel 1 (
        REM Wait a bit more to ensure Gradio is fully ready
        timeout /t 3 /nobreak >nul
        REM Double-check port is still listening
        netstat -ano | findstr ":7860" | findstr "LISTENING" >nul 2>&1
        if not errorlevel 1 (
            echo.
            echo [SUCCESS] Service is ready on port 7860 (after %%i seconds)
            set service_ready=1
            goto :service_ready
        )
    )
    
    REM Check if Python process is still running
    tasklist /FI "IMAGENAME eq python.exe" 2>nul | find /I "python.exe" >nul
    if errorlevel 1 (
        echo.
        echo [ERROR] Application process stopped unexpectedly!
        echo.
        echo Possible causes:
        echo   1. Missing dependencies - Check logs.txt
        echo   2. API key not configured - Create .env file with API keys
        echo   3. Port conflict - Another application may be using port 7860
        echo.
        echo Check logs.txt for detailed error messages
        echo.
        pause
        exit /b 1
    )
    
    if %%i LEQ 30 (
        echo Waiting for service... %%i/%max_wait% seconds
    ) else if %%i LEQ 60 (
        echo Still loading... %%i/%max_wait% seconds (loading ML models...)
    ) else (
        echo Almost ready... %%i/%max_wait% seconds
    )
)

:service_ready
if "!service_ready!"=="0" (
    echo.
    echo [WARNING] Service did not start after %max_wait% seconds
    echo.
    echo Checking application status...
    tasklist /FI "IMAGENAME eq python.exe" 2>nul | find /I "python.exe" >nul
    if errorlevel 1 (
        echo [ERROR] Application process not found. Service may have crashed.
        echo.
        echo Troubleshooting steps:
        echo   1. Check logs.txt for errors
        echo   2. Check if .env file exists with API keys
        echo   3. Try manual start: %PYTHON_CMD% app_with_logs.py
        echo.
        pause
        exit /b 1
    ) else (
        echo [INFO] Application process is running
        echo [INFO] Service may still be initializing (loading ML models)
        echo [INFO] Opening browser - if page doesn't load, wait 10-20 seconds and refresh
    )
) else (
    echo.
    echo [SUCCESS] Service is ready! Opening browser...
    timeout /t 2 /nobreak >nul
)

REM ========================================
REM Open Browser
REM ========================================
echo.
echo Opening browser at http://127.0.0.1:7860/...
start http://127.0.0.1:7860/
timeout /t 1 /nobreak >nul

REM Final port check
netstat -ano | findstr ":7860" | findstr "LISTENING" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Port check failed, but browser opened anyway
    echo Please wait 10-20 seconds and refresh the page
) else (
    echo [OK] Port confirmed listening
)

echo.
echo ========================================
echo   Setup and Startup Complete!
echo ========================================
echo.
echo   Access URL: http://127.0.0.1:7860/
echo.
echo   Virtual Environment: venv\
echo   Python Command: %PYTHON_CMD%
echo.
echo   If the page doesn't load:
echo   1. Wait a few more seconds and refresh
echo   2. Check if Python process is running
echo   3. Check logs.txt for errors
echo   4. Ensure .env file exists with API keys
echo.
echo   To stop the service, use "stop.bat"
echo ========================================
echo.
echo   Closing this window will NOT stop the application.
echo   The application is running in a separate window.
echo.
pause

endlocal
