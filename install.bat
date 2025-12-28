@echo off
title EXON Installer
echo ===================================================
echo      EXON - Automated Installer
echo ===================================================

echo Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python 3.10+.
    pause
    exit /b
)

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists.
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading PIP...
python -m pip install --upgrade pip

echo Installing dependencies...
pip install -r requirements.txt

echo ===================================================
echo      Installation Complete!
echo      You can now run 'start_exon.bat'
echo ===================================================
pause
