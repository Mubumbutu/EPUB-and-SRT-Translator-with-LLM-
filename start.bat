@echo off
REM --- Check if the virtual environment exists ---
if not exist myvenv\Scripts\activate (
    echo Virtual environment "myvenv" not found. Please run setup.bat first.
    pause
    exit /b 1
)

REM --- Activate the virtual environment ---
call myvenv\Scripts\activate

REM --- Launch the application ---
python main.py

REM --- (optional) Deactivate when done ---
deactivate
