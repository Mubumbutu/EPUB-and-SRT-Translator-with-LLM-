@echo off
REM --- Create a virtual environment with Python 3.11 ---
py -3.11 -m venv myvenv

REM --- Activate the environment ---
call myvenv\Scripts\activate

REM --- Upgrade pip ---
python -m pip install --upgrade pip

REM --- Install dependencies ---
pip install -r requirements.txt

REM --- Done! ---
echo.
echo The virtual environment "myvenv" is now active and all packages have been installed.
pause
