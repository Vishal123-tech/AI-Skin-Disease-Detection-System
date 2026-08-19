@echo off
cd /d "%~dp0"
set "PY=C:\Users\vy355\Documents\Codex\tfenv\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=py"
echo Starting AI Skin Disease Detection project...
echo Open http://127.0.0.1:5000 in your browser.
"%PY%" app.py
pause
