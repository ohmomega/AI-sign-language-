@echo off
cd /d "%~dp0app"
set PY=py
if exist "%~dp0.venv\Scripts\python.exe" set PY="%~dp0.venv\Scripts\python.exe"
start "" http://localhost:8000
%PY% -m http.server 8000
