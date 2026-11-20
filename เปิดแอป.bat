@echo off
cd /d "%~dp0app"
start "" http://localhost:8000
python -m http.server 8000
