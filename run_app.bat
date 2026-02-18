@echo off
call activate base
cd /d "%~dp0"
echo Starting Tire Inspection App...
streamlit run app/app.py
pause
