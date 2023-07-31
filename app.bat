@echo off
cd /d D:\Analyze-and-predict-student-performance
start /B cmd /C "python -m streamlit run main.py & timeout /T 1 > nul"