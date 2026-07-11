@echo off
cd /d "%~dp0"
py -3.13 -m shiny run --reload --port 8010 app.py
