@echo off
setlocal
pushd "%~dp0"
py -3.13 -m shiny run --port 8001 app.py
set "WORKBENCH_EXIT=%ERRORLEVEL%"
popd
exit /b %WORKBENCH_EXIT%
