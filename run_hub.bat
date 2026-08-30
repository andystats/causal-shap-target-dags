@echo off
rem Guided Causal Discovery Hub - port 8002 (ladder 8000, workbench 8001)
rem Machine-local configuration (detector paths, provider env) lives in
rem run_hub.local.bat, which is not tracked; create it beside this file.
pushd "%~dp0"
if exist run_hub.local.bat call .\run_hub.local.bat
py -3.13 -m shiny run --port 8002 --app-dir app hub.app:app
set EXIT=%ERRORLEVEL%
popd
exit /b %EXIT%
