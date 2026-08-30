@echo off
rem Guided Causal Discovery Hub - port 8002 (ladder 8000, workbench 8001)
rem Machine-local configuration (detector paths, provider env) lives in
rem run_hub.local.bat, which is not tracked; create it beside this file.
pushd "%~dp0"
if exist run_hub.local.bat call .\run_hub.local.bat

rem A second launch while a server holds the port would die instantly and,
rem from Explorer, close the window before the error could be read.
netstat -ano | findstr /r /c:":8002 .*LISTENING" >nul
if not errorlevel 1 (
    echo Port 8002 is already in use - the hub may already be running.
    echo Open http://localhost:8002 or stop the other server first.
    pause
    popd
    exit /b 1
)

py -3.13 -m shiny run --port 8002 --app-dir app --launch-browser hub.app:app
set EXIT=%ERRORLEVEL%
if not "%EXIT%"=="0" (
    echo.
    echo The hub exited with an error - see the messages above.
    pause
)
popd
exit /b %EXIT%
