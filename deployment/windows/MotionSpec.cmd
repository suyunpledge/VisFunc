@echo off
setlocal
set "PATH=%~dp0bin;%PATH%"
set "YOLO_CONFIG_DIR=%~dp0data\ultralytics"
"%~dp0runtime\python.exe" -m motionspec %*
exit /b %errorlevel%
