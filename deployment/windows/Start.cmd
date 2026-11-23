@echo off
setlocal
pushd "%~dp0"
runtime\python.exe -c "import ctypes; ctypes.WinDLL('msvcp140.dll'); ctypes.WinDLL('vcruntime140_1.dll')" >nul 2>&1
if errorlevel 1 (
    start "" /wait "prerequisites\vc_redist.x64.exe" /install /passive /norestart
)
runtime\python.exe launch.py %*
set "result=%errorlevel%"
popd
if not "%result%"=="0" if "%~1"=="" pause
exit /b %result%
