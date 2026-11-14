@echo off
REM Stable Diffusion 3D Converter - GUI Launcher (Fixed - waits for GUI)
if /I "%~1"=="-hidden" goto :MAIN
set "VBS=%TEMP%\SD3D_HideLaunch.vbs"
> "%VBS%" echo Set WshShell = CreateObject("WScript.Shell")
>>"%VBS%" echo WshShell.Run Chr(34) ^& "%~f0" ^& Chr(34) ^& " -hidden", 0
>>"%VBS%" echo Set WshShell = Nothing
wscript //nologo "%VBS%"
del /f /q "%VBS%" >nul 2>&1
exit /b

:MAIN
chcp 65001 >nul
cd /d "%~dp0"

if exist "venv_SD3D\Scripts\pythonw.exe" (
    set PYTHONW=venv_SD3D\Scripts\pythonw.exe
    set PYTHON=venv_SD3D\Scripts\python.exe
) else (
    set PYTHONW=pythonw
    set PYTHON=python
)

if not exist "SD3D_GUI.py" (
    echo [ERROR] SD3D_GUI.py not found
    pause
    exit /b
)

echo Starting GUI, please wait...
REM Clear old log
echo. > sd3d_debug.log

REM Try pythonw first (hidden CMD)
start "" "%PYTHONW%" "SD3D_GUI.py"

REM Wait 3 seconds for GUI to open, then close this launcher
REM GUI will stay alive because it's separate process
timeout /t 3 /nobreak >nul

REM Check if GUI process is running
tasklist /FI "IMAGENAME eq pythonw.exe" 2>nul | find /I "pythonw.exe" >nul
if %ERRORLEVEL% NEQ 0 (
    tasklist /FI "IMAGENAME eq python.exe" 2>nul | find /I "python.exe" >nul
)

REM Check log file - if GUI crashed, log will contain error
if exist "sd3d_debug.log" (
    for %%A in ("sd3d_debug.log") do if %%~zA GTR 10 (
        echo GUI started, closing launcher...
        timeout /t 1 /nobreak >nul
        exit
    )
)

REM If log still empty, GUI may have failed - try with console to show error
echo GUI did not start with pythonw, trying with console to show error...
"%PYTHON%" "SD3D_GUI.py"
pause