@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set CONDA_PKGS_DIRS=%CD%\.cache\conda-pkgs
set CONDA_ENVS_PATH=%CD%\env
if not exist "env\local-site\python.exe" (
  if not exist "%USERPROFILE%\miniconda3\python.exe" (
    echo Miniconda was not found. See docs\SETUP.md
    pause
    exit /b 1
  )
  "%USERPROFILE%\miniconda3\python.exe" -m conda create --prefix "%CD%\env\local-site" --override-channels -c conda-forge python=3.11 pip -y
  if errorlevel 1 goto failed
)
"env\local-site\python.exe" -m pip install --cache-dir "%CD%\.cache\pip" -r web\requirements-lock.txt
if errorlevel 1 goto failed
"env\local-site\python.exe" tools\prepare_pose_models.py
if errorlevel 1 goto failed
"env\local-site\python.exe" models\breaking_v1\verify_local_bundle.py
if errorlevel 1 goto failed
echo Setup completed. Double-click start_local.bat to run the website.
pause
exit /b 0
:failed
echo Setup failed. Keep this window open and share the error message.
pause
exit /b 1
