@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set OMP_NUM_THREADS=4
if not exist "env\local-site\python.exe" (
  echo Local environment is missing. See docs\local_website.md
  pause
  exit /b 1
)
echo HYPER Breaking - http://127.0.0.1:8000
echo Keep this window open. Press Ctrl+C to stop.
"env\local-site\python.exe" -m uvicorn web.backend.app:app --host 127.0.0.1 --port 8000
pause
