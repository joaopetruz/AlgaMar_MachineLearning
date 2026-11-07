@echo off
rem Sobe a API do ML (porta 8000) e o dashboard local (porta 3000), cada um na sua janela.
rem Para parar: feche as duas janelas.
cd /d "%~dp0"

set "PY=%USERPROFILE%\.venvs\algamar-ml\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0venv\Scripts\python.exe"

start "AlgaMar API (8000)" cmd /k ""%PY%" -m uvicorn api:app --port 8000"
start "AlgaMar Dashboard (3000)" cmd /k ""%PY%" dashboard_local.py"
