@echo off
setlocal EnableDelayedExpansion

:: ============================================================
:: OceanEmbed - One-Click Local Launcher
:: Double-click this file to start both backend and frontend.
:: The browser opens automatically ONLY after localhost:3000
:: responds — no more ERR_CONNECTION_REFUSED on first open.
:: ============================================================

set "PROJECT_ROOT=%~dp0..\.."
pushd "%PROJECT_ROOT%"
set "PROJECT_ROOT=%CD%"
popd

echo.
echo  OceanEmbed Local Launcher
echo  Project root: %PROJECT_ROOT%
echo.

:: ---- Python virtual environment ----
set "VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment not found at:
    echo         %VENV_PYTHON%
    echo.
    echo  To create it, run from %PROJECT_ROOT%:
    echo    py -3.11 -m venv .venv
    echo    .venv\Scripts\pip install -e ".[dev]"
    echo.
    pause
    exit /b 1
)

:: ---- Data root ----
if "%OCEANEMBED_DATA_ROOT%"=="" (
    pushd "%PROJECT_ROOT%\.."
    set "OCEANEMBED_DATA_ROOT=%CD%\oceanembed-data"
    popd
    echo [INFO] OCEANEMBED_DATA_ROOT not set.
    echo        Using default: %OCEANEMBED_DATA_ROOT%
    echo.
)

:: ---- Node / npm check ----
where npm >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm not found on PATH.
    echo         Install Node.js 18+ from https://nodejs.org/
    pause
    exit /b 1
)

:: ---- npm install if needed ----
set "WEB_DIR=%PROJECT_ROOT%\web"
if not exist "%WEB_DIR%\node_modules" (
    echo [INFO] node_modules not found. Running npm install...
    pushd "%WEB_DIR%"
    npm install
    if errorlevel 1 (
        echo [ERROR] npm install failed.
        popd
        pause
        exit /b 1
    )
    popd
)

:: ---- Delete stale helpers ----
set "BACKEND_CMD=%TEMP%\oe_backend.cmd"
set "FRONTEND_CMD=%TEMP%\oe_frontend.cmd"
if exist "%BACKEND_CMD%"  del /f /q "%BACKEND_CMD%"
if exist "%FRONTEND_CMD%" del /f /q "%FRONTEND_CMD%"

:: ---- Backend helper ----
(
    echo @echo off
    echo title OceanEmbed Backend
    echo cd /d "%PROJECT_ROOT%"
    echo set OCEANEMBED_DATA_ROOT=%OCEANEMBED_DATA_ROOT%
    echo echo.
    echo echo  OceanEmbed Backend
    echo echo  API:  http://localhost:8000
    echo echo  Docs: http://localhost:8000/api/docs
    echo echo.
    echo "%VENV_PYTHON%" -m uvicorn oceanembed.api.app:app --reload --port 8000
    echo pause
) > "%BACKEND_CMD%"

:: ---- Frontend helper ----
:: Starts Next.js in the background, then polls until port 3000 responds,
:: then opens the browser. This guarantees no ERR_CONNECTION_REFUSED.
(
    echo @echo off
    echo title OceanEmbed Frontend
    echo cd /d "%WEB_DIR%"
    echo echo.
    echo echo  OceanEmbed Frontend starting...
    echo echo  Will open http://localhost:3000/landing when ready.
    echo echo.
    echo.
    echo :: Start Next.js dev server in the background
    echo start /b npm run dev
    echo.
    echo :: Poll until port 3000 responds (check every 3 seconds, up to 120s)
    echo set TRIES=0
    echo :poll_loop
    echo set /a TRIES+=1
    echo if %TRIES% GTR 40 (
    echo     echo [WARN] Frontend did not start in time. Opening anyway...
    echo     goto open_browser
    echo )
    echo timeout /t 3 /nobreak ^>nul
    echo powershell -NoProfile -NonInteractive -Command ^
    echo     "try{[void](Invoke-WebRequest 'http://localhost:3000' -TimeoutSec 2 -UseBasicParsing);exit 0}catch{exit 1}" ^>nul 2^>^&1
    echo if errorlevel 1 goto poll_loop
    echo.
    echo :open_browser
    echo echo  Frontend ready. Opening landing page...
    echo start "" "http://localhost:3000/landing"
    echo.
    echo :: Keep window open so npm output stays visible
    echo echo  Press Ctrl+C to stop the frontend server.
    echo timeout /t 99999 /nobreak ^>nul
) > "%FRONTEND_CMD%"

:: ---- Launch backend ----
echo [STARTING] Backend  (FastAPI  on http://localhost:8000)
start "OceanEmbed Backend" cmd /k "%BACKEND_CMD%"
timeout /t 3 /nobreak >nul

:: ---- Launch frontend + auto-open ----
echo [STARTING] Frontend (Next.js on http://localhost:3000)
echo            Browser will open automatically when ready.
start "OceanEmbed Frontend" cmd /k "%FRONTEND_CMD%"

echo.
echo  Both servers are starting in separate windows.
echo  The browser will open at http://localhost:3000/landing
echo  automatically once the frontend is ready.
echo.
echo  Backend:   http://localhost:8000/api/health
echo  Landing:   http://localhost:3000/landing
echo  Dashboard: http://localhost:3000/dashboard
echo  API docs:  http://localhost:8000/api/docs
echo.
echo  Close the terminal windows to stop the servers.
echo.

endlocal
