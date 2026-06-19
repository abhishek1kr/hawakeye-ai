@echo off
echo ===================================================
echo   Starting Hawkeye AI - Road Infrastructure Suite  
echo ===================================================
echo.

echo [1/3] Starting FastAPI Backend Server...
start "Hawkeye AI Backend" cmd /k "cd /d d:\abhi_project\Abhishek_Project\backend && .\venv\Scripts\python.exe -m src.api.server"

echo [2/3] Starting React Frontend Application...
start "Hawkeye AI Frontend" cmd /k "cd /d d:\abhi_project\Abhishek_Project\frontend && npm run dev"

echo.
echo Waiting 5 seconds for servers to start...
timeout /t 5 >nul

echo [3/3] Opening browser...
start http://localhost:5173/

echo.
echo Hawkeye AI has been launched!
echo Keep the backend and frontend command windows open while using the application.
echo You can close this window now.
timeout /t 3 >nul
