@echo off
setlocal

cd /d "%~dp0"

echo starting robust etl preprocessor...
echo.

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

streamlit run preprocessv11.py --server.maxUploadSize 10000

endlocal
