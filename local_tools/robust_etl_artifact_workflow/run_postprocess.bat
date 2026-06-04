@echo off
setlocal

cd /d "%~dp0"

echo starting robust etl postprocessor...
echo.

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

streamlit run postprocessv8.py --server.maxUploadSize 10000

endlocal
