@echo off
setlocal

cd /d "%~dp0"

echo starting local csv anonymizer and chunker...
echo.

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

streamlit run csv_anonymizer_chunker.py --server.maxUploadSize 10000

endlocal
