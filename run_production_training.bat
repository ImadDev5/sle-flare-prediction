@echo off
REM Production SLE Model Training Script

echo Starting Production SLE Model Training...
echo ========================================

REM Change to the script's directory
cd /d "%~dp0"

echo Activating virtual environment...
call "venv_gpu\Scripts\activate.bat"

IF %ERRORLEVEL% NEQ 0 (
    echo Failed to activate virtual environment. Exiting.
    pause
    exit /b %ERRORLEVEL%
)

echo Installing/updating dependencies from requirements.txt...
python -m pip install -r requirements.txt

IF %ERRORLEVEL% NEQ 0 (
    echo Failed to install dependencies. Exiting.
    pause
    exit /b %ERRORLEVEL%
)

echo Running data processing script...
python src/data/process_real_data.py

IF %ERRORLEVEL% NEQ 0 (
    echo Data processing failed. Exiting.
    pause
    exit /b %ERRORLEVEL%
)

echo Running production model training...
python src/models/train_real_data_model.py

IF %ERRORLEVEL% NEQ 0 (
    echo Model training failed.
    pause
    exit /b %ERRORLEVEL%
)

echo ========================================
echo Production run completed successfully!
echo ========================================
pause