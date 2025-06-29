@echo off
REM Automated Code Quality Enhancement for SLE TAGT Project
REM Windows Batch Script for Easy Execution

echo ========================================
echo SLE TAGT Code Quality Automation
echo ========================================
echo.

REM Check if virtual environment exists
if exist "venv_gpu\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv_gpu\Scripts\activate.bat
) else (
    echo Virtual environment not found. Using system Python...
)

REM Check Python availability
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python or activate your virtual environment
    pause
    exit /b 1
)

echo Python version:
python --version
echo.

REM Check GPU availability
echo Checking system capabilities...
python -c "import torch; print(f'PyTorch CUDA Available: {torch.cuda.is_available()}')" 2>nul
if errorlevel 1 (
    echo PyTorch not installed - will install during automation
)

echo.
echo Starting automated code quality enhancement...
echo This may take several minutes...
echo.

REM Run the automation script
python automate_code_quality.py

if errorlevel 1 (
    echo.
    echo ERROR: Automation failed!
    echo Check the log file: code_quality_automation.log
    pause
    exit /b 1
)

echo.
echo ========================================
echo Automation completed successfully!
echo ========================================
echo.
echo Generated reports are available in: quality_reports\
echo Open quality_reports\quality_dashboard.html to view results
echo.

REM Open the dashboard in default browser
set /p open_dashboard="Open quality dashboard in browser? (y/n): "
if /i "%open_dashboard%"=="y" (
    start quality_reports\quality_dashboard.html
)

echo.
echo Press any key to exit...
pause >nul