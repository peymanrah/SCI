@echo off
REM Windows CMD script to activate SCI virtual environment

echo ========================================
echo  Activating SCI Virtual Environment
echo ========================================
echo.

REM Activate venv
call venv\Scripts\activate.bat

echo ✓ Virtual environment activated!
echo.
echo Quick commands:
echo   python tests/run_tests.py          - Run all tests
echo   python scripts/train_sci.py         - Start training
echo   python scripts/evaluate.py          - Evaluate model
echo.
