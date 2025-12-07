# Windows PowerShell script to activate SCI virtual environment

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Activating SCI Virtual Environment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Activate venv
.\venv\Scripts\Activate.ps1

Write-Host "✓ Virtual environment activated!" -ForegroundColor Green
Write-Host ""
Write-Host "Quick commands:" -ForegroundColor Yellow
Write-Host "  python tests/run_tests.py          - Run all tests" -ForegroundColor White
Write-Host "  python scripts/train_sci.py         - Start training" -ForegroundColor White
Write-Host "  python scripts/evaluate.py          - Evaluate model" -ForegroundColor White
Write-Host ""
