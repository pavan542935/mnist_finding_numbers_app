# MNIST Digit Recognition Web App Launcher
# PowerShell Script

Write-Host "================================" -ForegroundColor Blue
Write-Host "MNIST Digit Recognition Web App" -ForegroundColor Blue
Write-Host "================================" -ForegroundColor Blue
Write-Host ""
Write-Host "Starting the application..." -ForegroundColor Green
Write-Host ""

# Change to project directory
Set-Location "D:\warp\ml project"

# Activate virtual environment
& ".\mnist_env\Scripts\Activate.ps1"

# Run streamlit app
streamlit run streamlit_app.py

# Keep window open if there's an error
Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")