@echo off
echo ================================
echo MNIST Digit Recognition Web App
echo ================================
echo.
echo Starting the application...
echo.

:: Change to project directory
cd /d "D:\warp\ml project"

:: Activate virtual environment
call mnist_env\Scripts\activate.bat

:: Run streamlit app
streamlit run streamlit_app.py

:: Pause to see any error messages
pause