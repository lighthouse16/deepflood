@echo off
REM Initialize Git Repository and Create Initial Commit
REM Run this script from the project root directory

cd /d "d:\Study\SEAS flood"

REM Configure Git (one time only)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

REM Initialize repository
git init
git add .
git commit -m "Initial commit: Flood forecasting ML project for Central Vietnam"

REM Display instructions for remote setup
echo.
echo ===================================================
echo Git repository initialized successfully!
echo ===================================================
echo.
echo Next steps:
echo 1. Create a repository on GitHub: https://github.com/new
echo 2. Run these commands with your GitHub info:
echo.
echo    git remote add origin https://github.com/yourusername/flood-forecasting-ml.git
echo    git branch -M main
echo    git push -u origin main
echo.
echo ===================================================
pause
