@echo off
echo 🚗 Vehicle Price Prediction Setup
echo ================================
echo.

:: Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

:: Upgrade pip
echo 📈 Upgrading pip...
python -m pip install --upgrade pip

:: Install requirements
echo 📦 Installing requirements...
pip install -r requirements.txt

:: Install additional packages for the frontend
echo 📦 Installing additional packages...
pip install uvicorn fastapi python-multipart

echo.
echo ✅ Setup completed successfully!
echo.
echo 📋 Next steps:
echo 1. Process data: python data\dataloader.py --dataset_dir dataset\ --out outputs\
echo 2. Train model: python train.py
echo 3. Start app: start_app.bat
echo.
pause
