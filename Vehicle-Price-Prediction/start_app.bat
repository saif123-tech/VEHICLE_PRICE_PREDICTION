@echo off
echo 🚗 Starting Vehicle Price Prediction App...
echo.

:: Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ❌ Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Check if model files exist
if not exist "models\best_model.pkl" (
    echo ❌ Model files not found. Please train the model first:
    echo    python train.py
    pause
    exit /b 1
)

if not exist "outputs\preprocessor.joblib" (
    echo ❌ Preprocessor not found. Please run data processing first:
    echo    python data\dataloader.py --dataset_dir dataset\ --out outputs\
    pause
    exit /b 1
)

:: Start the FastAPI server
echo ✅ Starting FastAPI server with frontend...
echo 📱 Frontend will be available at: http://127.0.0.1:8000
echo 📚 API documentation at: http://127.0.0.1:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

uvicorn api_app:app --host 127.0.0.1 --port 8000 --reload

pause
