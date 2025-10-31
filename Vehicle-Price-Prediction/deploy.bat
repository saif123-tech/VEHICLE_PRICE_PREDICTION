@echo off
REM Vehicle Price Prediction - Windows Deployment Script

echo 🚀 Vehicle Price Prediction - Windows Deployment Script
echo ================================================

REM Check if Docker is running
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker Desktop.
    exit /b 1
)

REM Check required files
echo 🔍 Checking required files...
if not exist "models\best_model.pkl" (
    echo ❌ Required file missing: models\best_model.pkl
    exit /b 1
)
if not exist "outputs\preprocessor.joblib" (
    echo ❌ Required file missing: outputs\preprocessor.joblib
    exit /b 1
)
if not exist "api_app.py" (
    echo ❌ Required file missing: api_app.py
    exit /b 1
)
echo ✅ All required files found

REM Handle command line arguments
if "%1"=="stop" goto stop
if "%1"=="restart" goto restart
if "%1"=="logs" goto logs
if "%1"=="status" goto status
if "%1"=="clean" goto clean
if "%1"=="help" goto help

:deploy
echo 🔨 Building Docker images...
docker build -t vehicle-price-api:latest .
if %errorlevel% neq 0 (
    echo ❌ Failed to build Docker image
    exit /b 1
)

echo 🚀 Starting services...
if "%1"=="--with-dashboard" (
    docker-compose --profile dashboard up -d
    echo ✅ Started API and Dashboard services
    echo 📊 Dashboard: http://localhost:8501
) else (
    docker-compose up -d vehicle-price-api
    echo ✅ Started API service
)

echo 🌐 API: http://localhost:8000
echo 📚 API Docs: http://localhost:8000/docs

echo 🔍 Performing health check...
timeout /t 10 /nobreak > nul
curl -f http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ API is healthy!
) else (
    echo ⚠️ API might still be starting up. Check logs if issues persist.
)

echo 🎉 Deployment completed!
goto end

:stop
echo 🛑 Stopping services...
docker-compose down
goto end

:restart
echo 🔄 Restarting services...
docker-compose down
goto deploy

:logs
docker-compose logs -f
goto end

:status
docker-compose ps
goto end

:clean
echo 🧹 Cleaning up...
docker-compose down --volumes --remove-orphans
docker system prune -f
goto end

:help
echo Usage: deploy.bat [stop^|restart^|logs^|status^|clean^|help] [--with-dashboard]
echo.
echo Commands:
echo   stop     - Stop services
echo   restart  - Restart services
echo   logs     - Show service logs
echo   status   - Show service status
echo   clean    - Clean up containers and volumes
echo   help     - Show this help
echo.
echo Options:
echo   --with-dashboard  - Include Streamlit dashboard
goto end

:end
