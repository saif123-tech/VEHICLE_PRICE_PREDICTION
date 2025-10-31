#!/bin/bash
# Vehicle Price Prediction - Deployment Script

set -e

echo "🚀 Vehicle Price Prediction - Deployment Script"
echo "================================================"

# Check if required files exist
check_files() {
    echo "🔍 Checking required files..."
    
    required_files=("models/best_model.pkl" "outputs/preprocessor.joblib" "api_app.py" "Dockerfile")
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            echo "❌ Required file missing: $file"
            exit 1
        fi
    done
    
    echo "✅ All required files found"
}

# Build Docker images
build_images() {
    echo "🔨 Building Docker images..."
    
    # Build main API image
    docker build -t vehicle-price-api:latest .
    echo "✅ Built vehicle-price-api image"
    
    # Build Streamlit image if requested
    if [ "$1" = "--with-dashboard" ]; then
        docker build -f Dockerfile.streamlit -t vehicle-streamlit:latest .
        echo "✅ Built vehicle-streamlit image"
    fi
}

# Start services
start_services() {
    echo "🚀 Starting services..."
    
    if [ "$1" = "--with-dashboard" ]; then
        docker-compose --profile dashboard up -d
        echo "✅ Started API and Dashboard services"
        echo "📊 Dashboard: http://localhost:8501"
    else
        docker-compose up -d vehicle-price-api
        echo "✅ Started API service"
    fi
    
    echo "🌐 API: http://localhost:8000"
    echo "📚 API Docs: http://localhost:8000/docs"
}

# Health check
health_check() {
    echo "🔍 Performing health check..."
    
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8000/health >/dev/null 2>&1; then
            echo "✅ API is healthy!"
            return 0
        fi
        
        echo "⏳ Waiting for API to start... (attempt $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    
    echo "❌ API health check failed"
    return 1
}

# Main deployment function
deploy() {
    echo "Starting deployment with options: $@"
    
    check_files
    build_images "$@"
    start_services "$@"
    health_check
    
    echo ""
    echo "🎉 Deployment completed successfully!"
    echo "📋 Service Status:"
    docker-compose ps
}

# Handle command line arguments
case "$1" in
    "start")
        start_services "${@:2}"
        ;;
    "stop")
        echo "🛑 Stopping services..."
        docker-compose down
        ;;
    "restart")
        echo "🔄 Restarting services..."
        docker-compose down
        deploy "${@:2}"
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "status")
        docker-compose ps
        ;;
    "clean")
        echo "🧹 Cleaning up..."
        docker-compose down --volumes --remove-orphans
        docker system prune -f
        ;;
    "")
        deploy
        ;;
    *)
        echo "Usage: $0 [start|stop|restart|logs|status|clean] [--with-dashboard]"
        echo ""
        echo "Commands:"
        echo "  start    - Start services"
        echo "  stop     - Stop services"
        echo "  restart  - Restart services"
        echo "  logs     - Show service logs"
        echo "  status   - Show service status"
        echo "  clean    - Clean up containers and volumes"
        echo ""
        echo "Options:"
        echo "  --with-dashboard  - Include Streamlit dashboard"
        exit 1
        ;;
esac
