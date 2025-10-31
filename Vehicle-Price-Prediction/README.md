# Vehicle Price Prediction 🚗💰

A comprehensive machine learning system for predicting vehicle prices using advanced regression models with multiple deployment options.

## 🌟 Features

- **High Accuracy**: 90.8% R² score with XGBoost model
- **Multiple Interfaces**: CLI, REST API, Web Dashboard
- **Production Ready**: Docker containerization and deployment scripts
- **Model Explainability**: SHAP analysis for interpretability
- **Real-time Predictions**: Fast inference with preprocessing pipeline

## 📊 Model Performance

- **R² Score**: 90.8% (Excellent accuracy)
- **Average Error**: ₹129,795
- **Median Error**: ₹59,211
- **Training Data**: 12,283 vehicles from multiple datasets

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd vehicle-price-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Process data
python data/dataloader.py --dataset_dir dataset/ --out outputs/

# Train model with GPU support
python train.py --use_gpu --n_iter 10

# Evaluate model
python evaluate.py
```

## 🎯 Usage Options

### 1. Command Line Interface

```bash
# Basic prediction
python predict.py --make "Toyota" --fuel "Petrol" --year 2018 --engine_cc 1200

# Interactive mode
python predict.py --interactive

# JSON input
python predict.py --json car_details.json
```

### 2. REST API Service

```bash
# Start FastAPI server
uvicorn api_app:app --host 0.0.0.0 --port 8000

# Test the API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"make": "Toyota", "year": 2018, "fuel": "Petrol"}'

# View API documentation
# Visit: http://localhost:8000/docs
```

### 3. Web Dashboard

```bash
# Start Streamlit dashboard
streamlit run streamlit_app.py

# Visit: http://localhost:8501
```

### 4. Docker Deployment

```bash
# Quick deployment
./deploy.sh

# With dashboard
./deploy.sh --with-dashboard

# Or use Docker Compose
docker-compose up --build
```

## 📁 Project Structure

```
vehicle-price-prediction/
├── data/
│   └── dataloader.py          # Data processing pipeline
├── dataset/                   # Raw CSV datasets
├── models/                    # Trained model files
├── outputs/                   # Processed data and metrics
├── frontend/                  # Web interface files
├── api/                       # API service files
├── train.py                   # Model training script
├── evaluate.py                # Model evaluation script
├── predict.py                 # CLI prediction script
├── api_app.py                 # FastAPI web service
├── streamlit_app.py           # Streamlit dashboard
├── Dockerfile                 # Container configuration
├── docker-compose.yml         # Multi-service deployment
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔧 API Endpoints

### FastAPI Service

- **GET** `/` - Health check
- **POST** `/predict` - Price prediction
- **GET** `/model-info` - Model information
- **POST** `/predict-batch` - Batch predictions
- **GET** `/health` - Service health status
- **GET** `/docs` - Interactive API documentation

### Request Example

```json
{
  "make": "Toyota",
  "year": 2018,
  "fuel": "Petrol",
  "transmission": "Manual",
  "engine_cc": 1200,
  "km_driven": 50000,
  "max_power_bhp": 85.0,
  "owner": "First"
}
```

### Response Example

```json
{
  "predicted_price": 682007.0,
  "formatted_price": "₹682,007",
  "confidence_level": "High confidence",
  "price_category": "Mid-range car 🚙",
  "model_used": "XGBRegressor",
  "prediction_timestamp": "2025-08-17T13:04:58.729665"
}
```

## 📊 Model Features

### Input Features (14 core features)
- **Numeric**: km_driven, engine_cc, max_power_bhp, age, mileage, torque, seats
- **Categorical**: make, fuel, transmission, owner, seller_type

### Processed Features (108 after encoding)
- One-hot encoded categorical variables
- Scaled numeric features
- Engineered features (age from year)

### Top Important Features
1. **transmission_Manual** (17.6%)
2. **max_power_bhp** (16.9%)
3. **make_None** (9.0%)
4. **transmission_Automatic** (5.8%)
5. **owner_0** (5.1%)

## 🔍 Model Explainability

### SHAP Analysis
- Feature importance rankings
- Individual prediction explanations
- Global model behavior insights
- Dependence plots for key features

```python
# Run SHAP analysis
jupyter notebook vehicle_prediction_deployment.ipynb
```

## 🐳 Docker Deployment

### Single Container

```bash
# Build image
docker build -t vehicle-price-api .

# Run container
docker run -p 8000:8000 vehicle-price-api
```

### Multi-Service with Docker Compose

```bash
# Start all services
docker-compose up --build

# Start with dashboard
docker-compose --profile dashboard up --build

# Stop services
docker-compose down
```

### Services Included
- **API Service**: FastAPI on port 8000
- **Dashboard**: Streamlit on port 8501
- **Reverse Proxy**: Nginx on port 80

## 🧪 Testing

```bash
# Test API endpoints
python test_api.py

# Run evaluation
python evaluate.py

# Check model performance
cat outputs/enhanced_test_metrics.json
```

## 📈 Performance Analysis

### By Price Range
- **Under ₹5L** (58.3%): MAE ₹62K, R² 0.48
- **₹5L-10L** (29.5%): MAE ₹111K, R² -0.53
- **₹10L-20L** (7.2%): MAE ₹264K, R² -0.34
- **Above ₹20L** (5.0%): MAE ₹826K, R² 0.80

### Model Strengths
- Excellent overall accuracy (90.8% R²)
- Good performance on budget cars
- Strong performance on luxury vehicles
- Fast inference (~50ms per prediction)

### Areas for Improvement
- Mid-range segment (₹5L-20L) accuracy
- Feature engineering for specific price ranges
- Ensemble methods for better generalization

## 🛠️ Development

### Adding New Features

1. **Data Features**: Modify `data/dataloader.py`
2. **Model Architecture**: Update `train.py`
3. **API Endpoints**: Extend `api_app.py`
4. **Dashboard Components**: Enhance `streamlit_app.py`

### Configuration

- **Model Parameters**: `train.py` hyperparameter grids
- **API Settings**: `api_app.py` FastAPI configuration
- **Docker Settings**: `Dockerfile` and `docker-compose.yml`

## 📋 Requirements

### Python Dependencies
```
scikit-learn>=1.7.0
pandas>=2.3.0
numpy>=2.2.0
xgboost>=2.1.0
lightgbm>=4.5.0
fastapi>=0.116.0
streamlit>=1.48.0
shap>=0.48.0
uvicorn>=0.35.0
matplotlib>=3.10.0
seaborn>=0.13.0
```

### System Requirements
- **Python**: 3.11+
- **Memory**: 4GB+ RAM
- **Storage**: 2GB+ available space
- **GPU**: Optional (NVIDIA CUDA for training acceleration)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Datasets**: CarDekho vehicle data
- **Libraries**: scikit-learn, XGBoost, FastAPI, Streamlit
- **Tools**: Docker, SHAP, Jupyter

## 📞 Support

For questions and support:
- 📧 Email: support@vehiclepriceprediction.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 📖 Documentation: [Wiki](https://github.com/your-repo/wiki)

---

**Made with ❤️ for the automotive industry** 🚗

*Predict smarter, buy better!* 💡
