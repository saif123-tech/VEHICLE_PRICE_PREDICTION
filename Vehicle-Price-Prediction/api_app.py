#!/usr/bin/env python3
"""
Vehicle Price Prediction FastAPI Application

Usage:
    uvicorn api_app:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /                     - Health check
    POST /predict              - Price prediction
    GET  /model-info           - Model information
    GET  /docs                 - API documentation
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List
from datetime import datetime
import os
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator


# Pydantic models for request/response validation
class CarFeatures(BaseModel):
    """Car features input model"""
    make: Optional[str] = Field("Maruti", description="Car manufacturer")
    year: Optional[int] = Field(
        2019, ge=1990, le=2025, description="Manufacturing year")
    fuel: Optional[str] = Field("Petrol", description="Fuel type")
    transmission: Optional[str] = Field(
        "Manual", description="Transmission type")
    engine_cc: Optional[int] = Field(
        1200, ge=50, le=8000, description="Engine displacement in CC")
    km_driven: Optional[int] = Field(
        50000, ge=0, le=1000000, description="Kilometers driven")
    max_power_bhp: Optional[float] = Field(
        80.0, ge=10, le=1000, description="Maximum power in BHP")
    mileage_value: Optional[float] = Field(
        15.0, ge=5, le=50, description="Mileage in kmpl")
    seats: Optional[int] = Field(5, ge=2, le=10, description="Number of seats")
    owner: Optional[str] = Field("First", description="Owner type")
    seller_type: Optional[str] = Field("Individual", description="Seller type")
    torque_nm: Optional[float] = Field(
        100.0, ge=50, le=1000, description="Torque in Nm")
    torque_rpm: Optional[float] = Field(
        2000.0, ge=1000, le=8000, description="Torque RPM")
    mileage_unit: Optional[str] = Field("kmpl", description="Mileage unit")

    @validator('fuel')
    def validate_fuel(cls, v):
        allowed_fuels = ['Petrol', 'Diesel',
                         'CNG', 'Electric', 'Hybrid', 'LPG']
        if v not in allowed_fuels:
            raise ValueError(
                f'Fuel must be one of: {", ".join(allowed_fuels)}')
        return v

    @validator('transmission')
    def validate_transmission(cls, v):
        allowed_transmissions = ['Manual', 'Automatic']
        if v not in allowed_transmissions:
            raise ValueError(
                f'Transmission must be one of: {", ".join(allowed_transmissions)}')
        return v

    @validator('owner')
    def validate_owner(cls, v):
        allowed_owners = ['First', 'Second', 'Third',
                          'Fourth & Above', 'Test Drive Car']
        if v not in allowed_owners:
            raise ValueError(
                f'Owner must be one of: {", ".join(allowed_owners)}')
        return v


class PredictionResponse(BaseModel):
    """Prediction response model"""
    predicted_price: float = Field(description="Predicted price in rupees")
    formatted_price: str = Field(
        description="Formatted price with currency symbol")
    confidence_level: str = Field(description="Confidence level of prediction")
    price_category: str = Field(
        description="Price category (Budget/Mid-range/Premium/Luxury)")
    model_used: str = Field(description="Model algorithm used")
    features_used: Dict[str, Any] = Field(
        description="Input features processed")
    prediction_timestamp: str = Field(description="Timestamp of prediction")


class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str
    model_version: str
    training_date: str
    feature_count: int
    status: str


class PredictionError(BaseModel):
    """Error response model"""
    error: str
    detail: str
    timestamp: str


# Global variables for model and preprocessor
model = None
preprocessor = None
model_name = None
feature_names = []


def load_artifacts():
    """Load model and preprocessor on startup"""
    global model, preprocessor, model_name, feature_names

    try:
        # Load model
        model_path = "models/best_model.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model_pack = joblib.load(model_path)
        model = model_pack["model"]
        model_name = model_pack.get("algo", "Unknown")

        # Load preprocessor
        preprocessor_path = "outputs/preprocessor.joblib"
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(
                f"Preprocessor file not found: {preprocessor_path}")

        preprocessor = joblib.load(preprocessor_path)

        # Load feature names
        try:
            data_path = "outputs/processed_data.pkl"
            if os.path.exists(data_path):
                data_blob = joblib.load(data_path)
                feature_names = data_blob.get("feature_names", [])
        except Exception:
            pass

        print(
            f"✅ Successfully loaded {model_name} model with {len(feature_names)} features")

    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        raise e


def prepare_features(car_data: CarFeatures) -> pd.DataFrame:
    """Prepare features for prediction"""

    # Convert Pydantic model to dict
    data_dict = car_data.dict()

    # Calculate age from year
    if data_dict.get('year'):
        current_year = datetime.now().year
        data_dict['age'] = current_year - data_dict['year']
    else:
        data_dict['age'] = 5  # default age

    # Ensure all required features are present with defaults
    feature_defaults = {
        'km_driven': 50000,
        'mileage_value': 15.0,
        'engine_cc': 1200,
        'max_power_bhp': 80,
        'torque_nm': 100,
        'torque_rpm': 2000,
        'seats': 5,
        'age': 5,
        'fuel': 'Petrol',
        'transmission': 'Manual',
        'owner': 'First',
        'seller_type': 'Individual',
        'mileage_unit': 'kmpl',
        'make': 'Maruti'
    }

    # Fill missing values
    for feature, default in feature_defaults.items():
        if feature not in data_dict or data_dict[feature] is None:
            data_dict[feature] = default

    # Convert to DataFrame
    df = pd.DataFrame([data_dict])

    return df


def get_price_category(price: float) -> str:
    """Categorize price range"""
    if price < 300000:
        return "Budget car 🚗"
    elif price < 800000:
        return "Mid-range car 🚙"
    elif price < 1500000:
        return "Premium car 🚘"
    else:
        return "Luxury car 🏎️"


def get_confidence_level(price: float) -> str:
    """Determine confidence level based on price range"""
    if price < 500000 or price > 2000000:
        return "High confidence"
    elif price < 1000000:
        return "Medium confidence"
    else:
        return "Good confidence"


# Initialize FastAPI app
app = FastAPI(
    title="Vehicle Price Prediction API",
    description="AI-powered vehicle price prediction service using XGBoost 🚗💨",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Startup event to load model
@app.on_event("startup")
async def startup_event():
    """Load model and preprocessor on startup"""
    load_artifacts()


# Root endpoint - serve the frontend
@app.get("/")
async def serve_frontend():
    """Serve the main frontend application"""
    return FileResponse("frontend/index.html")


# API root endpoint
@app.get("/api")
async def api_root():
    """API health check endpoint"""
    return {
        "message": "Vehicle Price Prediction API is running 🚗💨",
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


# Model information endpoint
@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    return ModelInfo(
        model_name=model_name,
        model_version="1.0.0",
        training_date="2025-08-17",
        feature_count=len(feature_names),
        status="active"
    )


# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_price(car_features: CarFeatures):
    """Predict vehicle price based on features"""

    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or preprocessor not loaded"
        )

    try:
        # Prepare features
        df = prepare_features(car_features)

        # Select features for preprocessing
        numeric_features = ['km_driven', 'mileage_value', 'engine_cc', 'max_power_bhp',
                            'torque_nm', 'torque_rpm', 'seats', 'age']
        categorical_features = ['fuel', 'transmission', 'owner', 'seller_type',
                                'mileage_unit', 'make']

        feature_data = df[numeric_features + categorical_features]

        # Apply preprocessing
        X_processed = preprocessor.transform(feature_data)

        # Make prediction
        prediction = model.predict(X_processed)[0]

        # Ensure prediction is positive
        prediction = max(prediction, 50000)  # Minimum price

        # Format response
        response = PredictionResponse(
            predicted_price=float(prediction),
            formatted_price=f"₹{prediction:,.0f}",
            confidence_level=get_confidence_level(prediction),
            price_category=get_price_category(prediction),
            model_used=model_name,
            features_used=df.iloc[0].to_dict(),
            prediction_timestamp=datetime.now().isoformat()
        )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


# Batch prediction endpoint
@app.post("/predict-batch", response_model=List[PredictionResponse])
async def predict_batch(car_features_list: List[CarFeatures]):
    """Predict prices for multiple vehicles"""

    if len(car_features_list) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 100 predictions per batch"
        )

    results = []
    for car_features in car_features_list:
        try:
            result = await predict_price(car_features)
            results.append(result)
        except Exception as e:
            # Add error result for failed predictions
            error_result = PredictionResponse(
                predicted_price=0.0,
                formatted_price="Error",
                confidence_level="No confidence",
                price_category="Error",
                model_used=model_name,
                features_used=car_features.dict(),
                prediction_timestamp=datetime.now().isoformat()
            )
            results.append(error_result)

    return results


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not_loaded"
    preprocessor_status = "loaded" if preprocessor is not None else "not_loaded"

    return {
        "status": "healthy",
        "model_status": model_status,
        "preprocessor_status": preprocessor_status,
        "feature_count": len(feature_names),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
