#!/usr/bin/env python3
"""
Vehicle Price Prediction - Standalone Predictor

Usage:
    # CLI mode
    python predict.py --make "Toyota" --fuel "Petrol" --transmission "Manual" --engine_cc 1200 --year 2018

    # JSON file mode
    python predict.py --json car_details.json

    # Interactive mode
    python predict.py --interactive
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
from datetime import datetime


class VehiclePricePredictor:
    """Vehicle price prediction class with preprocessing and model loading"""

    def __init__(self, model_path: str = "models/best_model.pkl",
                 preprocessor_path: str = "outputs/preprocessor.joblib"):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.model_name = None

        self._load_artifacts()

    def _load_artifacts(self):
        """Load model and preprocessor"""
        try:
            # Load model
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model file not found: {self.model_path}")

            model_pack = joblib.load(self.model_path)
            self.model = model_pack["model"]
            self.model_name = model_pack.get("algo", "Unknown")
            print(f"✅ Loaded model: {self.model_name}")

            # Load preprocessor
            if not os.path.exists(self.preprocessor_path):
                raise FileNotFoundError(
                    f"Preprocessor file not found: {self.preprocessor_path}")

            self.preprocessor = joblib.load(self.preprocessor_path)
            print(f"✅ Loaded preprocessor")

            # Try to get feature names from processed data
            try:
                data_path = "outputs/processed_data.pkl"
                if os.path.exists(data_path):
                    data_blob = joblib.load(data_path)
                    self.feature_names = data_blob.get("feature_names", [])
                    print(f"✅ Loaded {len(self.feature_names)} feature names")
            except Exception as e:
                print(f"⚠️ Could not load feature names: {e}")

        except Exception as e:
            print(f"❌ Error loading artifacts: {e}")
            sys.exit(1)

    def _validate_and_prepare_input(self, car_data: Dict[str, Any]) -> pd.DataFrame:
        """Validate and prepare input data for prediction"""

        # Define expected features and their defaults
        expected_features = {
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

        # Fill missing values with defaults
        processed_data = {}
        for feature, default in expected_features.items():
            if feature in car_data and car_data[feature] is not None:
                processed_data[feature] = car_data[feature]
            else:
                processed_data[feature] = default
                print(f"⚠️ Using default value for {feature}: {default}")

        # Calculate age if year is provided instead
        if 'year' in car_data and car_data['year'] is not None:
            current_year = datetime.now().year
            processed_data['age'] = current_year - int(car_data['year'])
            print(f"📅 Calculated age: {processed_data['age']} years")

        # Validate numeric ranges
        validations = {
            'km_driven': (0, 1000000),
            'engine_cc': (50, 8000),
            'max_power_bhp': (10, 1000),
            'seats': (2, 10),
            'age': (0, 50),
            'mileage_value': (5, 50)
        }

        for field, (min_val, max_val) in validations.items():
            if field in processed_data:
                value = float(processed_data[field])
                if not (min_val <= value <= max_val):
                    print(
                        f"⚠️ {field} value {value} is outside expected range [{min_val}, {max_val}]")

        # Convert to DataFrame
        df = pd.DataFrame([processed_data])

        return df

    def predict(self, car_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make price prediction for given car data"""

        try:
            # Prepare input data
            df = self._validate_and_prepare_input(car_data)

            # Get the expected feature columns for preprocessing
            numeric_features = ['km_driven', 'mileage_value', 'engine_cc', 'max_power_bhp',
                                'torque_nm', 'torque_rpm', 'seats', 'age']
            categorical_features = ['fuel', 'transmission', 'owner', 'seller_type',
                                    'mileage_unit', 'make']

            # Select only the features needed for preprocessing
            feature_data = df[numeric_features + categorical_features]

            # Apply preprocessing
            X_processed = self.preprocessor.transform(feature_data)

            # Make prediction
            prediction = self.model.predict(X_processed)[0]

            # Format result
            result = {
                "predicted_price": float(prediction),
                "formatted_price": f"₹{prediction:,.0f}",
                "model_used": self.model_name,
                "input_features": car_data,
                "prediction_timestamp": datetime.now().isoformat()
            }

            return result

        except Exception as e:
            return {
                "error": str(e),
                "predicted_price": None,
                "formatted_price": "Error in prediction",
                "prediction_timestamp": datetime.now().isoformat()
            }


def interactive_mode():
    """Interactive mode for getting car details"""
    print("\n🚗 Vehicle Price Prediction - Interactive Mode")
    print("=" * 50)

    car_data = {}

    # Get basic details
    print("\n📝 Enter car details (press Enter for default values):")

    # Make
    make_options = ["Maruti", "Toyota", "Hyundai", "Honda", "Tata", "Mahindra",
                    "Ford", "Chevrolet", "Volkswagen", "BMW", "Mercedes-Benz", "Audi"]
    print(f"Available makes: {', '.join(make_options)}")
    make = input("🏭 Car Make [Maruti]: ").strip()
    car_data['make'] = make if make else "Maruti"

    # Year
    year = input("📅 Manufacturing Year [2019]: ").strip()
    if year:
        try:
            car_data['year'] = int(year)
        except ValueError:
            print("⚠️ Invalid year, using default")
            car_data['year'] = 2019
    else:
        car_data['year'] = 2019

    # Fuel type
    fuel_options = ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"]
    print(f"Fuel types: {', '.join(fuel_options)}")
    fuel = input("⛽ Fuel Type [Petrol]: ").strip()
    car_data['fuel'] = fuel if fuel in fuel_options else "Petrol"

    # Transmission
    transmission = input(
        "⚙️ Transmission (Manual/Automatic) [Manual]: ").strip()
    car_data['transmission'] = transmission if transmission.lower() in [
        'manual', 'automatic'] else "Manual"

    # Engine CC
    engine_cc = input("🔧 Engine CC [1200]: ").strip()
    if engine_cc:
        try:
            car_data['engine_cc'] = int(engine_cc)
        except ValueError:
            car_data['engine_cc'] = 1200
    else:
        car_data['engine_cc'] = 1200

    # KM Driven
    km_driven = input("🛣️ Kilometers Driven [50000]: ").strip()
    if km_driven:
        try:
            car_data['km_driven'] = int(km_driven)
        except ValueError:
            car_data['km_driven'] = 50000
    else:
        car_data['km_driven'] = 50000

    # Max Power
    max_power = input("💪 Max Power (BHP) [80]: ").strip()
    if max_power:
        try:
            car_data['max_power_bhp'] = float(max_power)
        except ValueError:
            car_data['max_power_bhp'] = 80
    else:
        car_data['max_power_bhp'] = 80

    # Owner
    owner_options = ["First", "Second", "Third", "Fourth & Above"]
    print(f"Owner types: {', '.join(owner_options)}")
    owner = input("👤 Owner Type [First]: ").strip()
    car_data['owner'] = owner if owner in owner_options else "First"

    return car_data


def main():
    parser = argparse.ArgumentParser(description="Vehicle Price Prediction")
    parser.add_argument("--json", help="JSON file with car details")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode")

    # CLI arguments for direct input
    parser.add_argument("--make", help="Car manufacturer")
    parser.add_argument("--year", type=int, help="Manufacturing year")
    parser.add_argument("--fuel", help="Fuel type")
    parser.add_argument("--transmission", help="Transmission type")
    parser.add_argument("--engine_cc", type=int,
                        help="Engine displacement in CC")
    parser.add_argument("--km_driven", type=int, help="Kilometers driven")
    parser.add_argument("--max_power_bhp", type=float,
                        help="Maximum power in BHP")
    parser.add_argument("--owner", help="Owner type")
    parser.add_argument("--seller_type", help="Seller type")

    args = parser.parse_args()

    # Initialize predictor
    print("🔄 Loading model and preprocessor...")
    predictor = VehiclePricePredictor()

    # Get car data based on input method
    car_data = {}

    if args.json:
        # Load from JSON file
        try:
            with open(args.json, 'r') as f:
                car_data = json.load(f)
            print(f"📁 Loaded car data from {args.json}")
        except Exception as e:
            print(f"❌ Error loading JSON file: {e}")
            return

    elif args.interactive:
        # Interactive mode
        car_data = interactive_mode()

    else:
        # CLI arguments
        for arg_name in ['make', 'year', 'fuel', 'transmission', 'engine_cc',
                         'km_driven', 'max_power_bhp', 'owner', 'seller_type']:
            value = getattr(args, arg_name)
            if value is not None:
                car_data[arg_name] = value

        if not car_data:
            print("❌ No car data provided. Use --interactive, --json, or CLI arguments.")
            parser.print_help()
            return

    # Make prediction
    print("\n🔮 Making prediction...")
    result = predictor.predict(car_data)

    # Display results
    print("\n" + "=" * 60)
    print("🎯 PRICE PREDICTION RESULT")
    print("=" * 60)

    if result.get("error"):
        print(f"❌ Error: {result['error']}")
    else:
        print(f"🚗 Car Details:")
        for key, value in result['input_features'].items():
            print(f"  {key}: {value}")

        print(f"\n💰 Predicted Price: {result['formatted_price']}")
        print(f"🤖 Model Used: {result['model_used']}")
        print(f"📅 Prediction Time: {result['prediction_timestamp']}")

        # Price interpretation
        price = result['predicted_price']
        if price < 300000:
            category = "Budget car 🚗"
        elif price < 800000:
            category = "Mid-range car 🚙"
        elif price < 1500000:
            category = "Premium car 🚘"
        else:
            category = "Luxury car 🏎️"

        print(f"📊 Category: {category}")

    print("=" * 60)


if __name__ == "__main__":
    main()
