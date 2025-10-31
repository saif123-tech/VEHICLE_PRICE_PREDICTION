#!/usr/bin/env python3
"""
Test script for Vehicle Price Prediction API
"""
import requests
import json

# Test data
test_car = {
    "make": "Toyota",
    "year": 2018,
    "fuel": "Petrol",
    "transmission": "Manual",
    "engine_cc": 1200,
    "km_driven": 50000,
    "max_power_bhp": 85.0,
    "owner": "First"
}


def test_api():
    print("🧪 Testing Vehicle Price Prediction API")
    print("=" * 50)

    base_url = "http://127.0.0.1:8000"

    # Test health endpoint
    try:
        print("1. Testing health endpoint...")
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()['message']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Could not connect to API: {e}")
        print("   Make sure the API is running: uvicorn api_app:app --host 127.0.0.1 --port 8000")
        return

    # Test prediction endpoint
    try:
        print("\n2. Testing prediction endpoint...")
        response = requests.post(
            f"{base_url}/predict",
            json=test_car,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful")
            print(f"   Car: {test_car['make']} {test_car['year']}")
            print(f"   Predicted Price: {result['formatted_price']}")
            print(f"   Category: {result['price_category']}")
            print(f"   Model: {result['model_used']}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Prediction request failed: {e}")

    # Test model info endpoint
    try:
        print("\n3. Testing model info endpoint...")
        response = requests.get(f"{base_url}/model-info")
        if response.status_code == 200:
            info = response.json()
            print("✅ Model info retrieved")
            print(f"   Model: {info['model_name']}")
            print(f"   Features: {info['feature_count']}")
            print(f"   Status: {info['status']}")
        else:
            print(f"❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Model info request failed: {e}")

    print("\n🎉 API testing completed!")


if __name__ == "__main__":
    test_api()
