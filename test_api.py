import requests
import json

# Base URL for the API
BASE_URL = "http://localhost:8000"

def test_single_prediction():
    """Test single prediction endpoint"""
    print("Testing single prediction...")
    
    # Earth-like sample
    sample_data = {
        "period": 365.25,
        "prad": 1.0,
        "teq": 288.0,
        "srad": 1.0,
        "slog_g": 4.44,
        "steff": 5778,
        "impact": 0.0,
        "duration": 13.0,
        "depth": 84.0
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=sample_data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Single prediction successful!")
        print(f"Random Forest: {result['random_forest']['prediction']}")
        print(f"Decision Tree: {result['decision_tree']['prediction']}")
        print(f"KNN: {result['knn']['prediction']}")
        print(f"CNN: {result['cnn']['prediction']}")
        print(f"ESI: {result['esi']:.4f}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\nTesting batch prediction...")
    
    batch_data = {
        "earth_like": {
            "period": 365.25,
            "prad": 1.0,
            "teq": 288.0,
            "srad": 1.0,
            "slog_g": 4.44,
            "steff": 5778,
            "impact": 0.0,
            "duration": 13.0,
            "depth": 84.0
        },
        "hot_jupiter": {
            "period": 3.5,
            "prad": 1.2,
            "teq": 1200.0,
            "srad": 1.1,
            "slog_g": 4.3,
            "steff": 6000,
            "impact": 0.2,
            "duration": 2.5,
            "depth": 5000.0
        }
    }
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data)
    
    if response.status_code == 200:
        results = response.json()
        print("✅ Batch prediction successful!")
        for sample_name, result in results.items():
            print(f"\n{sample_name.upper()}:")
            for model, prediction in result['predictions'].items():
                print(f"  {model}: {prediction['prediction']}")
            print(f"  ESI: {result['esi']:.4f}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_health_check():
    """Test health check endpoint"""
    print("\nTesting health check...")
    
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Health check: {result['status']} - {result['message']}")
    else:
        print(f"❌ Health check failed: {response.status_code}")

def test_models_info():
    """Test models info endpoint"""
    print("\nTesting models info...")
    
    response = requests.get(f"{BASE_URL}/models/info")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Models info retrieved successfully!")
        print(f"Available models: {len(result['available_models'])}")
        for model in result['available_models']:
            print(f"  - {model['name']} ({model['type']})")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_with_notebook_samples():
    """Test with the samples from your notebook"""
    print("\nTesting with notebook samples...")
    
    # Sample data from your notebook
    samples = {
        "earth_like": {
            "period": 365.25,
            "prad": 1.0,
            "teq": 288.0,
            "srad": 1.0,
            "slog_g": 4.44,
            "steff": 5778,
            "impact": 0.0,
            "duration": 13.0,
            "depth": 84.0
        },
        "hot_jupiter": {
            "period": 3.5,
            "prad": 1.2,
            "teq": 1200.0,
            "srad": 1.1,
            "slog_g": 4.3,
            "steff": 6000,
            "impact": 0.2,
            "duration": 2.5,
            "depth": 5000.0
        },
        "super_earth": {
            "period": 50.0,
            "prad": 1.8,
            "teq": 350.0,
            "srad": 0.9,
            "slog_g": 4.5,
            "steff": 5200,
            "impact": 0.3,
            "duration": 8.0,
            "depth": 250.0
        }
    }
    
    for sample_name, sample_data in samples.items():
        print(f"\n--- Testing {sample_name.replace('_', ' ').title()} ---")
        response = requests.post(f"{BASE_URL}/predict", json=sample_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful!")
            
            # Display results in a clean format
            for model_name in ['random_forest', 'decision_tree', 'knn', 'cnn']:
                prediction_data = result[model_name]['prediction']
                if isinstance(prediction_data, dict):
                    pred_result = "Exoplanet" if prediction_data.get('prediction', 0) == 1 else "Not Exoplanet"
                    confidence = prediction_data.get('probability', {}).get('exoplanet', 0) * 100
                    print(f"  {model_name.replace('_', ' ').title()}: {pred_result} (Confidence: {confidence:.1f}%)")
                else:
                    print(f"  {model_name.replace('_', ' ').title()}: {prediction_data}")
            
            print(f"  ESI: {result['esi']:.4f}")
        else:
            print(f"❌ Error for {sample_name}: {response.status_code} - {response.text}")

if __name__ == "__main__":
    print("🚀 Testing ExoBengal API")
    print("=" * 50)
    
    try:
        # Test health check first
        test_health_check()
        
        # Test models info
        test_models_info()
        
        # Test single prediction
        test_single_prediction()
        
        # Test batch prediction
        test_batch_prediction()
        
        # Test with notebook samples
        test_with_notebook_samples()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Make sure the API server is running on http://localhost:8000")
        print("   To start the server, run: python main.py")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
