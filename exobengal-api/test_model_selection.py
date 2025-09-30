import requests
import json

# Base URL for the API
BASE_URL = "http://localhost:8000"

def test_specific_models():
    """Test prediction with specific models only"""
    print("🔬 Testing specific model selection...")
    
    # Earth-like sample with only Random Forest and CNN
    sample_data = {
        "period": 365.25,
        "prad": 1.0,
        "teq": 288.0,
        "srad": 1.0,
        "slog_g": 4.44,
        "steff": 5778,
        "impact": 0.0,
        "duration": 13.0,
        "depth": 84.0,
        "models": ["random_forest", "cnn"]  # Only run these two models
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=sample_data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Specific model prediction successful!")
        print(f"Models executed: {result['models_executed']}")
        
        # Only show results for requested models
        if result['random_forest']:
            print(f"Random Forest: {result['random_forest']['prediction']}")
        if result['cnn']:
            print(f"CNN: {result['cnn']['prediction']}")
        if result['decision_tree']:
            print("Decision Tree: Not requested (should be None)")
        if result['knn']:
            print("KNN: Not requested (should be None)")
            
        print(f"ESI: {result['esi']:.4f}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_single_model():
    """Test prediction with only one model"""
    print("\n🎯 Testing single model (CNN only)...")
    
    sample_data = {
        "period": 3.5,
        "prad": 1.2,
        "teq": 1200.0,
        "srad": 1.1,
        "slog_g": 4.3,
        "steff": 6000,
        "impact": 0.2,
        "duration": 2.5,
        "depth": 5000.0,
        "models": ["cnn"]  # Only CNN
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=sample_data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Single model prediction successful!")
        print(f"Models executed: {result['models_executed']}")
        print(f"CNN: {result['cnn']['prediction']}")
        print(f"ESI: {result['esi']:.4f}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_all_models():
    """Test prediction with all models (default behavior)"""
    print("\n🌟 Testing all models (default)...")
    
    sample_data = {
        "period": 50.0,
        "prad": 1.8,
        "teq": 350.0,
        "srad": 0.9,
        "slog_g": 4.5,
        "steff": 5200,
        "impact": 0.3,
        "duration": 8.0,
        "depth": 250.0
        # No "models" field - should run all models
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=sample_data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ All models prediction successful!")
        print(f"Models executed: {result['models_executed']}")
        
        for model in ['random_forest', 'decision_tree', 'knn', 'cnn']:
            if result[model]:
                print(f"{model.replace('_', ' ').title()}: {result[model]['prediction']}")
        
        print(f"ESI: {result['esi']:.4f}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_invalid_model():
    """Test prediction with invalid model name"""
    print("\n❌ Testing invalid model name...")
    
    sample_data = {
        "period": 365.0,
        "prad": 1.0,
        "teq": 288.0,
        "srad": 1.0,
        "slog_g": 4.44,
        "steff": 5778,
        "impact": 0.1,
        "duration": 5.0,
        "depth": 100.0,
        "models": ["random_forest", "invalid_model", "cnn"]  # Invalid model name
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=sample_data)
    
    if response.status_code == 400:
        print("✅ Invalid model correctly rejected!")
        print(f"Error message: {response.json()['detail']}")
    else:
        print(f"❌ Unexpected response: {response.status_code} - {response.text}")

def test_batch_with_models():
    """Test batch prediction with different models for each sample"""
    print("\n📦 Testing batch prediction with model selection...")
    
    batch_data = {
        "earth_like_rf_only": {
            "period": 365.25,
            "prad": 1.0,
            "teq": 288.0,
            "srad": 1.0,
            "slog_g": 4.44,
            "steff": 5778,
            "impact": 0.0,
            "duration": 13.0,
            "depth": 84.0,
            "models": ["random_forest"]  # Only Random Forest
        },
        "hot_jupiter_cnn_knn": {
            "period": 3.5,
            "prad": 1.2,
            "teq": 1200.0,
            "srad": 1.1,
            "slog_g": 4.3,
            "steff": 6000,
            "impact": 0.2,
            "duration": 2.5,
            "depth": 5000.0,
            "models": ["cnn", "knn"]  # Only CNN and KNN
        }
    }
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data)
    
    if response.status_code == 200:
        results = response.json()
        print("✅ Batch prediction with model selection successful!")
        
        for sample_name, result in results.items():
            print(f"\n{sample_name.upper()}:")
            print(f"  Models executed: {result['models_executed']}")
            for model, prediction in result['predictions'].items():
                if 'prediction' in prediction:
                    print(f"  {model}: {prediction['prediction']}")
            print(f"  ESI: {result['esi']:.4f}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    print("🚀 Testing ExoBengal API Model Selection Feature")
    print("=" * 60)
    
    try:
        # Test various scenarios
        test_specific_models()
        test_single_model()
        test_all_models()
        test_invalid_model()
        test_batch_with_models()
        
        print("\n" + "=" * 60)
        print("✅ All model selection tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Make sure the API server is running on http://localhost:8000")
        print("   To start the server, run: python main.py")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
