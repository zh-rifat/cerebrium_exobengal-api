# ExoBengal API - Cerebrium Deployment

## Overview

ExoBengal API is a machine learning-powered exoplanet detection service deployed on Cerebrium. It uses multiple ML models (Random Forest, Decision Tree, KNN, and CNN) to classify celestial objects and calculate their Earth Similarity Index (ESI).

**Live API Endpoint:** `https://api.aws.us-east-1.cerebrium.ai/v4/p-e08fc93f/exobengal-api/`

---

## Table of Contents

- [Features](#features)
- [Available Models](#available-models)
- [API Endpoints](#api-endpoints)
- [Usage Examples](#usage-examples)
- [Input Parameters](#input-parameters)
- [Response Format](#response-format)
- [Error Handling](#error-handling)
- [Deployment Configuration](#deployment-configuration)

---

## Features

- **Multiple ML Models**: Random Forest, Decision Tree, KNN, and CNN
- **Earth Similarity Index**: Automatic ESI calculation
- **Auto-scaling**: Scales from 0-3 replicas based on demand
- **Fast Response**: 10-second cooldown for quick scaling
- **Error Resilience**: Continues with other models if one fails
- **Model Selection**: Choose specific models or run all

---

## Available Models

| Model | Type | Description |
|-------|------|-------------|
| **Random Forest** | Ensemble Learning | Random Forest Classifier for robust predictions |
| **Decision Tree** | Tree-based Learning | Decision Tree for interpretable results |
| **K-Nearest Neighbors** | Instance-based Learning | KNN with ESI calculation |
| **Convolutional Neural Network** | Deep Learning | CNN for complex pattern recognition |

---

## API Endpoints

### Base URL
```
https://api.aws.us-east-1.cerebrium.ai/v4/p-e08fc93f/exobengal-api/
```

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Main prediction endpoint |
| `/health_check` | POST | Health status check |
| `/get_model_info` | POST | Get model information |
| `/init` | POST | Initialize models manually |

---

## Usage Examples

### 1. Health Check

```bash
curl -X POST https://api.aws.us-east-1.cerebrium.ai/v4/p-e08fc93f/exobengal-api/health_check \
  -H "Content-Type: application/json" \
  -H "Accept: application/json"
```

**Response:**
```json
{
  "status": "healthy",
  "message": "ExoBengal API is running successfully"
}
```

### 2. Get Model Information

```bash
curl -X POST https://api.aws.us-east-1.cerebrium.ai/v4/p-e08fc93f/exobengal-api/get_model_info \
  -H "Content-Type: application/json" \
  -H "Accept: application/json"
```

### 3. Predict with All Models

```bash
curl -X POST https://api.aws.us-east-1.cerebrium.ai/v4/p-e08fc93f/exobengal-api/predict \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "item": {
      "period": 365.0,
      "prad": 1.0,
      "teq": 288.0,
      "srad": 1.0,
      "slog_g": 4.44,
      "steff": 5778.0,
      "impact": 0.1,
      "duration": 5.0,
      "depth": 100.0
    }
  }'
```

### 4. Predict with Specific Models

```bash
curl -X POST https://api.aws.us-east-1.cerebrium.ai/v4/p-e08fc93f/exobengal-api/predict \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "item": {
      "period": 365.0,
      "prad": 1.0,
      "teq": 288.0,
      "srad": 1.0,
      "slog_g": 4.44,
      "steff": 5778.0,
      "impact": 0.1,
      "duration": 5.0,
      "depth": 100.0,
      "models": ["random_forest", "cnn"]
    }
  }'
```

### 5. Python Request Examples

```python
import requests
import json

# API Configuration
BASE_URL = "https://api.aws.us-east-1.cerebrium.ai/v4/p-e08fc93f/exobengal-api"
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json"
}

# Health Check
def check_health():
    response = requests.post(f"{BASE_URL}/health_check", headers=HEADERS)
    return response.json()

# Get Model Information
def get_model_info():
    response = requests.post(f"{BASE_URL}/get_model_info", headers=HEADERS)
    return response.json()

# Predict with All Models
def predict_all_models():
    data = {
        "item": {
            "period": 365.0,
            "prad": 1.0,
            "teq": 288.0,
            "srad": 1.0,
            "slog_g": 4.44,
            "steff": 5778.0,
            "impact": 0.1,
            "duration": 5.0,
            "depth": 100.0
        }
    }
    response = requests.post(f"{BASE_URL}/predict", headers=HEADERS, json=data)
    return response.json()

# Predict with Specific Models
def predict_specific_models():
    data = {
        "item": {
            "period": 365.0,
            "prad": 1.0,
            "teq": 288.0,
            "srad": 1.0,
            "slog_g": 4.44,
            "steff": 5778.0,
            "impact": 0.1,
            "duration": 5.0,
            "depth": 100.0,
            "models": ["random_forest", "cnn"]
        }
    }
    response = requests.post(f"{BASE_URL}/predict", headers=HEADERS, json=data)
    return response.json()

# Example usage
if __name__ == "__main__":
    # Check API health
    health = check_health()
    print("Health Check:", health)
    
    # Get model information
    models = get_model_info()
    print("Available Models:", models)
    
    # Make prediction
    result = predict_all_models()
    print("Prediction Result:", result)
```

### 6. JavaScript Request Examples

```javascript
// API Configuration
const BASE_URL = "https://api.aws.us-east-1.cerebrium.ai/v4/p-e08fc93f/exobengal-api";
const HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json"
};

// Health Check
async function checkHealth() {
    try {
        const response = await fetch(`${BASE_URL}/health_check`, {
            method: 'POST',
            headers: HEADERS
        });
        return await response.json();
    } catch (error) {
        console.error('Health check failed:', error);
        return null;
    }
}

// Get Model Information
async function getModelInfo() {
    try {
        const response = await fetch(`${BASE_URL}/get_model_info`, {
            method: 'POST',
            headers: HEADERS
        });
        return await response.json();
    } catch (error) {
        console.error('Model info request failed:', error);
        return null;
    }
}

// Predict with All Models
async function predictAllModels() {
    const data = {
        item: {
            period: 365.0,
            prad: 1.0,
            teq: 288.0,
            srad: 1.0,
            slog_g: 4.44,
            steff: 5778.0,
            impact: 0.1,
            duration: 5.0,
            depth: 100.0
        }
    };
    
    try {
        const response = await fetch(`${BASE_URL}/predict`, {
            method: 'POST',
            headers: HEADERS,
            body: JSON.stringify(data)
        });
        return await response.json();
    } catch (error) {
        console.error('Prediction request failed:', error);
        return null;
    }
}

// Predict with Specific Models
async function predictSpecificModels() {
    const data = {
        item: {
            period: 365.0,
            prad: 1.0,
            teq: 288.0,
            srad: 1.0,
            slog_g: 4.44,
            steff: 5778.0,
            impact: 0.1,
            duration: 5.0,
            depth: 100.0,
            models: ["random_forest", "cnn"]
        }
    };
    
    try {
        const response = await fetch(`${BASE_URL}/predict`, {
            method: 'POST',
            headers: HEADERS,
            body: JSON.stringify(data)
        });
        return await response.json();
    } catch (error) {
        console.error('Specific models prediction failed:', error);
        return null;
    }
}

// Example usage
async function runExamples() {
    // Check API health
    const health = await checkHealth();
    console.log('Health Check:', health);
    
    // Get model information
    const models = await getModelInfo();
    console.log('Available Models:', models);
    
    // Make prediction
    const result = await predictAllModels();
    console.log('Prediction Result:', result);
}

// For Node.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        checkHealth,
        getModelInfo,
        predictAllModels,
        predictSpecificModels
    };
}

// For browser - auto-run examples
// runExamples();
```

### 7. Hot Jupiter Detection Example

```python
# Python - Hot Jupiter scenario
def detect_hot_jupiter():
    data = {
        "item": {
            "period": 3.5,        # Short orbital period
            "prad": 1.2,          # Jupiter-like radius
            "teq": 1200.0,        # Very hot temperature
            "srad": 1.1,          # Slightly larger star
            "slog_g": 4.3,        # Stellar surface gravity
            "steff": 6000.0,      # Hot star temperature
            "impact": 0.2,        # Impact parameter
            "duration": 2.5,      # Short transit duration
            "depth": 5000.0,      # Deep transit (large planet)
            "models": ["random_forest", "cnn"]
        }
    }
    response = requests.post(f"{BASE_URL}/predict", headers=HEADERS, json=data)
    return response.json()
```

```javascript
// JavaScript - Hot Jupiter scenario
async function detectHotJupiter() {
    const data = {
        item: {
            period: 3.5,        // Short orbital period
            prad: 1.2,          // Jupiter-like radius
            teq: 1200.0,        // Very hot temperature
            srad: 1.1,          // Slightly larger star
            slog_g: 4.3,        // Stellar surface gravity
            steff: 6000.0,      // Hot star temperature
            impact: 0.2,        // Impact parameter
            duration: 2.5,      // Short transit duration
            depth: 5000.0,      // Deep transit (large planet)
            models: ["random_forest", "cnn"]
        }
    };
    
    const response = await fetch(`${BASE_URL}/predict`, {
        method: 'POST',
        headers: HEADERS,
        body: JSON.stringify(data)
    });
    return await response.json();
}
```


---

## Input Parameters

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `period` | float | Required | Orbital period (days) | 365.0 |
| `prad` | float | Required | Planet radius (Earth radii) | 1.0 |
| `teq` | float | Required | Equilibrium temperature (Kelvin) | 288.0 |
| `srad` | float | Required | Stellar radius (solar radii) | 1.0 |
| `slog_g` | float | Required | Stellar surface gravity (log scale) | 4.44 |
| `steff` | float | Required | Stellar effective temperature (Kelvin) | 5778.0 |
| `impact` | float | Required | Impact parameter | 0.1 |
| `duration` | float | Required | Transit duration (hours) | 5.0 |
| `depth` | float | Required | Transit depth (ppm) | 100.0 |
| `models` | array | Optional | Specific models to run | ["random_forest", "cnn"] |

### Available Model Options:
- `"random_forest"`
- `"decision_tree"`
- `"knn"`
- `"cnn"`

---

## Response Format

### Successful Prediction Response

```json
{
  "run_id": "c514a61a-34f4-9cb7-bba8-a11679c5e2d5",
  "result": {
    "predictions": {
      "random_forest": {
        "prediction": {
          "prediction": "Not a Planet",
          "probability": 0.452
        },
        "model_type": "Random Forest"
      },
      "decision_tree": {
        "prediction": {
          "prediction": "Not a Planet",
          "probability": 0
        },
        "model_type": "Decision Tree"
      },
      "knn": {
        "prediction": {
          "prediction": "Planet",
          "probability": 1,
          "ESI": 0.021
        },
        "model_type": "Knn"
      },
      "cnn": {
        "prediction": {
          "prediction": "Not a Planet",
          "probability": 0.5783929228782654
        },
        "model_type": "Cnn"
      }
    },
    "esi": 1,
    "input_data": {
      "period": 365,
      "prad": 1,
      "teq": 288,
      "srad": 1,
      "slog_g": 4.44,
      "steff": 5778,
      "impact": 0.1,
      "duration": 5,
      "depth": 100
    },
    "models_executed": [
      "random_forest",
      "decision_tree",
      "knn",
      "cnn"
    ],
    "status": "success"
  },
  "run_time_ms": 7866.67275428772
}
```

### Earth Similarity Index (ESI)
The ESI is calculated automatically for all predictions:
- **Range**: 0.0 to 1.0
- **1.0**: Perfect Earth-like conditions
- **0.0**: Completely unlike Earth
- **Formula**: Based on planet radius and equilibrium temperature

---

## Error Handling

### Missing Parameters Error
```json
{
  "error": "Missing required fields: ['period', 'prad']",
  "status": "error"
}
```

### Invalid Model Error
```json
{
  "error": "Invalid model(s): ['invalid_model']. Available models: ['random_forest', 'decision_tree', 'knn', 'cnn']",
  "status": "error"
}
```

### Model Execution Error
```json
{
  "predictions": {
    "random_forest": {
      "error": "Model execution failed: [error details]",
      "model_type": "Random Forest"
    }
  },
  "status": "success"
}
```

### Initialization Error
```json
{
  "status": "error",
  "message": "Initialization failed: [error details]"
}
```

---

## Deployment Configuration

### Hardware Specifications
- **CPU**: 2.0 cores
- **Memory**: 4.0 GB
- **Compute**: CPU (AWS)
- **Python**: 3.10
- **Base Image**: python:3.10-bookworm

### Auto-scaling Settings
- **Min Replicas**: 0 (scales to zero when idle)
- **Max Replicas**: 3
- **Cooldown**: 10 seconds
- **Concurrency**: 1 request per replica
- **Metric**: Concurrency utilization

### Model Files
- Random Forest Classifier (`.pkl`)
- Decision Tree Classifier (`.pkl`)
- CNN Model (`.h5`)
- KNN Model (`.pkl`)
- Feature Scaler (`.pkl`)
- Data Imputer (`.pkl`)

---

## Development

### Local Testing
```python
# Test the predict function locally
from main import predict, health_check, get_model_info

# Test health check
health = health_check()
print(health)

# Test model info
info = get_model_info()
print(info)

# Test prediction
test_data = {
    "period": 365.0,
    "prad": 1.0,
    "teq": 288.0,
    "srad": 1.0,
    "slog_g": 4.44,
    "steff": 5778.0,
    "impact": 0.1,
    "duration": 5.0,
    "depth": 100.0
}

result = predict(test_data)
print(result)
```

### Redeployment
```bash
# Deploy to Cerebrium
cerebrium deploy

# Check deployment status
cerebrium app list

# View logs
cerebrium logs exobengal-api
```

### File Structure
```
cerebrium_exobengal-api/  # Root deployment directory
├── main.py               # Cerebrium function entrypoints
├── cerebrium.toml        # Cerebrium configuration
├── requirements.txt      # Dependencies
├── exobengal-api/        # Original FastAPI code
│   └── app.py            # FastAPI application (backup)
└── models/               # ML models directory
    ├── random_forest_classifier.pkl
    ├── decision_tree_classifier.pkl
    ├── cnn_model.h5
    ├── knn_model.pkl
    ├── scaler.pkl
    └── imputer.pkl
```

---

## Performance

- **Cold Start**: ~10-15 seconds (model loading)
- **Warm Response**: ~100-500ms
- **Scaling**: Automatic based on traffic
- **Availability**: 99.9% uptime on Cerebrium
- **Throughput**: Up to 3 concurrent requests (max replicas)

---

## Example Scenarios

### 1. Planet Detection
```bash
# Characteristics similar to Earth
curl -X POST https://api.aws.us-east-1.cerebrium.ai/v4/p-e08fc93f/exobengal-api/predict \
  -H "Content-Type: application/json" \
  -d '{"item": {"period": 365.0, "prad": 1.0, "teq": 288.0, "srad": 1.0, "slog_g": 4.44, "steff": 5778.0, "impact": 0.1, "duration": 5.0, "depth": 100.0}}'
```


### Common Issues
1. **Missing Parameters**: Ensure all required fields are included
2. **Invalid Models**: Check model names in the `models` array
3. **Cold Start Delays**: First request after idle period takes longer
4. **Type Errors**: Ensure numeric values are properly formatted

---

## Use Cases

1. **Exoplanet Research**: Classify newly discovered celestial objects
2. **Educational Tools**: Demonstrate ML in astronomy
3. **Space Missions**: Quick assessment of potential targets
4. **Comparative Analysis**: Test different detection algorithms
5. **Habitability Studies**: Use ESI calculations for astrobiology

---

## Additional Resources

- **ExoBengal Package**: [https://pypi.org/project/exobengal/](https://pypi.org/project/exobengal/)
- **Cerebrium Documentation**: [https://docs.cerebrium.ai](https://docs.cerebrium.ai)
- **NASA Exoplanet Archive**: Source of training data
- **Kepler Space Telescope**: Data source for model training

---

**Start exploring the universe with ExoBengal API!**

*Deployed with love on Cerebrium Cloud Platform*
