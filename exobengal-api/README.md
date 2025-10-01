# ExoBengal API
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A FastAPI-based REST API for exoplanet detection and analysis using machine learning models. This API provides endpoints to predict exoplanet characteristics using various ML algorithms including Random Forest, Decision Tree, K-Nearest Neighbors, and Convolutional Neural Networks.

## Features
- **Multiple ML Models**: Support for Random Forest, Decision Tree, KNN, and CNN models
- **Earth Similarity Index (ESI)**: Calculate how similar an exoplanet is to Earth
- **Batch Processing**: Analyze multiple exoplanets in a single request
- **Model Selection**: Choose specific models to run or execute all available models
- **RESTful API**: Clean, documented endpoints with automatic OpenAPI documentation

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/gazi-faysal-jubayer/ExoBengal
cd ExoBengal/exobengal-api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure model files are present in the `../models/` directory:
- `random_forest_classifier.pkl`
- `decision_tree_classifier.pkl`
- `knn_model.pkl`
- `cnn_model.h5`
- `scaler.pkl`
- `imputer.pkl`

4. Run the API:
```bash
python app.py
```

The API will be available at `http://localhost:8000`

## API Documentation

### Interactive Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
##

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Root endpoint with API information |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/docs` | Interactive API documentation (Swagger UI) |
| `GET` | `/models/info` | Information about available models |
| `POST` | `/predict` | Single exoplanet prediction |
| `POST` | `/predict/batch` | Batch predictions |

#

### Endpoint Details
#### `GET /`
Welcome endpoint with basic API information.

#### `GET /health`
Health check endpoint to verify API status.

**Response:**
```json
{
  "status": "healthy",
  "message": "ExoBengal API is running successfully"
}
```

## `POST /predict`
Predict exoplanet characteristics using specified or all available models.

**Request Body:**
```json
{
  "period": 365.0,
  "prad": 1.0,
  "teq": 288.0,
  "srad": 1.0,
  "slog_g": 4.44,
  "steff": 5778,
  "impact": 0.1,
  "duration": 5.0,
  "depth": 100.0,
  "models": ["random_forest", "cnn"]
}
```

**Input Parameters**

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `period` | float | Orbital period (days) | 365.25 |
| `prad` | float | Planet radius (Earth radii) | 1.0 |
| `teq` | float | Equilibrium temperature (Kelvin) | 288.0 |
| `srad` | float | Stellar radius (solar radii) | 1.0 |
| `slog_g` | float | Stellar surface gravity (log scale) | 4.44 |
| `steff` | float | Stellar effective temperature (Kelvin) | 5778 |
| `impact` | float | Impact parameter | 0.0 |
| `duration` | float | Transit duration (hours) | 13.0 |
| `depth` | float | Transit depth (parts per million) | 84.0 |
| `models` | array | Optional: specific models to run | ["random_forest", "cnn", "knn", "decision_tree"] |
##
**Response:**
```json
{
  "random_forest": {
    "prediction": {
      "prediction": "Not a Planet",
      "probability": 0.452
    },
    "model_type": "Random Forest"
  },
  "decision_tree": null,
  "knn": null,
  "cnn": {
    "prediction": {
      "prediction": "Not a Planet",
      "probability": 0.5783929228782654
    },
    "model_type": "Cnn"
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
    "depth": 100,
    "models": [
      "random_forest",
      "cnn"
    ]
  },
  "models_executed": [
    "random_forest",
    "cnn"
  ]
}
```

#### `POST /predict/batch`
Predict multiple exoplanet samples in a single request.

**Request Body:**
```json
{
  "sample1": {
    "period": 365.0,
    "prad": 1.0,
    "teq": 288.0,
    "srad": 1.0,
    "slog_g": 4.44,
    "steff": 5778,
    "impact": 0.1,
    "duration": 5.0,
    "depth": 100.0
  },
  "sample2": {
    "period": 687.0,
    "prad": 0.53,
    "teq": 210.0,
    "srad": 1.0,
    "slog_g": 4.44,
    "steff": 5778,
    "impact": 0.2,
    "duration": 6.0,
    "depth": 150.0,
    "models": ["random_forest"]
  }
}
```

**Response:**
Returns the same structure as `/predict` but with results grouped by sample name:
```json
{
  "sample1": { /* same as /predict response */ },
  "sample2": { /* same as /predict response */ }
}
```

#### `GET /models/info`
Get information about available models and input parameters.

**Response:**
```json
{
  "available_models": [
    {
      "name": "Random Forest",
      "type": "Ensemble Learning",
      "description": "Random Forest Classifier for exoplanet detection"
    }
  ],
  "input_features": [
    "period: Orbital period (days)",
    "prad: Planet radius (Earth radii)"
  ],
  "additional_calculations": [
    "ESI: Earth Similarity Index"
  ]
}
```

## Usage Examples

### Python Example

```python
import requests

# Single prediction
url = "http://localhost:8000/predict"
data = {
    "period": 365.0,
    "prad": 1.0,
    "teq": 288.0,
    "srad": 1.0,
    "slog_g": 4.44,
    "steff": 5778,
    "impact": 0.1,
    "duration": 5.0,
    "depth": 100.0,
    "models": ["random_forest", "cnn"]
}

response = requests.post(url, json=data)
result = response.json()
print(f"ESI: {result['esi']}")
print(f"Random Forest Prediction: {result['random_forest']['prediction']}")
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "period": 365.0,
       "prad": 1.0,
       "teq": 288.0,
       "srad": 1.0,
       "slog_g": 4.44,
       "steff": 5778,
       "impact": 0.1,
       "duration": 5.0,
       "depth": 100.0
     }'
```

### JavaScript Example

```javascript
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    period: 365.0,
    prad: 1.0,
    teq: 288.0,
    srad: 1.0,
    slog_g: 4.44,
    steff: 5778,
    impact: 0.1,
    duration: 5.0,
    depth: 100.0,
    models: ["random_forest"]
  })
});

const data = await response.json();
console.log('Prediction:', data);
```

## Model Information

### Available Models

1. **Random Forest**: Ensemble learning method using multiple decision trees
2. **Decision Tree**: Tree-based learning algorithm for classification
3. **K-Nearest Neighbors (KNN)**: Instance-based learning algorithm
4. **Convolutional Neural Network (CNN)**: Deep learning model for pattern recognition

### Earth Similarity Index (ESI)

The ESI is calculated based on planet radius and equilibrium temperature, providing a measure of how similar an exoplanet is to Earth (scale: 0-1, where 1 is Earth-like).

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid input parameters)
- `422`: Validation Error (missing required fields)
- `500`: Internal Server Error (model execution failure)
