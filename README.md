# ExoBengal API 🌌

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A powerful REST API for exoplanet detection and classification using multiple machine learning models. ExoBengal provides predictions using Random Forest, Decision Tree, K-Nearest Neighbors, and Convolutional Neural Network models.

## 🚀 Features

- **Multiple ML Models**: Random Forest, Decision Tree, KNN, and CNN
- **Selective Model Execution**: Run specific models or all models
- **Earth Similarity Index (ESI)**: Calculate habitability scores
- **Batch Processing**: Process multiple samples simultaneously
- **Interactive Documentation**: Auto-generated Swagger UI
- **Model Information**: Get details about available models and features

## 📁 Project Structure

```
fastapi_exobengal-api/                 # Root directory
├── .git/                              # Git repository data
├── exobengal-api/                     # API application directory
│   ├── app.py                        # FastAPI application
│   ├── requirements.txt               # Python dependencies
│   ├── start_api.sh                   # API startup script
│   ├── test_api.py                    # API test script
│   └── test_model_selection.py        # Model selection test script
├── models/                            # Pre-trained models
│   ├── random_forest_classifier.pkl  # Random Forest model
│   ├── decision_tree_classifier.pkl  # Decision Tree model
│   ├── cnn_model.h5                  # CNN model
│   ├── knn_model.pkl                 # KNN model
│   ├── scaler.pkl                    # Feature scaler
│   └── imputer.pkl                   # Missing value imputer
├── data/                              # Dataset files
│   ├── cumulative_2025.09.20_12.15.37.csv        # Cumulative dataset
│   └── q1_q17_dr24_koi_2025.09.21_22.02.00.csv   # KOI dataset
├── notebooks/                         # Jupyter notebooks
│   ├── z.ipynb                       # Model testing notebook
│   └── zz.ipynb                      # Additional testing notebook
└── README.md                         # Documentation
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Anaconda/Miniconda (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/zh-rifat/fastapi_exobengal-api
   cd exobengal-api
   ```

2. **Create and activate conda environment**
   ```bash
   conda create -n exobengal python=3.10
   conda activate exobengal
   ```

3. **Install dependencies**
   ```bash
   cd exobengal-api
   pip install -r requirements.txt
   ```

4. **Run the API**
   ```bash
   # From the exobengal-api directory
   python main.py
   
   # Or use the startup script
   chmod +x start_api.sh
   ./start_api.sh
   ```

The API will be available at `http://localhost:8000`

## 📖 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Root endpoint with API information |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/docs` | Interactive API documentation (Swagger UI) |
| `GET` | `/models/info` | Information about available models |
| `POST` | `/predict` | Single exoplanet prediction |
| `POST` | `/predict/batch` | Batch predictions |

### Input Parameters

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

## 🔬 Usage Examples

### Single Prediction (All Models)

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "period": 365.25,
  "prad": 1.0,
  "teq": 288.0,
  "srad": 1.0,
  "slog_g": 4.44,
  "steff": 5778,
  "impact": 0.0,
  "duration": 13.0,
  "depth": 84.0
}'
```

### Single Prediction (Specific Models)

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "period": 365.25,
  "prad": 1.0,
  "teq": 288.0,
  "srad": 1.0,
  "slog_g": 4.44,
  "steff": 5778,
  "impact": 0.0,
  "duration": 13.0,
  "depth": 84.0,
  "models": ["random_forest", "cnn"]
}'
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
-H "Content-Type: application/json" \
-d '{
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
    "depth": 5000.0,
    "models": ["random_forest"]
  }
}'
```

### Python Client Example

```python
import requests
import json

# API endpoint
url = "http://localhost:8000/predict"

# Earth-like exoplanet data
data = {
    "period": 365.25,
    "prad": 1.0,
    "teq": 288.0,
    "srad": 1.0,
    "slog_g": 4.44,
    "steff": 5778,
    "impact": 0.0,
    "duration": 13.0,
    "depth": 84.0,
    "models": ["random_forest", "cnn"]  # Optional
}

# Make request
response = requests.post(url, json=data)
result = response.json()

# Access nested prediction results
rf_result = result['random_forest']['prediction']
cnn_result = result['cnn']['prediction']

print(f"Random Forest: {rf_result['prediction']} (Confidence: {rf_result['probability']:.3f})")
print(f"CNN: {cnn_result['prediction']} (Confidence: {cnn_result['probability']:.3f})")
print(f"Overall ESI: {result['esi']:.4f}")
print(f"Models executed: {result['models_executed']}")
```

## 📊 Response Format

### Single Prediction Response

```json
{
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
    "models": null
  },
  "models_executed": ["random_forest", "decision_tree", "knn", "cnn"]
}
```

### Batch Prediction Response

```json
{
  "earth_like": {
    "predictions": {
      "random_forest": {
        "prediction": {
          "prediction": "Not a Planet",
          "probability": 0.452
        },
        "model_type": "Random Forest"
      }
    },
    "esi": 1,
    "input_data": { /* input parameters */ },
    "models_executed": ["random_forest"]
  },
  "hot_jupiter": {
    "predictions": {
      "random_forest": {
        "prediction": {
          "prediction": "Planet",
          "probability": 0.823
        },
        "model_type": "Random Forest"
      }
    },
    "esi": 0.1234,
    "input_data": { /* input parameters */ },
    "models_executed": ["random_forest"]
  }
}
```

### Response Fields Explanation

| Field | Description |
|-------|-------------|
| `prediction.prediction` | Classification result: "Planet" or "Not a Planet" |
| `prediction.probability` | Confidence score (0-1) for the prediction |
| `prediction.ESI` | Earth Similarity Index (only available in KNN model results) |
| `model_type` | Name of the machine learning model used |
| `esi` | Overall Earth Similarity Index calculated for the input |
| `input_data` | Echo of the input parameters sent to the API |
| `models_executed` | List of models that successfully executed |

### Model-Specific Behaviors

- **Random Forest & Decision Tree**: Return prediction label and probability
- **KNN**: May include additional ESI calculation in prediction object
- **CNN**: Returns prediction label and raw probability score
- **All Models**: Wrapped in consistent structure with model_type identifier

## 🧪 Testing

### Using Jupyter Notebooks

1. **Model Testing**
   ```bash
   jupyter notebook notebooks/z.ipynb
   ```

2. **Additional Testing**
   ```bash
   jupyter notebook notebooks/zz.ipynb
   ```

### API Testing Scripts

1. **Basic API Testing**
   ```bash
   cd exobengal-api
   python test_api.py
   ```

2. **Model Selection Testing**
   ```bash
   cd exobengal-api
   python test_model_selection.py
   ```

### Health Check

```bash
curl http://localhost:8000/health
```

### Get Model Information

```bash
curl http://localhost:8000/models/info
```

## 🤖 Available Models

| Model | Type | Description |
|-------|------|-------------|
| **Random Forest** | Ensemble Learning | Random Forest Classifier for robust predictions |
| **Decision Tree** | Tree-based Learning | Decision Tree Classifier for interpretable results |
| **K-Nearest Neighbors** | Instance-based Learning | KNN Classifier for similarity-based predictions |
| **Convolutional Neural Network** | Deep Learning | CNN model for complex pattern recognition |

## 📈 Earth Similarity Index (ESI)

The API automatically calculates the Earth Similarity Index, which measures how similar an exoplanet is to Earth based on:
- Planet radius
- Equilibrium temperature

ESI ranges from 0 (completely different) to 1 (identical to Earth).


## 📋 Requirements

- fastapi==0.104.1
- uvicorn==0.24.0
- pydantic==2.4.2
- scikit-learn==1.6.1
- tensorflow==2.13.0
- numpy==1.24.3
- pandas==2.0.3
- requests==2.31.0
