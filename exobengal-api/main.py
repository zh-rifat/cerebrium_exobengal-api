#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import uvicorn
from exobengal import DetectExoplanet
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Initialize FastAPI app
app = FastAPI(
    title="ExoBengal API",
    description="Exoplanet Detection API using Machine Learning Models",
    version="1.0.0"
)

# Initialize the detector
detector = DetectExoplanet(
    rf_model_path='../models/random_forest_classifier.pkl',
    dt_model_path='../models/decision_tree_classifier.pkl',
    cnn_model_path='../models/cnn_model.h5',
    knn_model_path='../models/knn_model.pkl',
    scaler_path='../models/scaler.pkl',
    imputer_path='../models/imputer.pkl'
)

# Pydantic models for request/response
class ExoplanetInput(BaseModel):
    period: float = Field(..., description="Orbital period (days)", example=365.0)
    prad: float = Field(..., description="Planet radius (Earth radii)", example=1.0)
    teq: float = Field(..., description="Equilibrium temperature (Kelvin)", example=288.0)
    srad: float = Field(..., description="Stellar radius (solar radii)", example=1.0)
    slog_g: float = Field(..., description="Stellar surface gravity (log scale)", example=4.44)
    steff: float = Field(..., description="Stellar effective temperature (Kelvin)", example=5778)
    impact: float = Field(..., description="Impact parameter (related to transit geometry)", example=0.1)
    duration: float = Field(..., description="Transit duration (hours)", example=5.0)
    depth: float = Field(..., description="Transit depth (parts per million)", example=100.0)
    models: Optional[List[str]] = Field(
        default=None, 
        description="Specific models to run. Options: 'random_forest', 'decision_tree', 'knn', 'cnn'. If not specified, all models will run.",
        example=["random_forest", "cnn"]
    )

class PredictionResponse(BaseModel):
    random_forest: Optional[Dict[str, Any]] = None
    decision_tree: Optional[Dict[str, Any]] = None
    knn: Optional[Dict[str, Any]] = None
    cnn: Optional[Dict[str, Any]] = None
    esi: float
    input_data: Dict[str, Any]
    models_executed: List[str]

class HealthResponse(BaseModel):
    status: str
    message: str

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to ExoBengal API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="ExoBengal API is running successfully"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_exoplanet(input_data: ExoplanetInput):
    """
    Predict exoplanet characteristics using specified or all available models
    
    Parameters:
    - period: Orbital period (days)
    - prad: Planet radius (Earth radii)
    - teq: Equilibrium temperature (Kelvin)
    - srad: Stellar radius (solar radii)
    - slog_g: Stellar surface gravity (log scale)
    - steff: Stellar effective temperature (Kelvin)
    - impact: Impact parameter (related to transit geometry)
    - duration: Transit duration (hours)
    - depth: Transit depth (parts per million)
    - models: Optional list of specific models to run ['random_forest', 'decision_tree', 'knn', 'cnn']
    
    Returns predictions from specified models and ESI calculation
    """
    try:
        # Convert input to list format expected by models
        sample = [
            input_data.period,
            input_data.prad,
            input_data.teq,
            input_data.srad,
            input_data.slog_g,
            input_data.steff,
            input_data.impact,
            input_data.duration,
            input_data.depth
        ]
        
        # Determine which models to run
        available_models = ["random_forest", "decision_tree", "knn", "cnn"]
        if input_data.models is not None:
            # Validate specified models
            invalid_models = [m for m in input_data.models if m not in available_models]
            if invalid_models:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid model(s): {invalid_models}. Available models: {available_models}"
                )
            models_to_run = input_data.models
        else:
            models_to_run = available_models
        
        # Get predictions from specified models
        predictions = {}
        models_executed = []
        
        for model in models_to_run:
            try:
                prediction = getattr(detector, model)(sample)
                predictions[model] = {
                    "prediction": prediction,
                    "model_type": model.replace("_", " ").title()
                }
                models_executed.append(model)
            except Exception as model_error:
                # Log model-specific error but continue with other models
                print(f"Error running {model}: {model_error}")
                predictions[model] = {
                    "error": f"Model execution failed: {str(model_error)}",
                    "model_type": model.replace("_", " ").title()
                }
        
        # Calculate ESI
        esi = detector.calculate_esi(input_data.prad, input_data.teq)
        
        # Prepare response with only the models that were requested
        response_data = {
            "esi": esi,
            "input_data": input_data.dict(),
            "models_executed": models_executed
        }
        
        # Add model predictions to response
        for model in available_models:
            if model in predictions:
                response_data[model] = predictions[model]
            else:
                response_data[model] = None
        
        return PredictionResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(samples: Dict[str, ExoplanetInput]):
    """
    Predict multiple exoplanet samples in batch
    
    Input: Dictionary with sample names as keys and ExoplanetInput as values
    Returns: Dictionary with predictions for each sample
    """
    try:
        results = {}
        
        for sample_name, input_data in samples.items():
            # Convert input to list format
            sample = [
                input_data.period,
                input_data.prad,
                input_data.teq,
                input_data.srad,
                input_data.slog_g,
                input_data.steff,
                input_data.impact,
                input_data.duration,
                input_data.depth
            ]
            
            # Determine which models to run for this sample
            available_models = ["random_forest", "decision_tree", "knn", "cnn"]
            if input_data.models is not None:
                # Validate specified models
                invalid_models = [m for m in input_data.models if m not in available_models]
                if invalid_models:
                    results[sample_name] = {
                        "error": f"Invalid model(s): {invalid_models}. Available models: {available_models}",
                        "input_data": input_data.dict()
                    }
                    continue
                models_to_run = input_data.models
            else:
                models_to_run = available_models
            
            # Get predictions for this sample
            predictions = {}
            models_executed = []
            
            for model in models_to_run:
                try:
                    prediction = getattr(detector, model)(sample)
                    predictions[model] = {
                        "prediction": prediction,
                        "model_type": model.replace("_", " ").title()
                    }
                    models_executed.append(model)
                except Exception as model_error:
                    predictions[model] = {
                        "error": f"Model execution failed: {str(model_error)}",
                        "model_type": model.replace("_", " ").title()
                    }
            
            # Calculate ESI
            esi = detector.calculate_esi(input_data.prad, input_data.teq)
            
            results[sample_name] = {
                "predictions": predictions,
                "esi": esi,
                "input_data": input_data.dict(),
                "models_executed": models_executed
            }
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/models/info")
async def get_models_info():
    """Get information about available models"""
    return {
        "available_models": [
            {
                "name": "Random Forest",
                "type": "Ensemble Learning",
                "description": "Random Forest Classifier for exoplanet detection"
            },
            {
                "name": "Decision Tree",
                "type": "Tree-based Learning",
                "description": "Decision Tree Classifier for exoplanet detection"
            },
            {
                "name": "K-Nearest Neighbors",
                "type": "Instance-based Learning",
                "description": "KNN Classifier for exoplanet detection"
            },
            {
                "name": "Convolutional Neural Network",
                "type": "Deep Learning",
                "description": "CNN model for exoplanet detection"
            }
        ],
        "input_features": [
            "period: Orbital period (days)",
            "prad: Planet radius (Earth radii)",
            "teq: Equilibrium temperature (Kelvin)",
            "srad: Stellar radius (solar radii)",
            "slog_g: Stellar surface gravity (log scale)",
            "steff: Stellar effective temperature (Kelvin)",
            "impact: Impact parameter (related to transit geometry)",
            "duration: Transit duration (hours)",
            "depth: Transit depth (parts per million)"
        ],
        "additional_calculations": [
            "ESI: Earth Similarity Index"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
