#!/usr/bin/env python3

import sys
import os
from typing import Dict, Any, List, Optional
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Add the exobengal-api directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'exobengal-api'))

# Suppress warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Global detector variable
detector = None

def init():
    """Initialize the model - called once when the deployment starts"""
    global detector
    
    try:
        from exobengal import DetectExoplanet
        
        detector = DetectExoplanet(
            rf_model_path='./models/random_forest_classifier.pkl',
            dt_model_path='./models/decision_tree_classifier.pkl',
            cnn_model_path='./models/cnn_model.h5',
            knn_model_path='./models/knn_model.pkl',
            scaler_path='./models/scaler.pkl',
            imputer_path='./models/imputer.pkl'
        )
        return {"status": "ExoBengal API initialized successfully"}
    except Exception as e:
        return {"status": "error", "message": f"Initialization failed: {str(e)}"}

def predict(item: Dict[str, Any]):
    """
    Main prediction function for Cerebrium
    
    Expected input format:
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
        "models": ["random_forest", "cnn"]  # optional
    }
    """
    global detector
    
    if detector is None:
        init_result = init()
        if "error" in init_result.get("status", ""):
            return init_result
    
    try:
        # Extract input data
        period = item.get("period")
        prad = item.get("prad")
        teq = item.get("teq")
        srad = item.get("srad")
        slog_g = item.get("slog_g")
        steff = item.get("steff")
        impact = item.get("impact")
        duration = item.get("duration")
        depth = item.get("depth")
        models_requested = item.get("models", None)
        
        # Validate required fields
        required_fields = ["period", "prad", "teq", "srad", "slog_g", "steff", "impact", "duration", "depth"]
        missing_fields = [field for field in required_fields if item.get(field) is None]
        
        if missing_fields:
            return {
                "error": f"Missing required fields: {missing_fields}",
                "status": "error"
            }
        
        # Convert input to list format expected by models
        sample = [period, prad, teq, srad, slog_g, steff, impact, duration, depth]
        
        # Determine which models to run
        available_models = ["random_forest", "decision_tree", "knn", "cnn"]
        if models_requested is not None:
            # Validate specified models
            invalid_models = [m for m in models_requested if m not in available_models]
            if invalid_models:
                return {
                    "error": f"Invalid model(s): {invalid_models}. Available models: {available_models}",
                    "status": "error"
                }
            models_to_run = models_requested
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
                predictions[model] = {
                    "error": f"Model execution failed: {str(model_error)}",
                    "model_type": model.replace("_", " ").title()
                }
        
        # Calculate ESI
        esi = detector.calculate_esi(prad, teq)
        
        # Prepare response
        response = {
            "predictions": predictions,
            "esi": esi,
            "input_data": {
                "period": period,
                "prad": prad,
                "teq": teq,
                "srad": srad,
                "slog_g": slog_g,
                "steff": steff,
                "impact": impact,
                "duration": duration,
                "depth": depth
            },
            "models_executed": models_executed,
            "status": "success"
        }
        
        return response
        
    except Exception as e:
        return {
            "error": f"Prediction error: {str(e)}",
            "status": "error"
        }

def health_check():
    """Health check function for Cerebrium"""
    global detector
    
    if detector is None:
        try:
            init()
            return {
                "status": "healthy",
                "message": "ExoBengal API initialized and running successfully"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Failed to initialize: {str(e)}"
            }
    
    return {
        "status": "healthy",
        "message": "ExoBengal API is running successfully"
    }

def get_model_info():
    """Get information about available models"""
    return {
        "available_models": [
            {
                "name": "Random Forest",
                "type": "Ensemble Learning",
                "description": "Random Forest Classifier for exoplanet detection",
                "key": "random_forest"
            },
            {
                "name": "Decision Tree",
                "type": "Tree-based Learning",
                "description": "Decision Tree Classifier for exoplanet detection",
                "key": "decision_tree"
            },
            {
                "name": "K-Nearest Neighbors",
                "type": "Instance-based Learning",
                "description": "KNN Classifier for exoplanet detection",
                "key": "knn"
            },
            {
                "name": "Convolutional Neural Network",
                "type": "Deep Learning",
                "description": "CNN model for exoplanet detection",
                "key": "cnn"
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
        ],
        "example_request": {
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
    }
