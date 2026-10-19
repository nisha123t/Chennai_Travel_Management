"""
Inference service for ML travel time models.
Provides simple functional APIs to predict travel times for single trips or complete routes.
"""
from pathlib import Path
import pandas as pd
from linear_regression_model import LinearRegressionTravelTimeModel
from random_forest_model import RandomForestTravelTimeModel

SAVED_MODELS_DIR = Path(__file__).resolve().parent / "saved_models"

_CACHED_MODELS = {}

def get_model(model_type="random_forest"):
    """Loads and caches the requested ML model."""
    model_type = model_type.lower()
    if model_type in _CACHED_MODELS:
        return _CACHED_MODELS[model_type]

    if model_type in ("random_forest", "rf"):
        model_path = SAVED_MODELS_DIR / "random_forest.joblib"
        if not model_path.exists():
            from train_models import run_training_pipeline
            run_training_pipeline()
        model = RandomForestTravelTimeModel.load(str(model_path))
        _CACHED_MODELS["random_forest"] = model
        _CACHED_MODELS["rf"] = model
        return model

    elif model_type in ("linear_regression", "lr", "linear"):
        model_path = SAVED_MODELS_DIR / "linear_regression.joblib"
        if not model_path.exists():
            from train_models import run_training_pipeline
            run_training_pipeline()
        model = LinearRegressionTravelTimeModel.load(str(model_path))
        _CACHED_MODELS["linear_regression"] = model
        _CACHED_MODELS["lr"] = model
        return model

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Expected 'random_forest' or 'linear_regression'.")


def predict_travel_time(
    distance_km: float,
    transport_type: str,
    fare_inr: float = 0.0,
    stops_count: int = 1,
    time_of_day: str = "morning_peak",
    day_type: str = "weekday",
    model_type: str = "random_forest"
) -> float:
    """
    Predict travel time in minutes for a single transit leg or trip.
    
    Parameters:
    - distance_km: Trip distance in km
    - transport_type: Mode (metro, rail, bus, walk, auto, cab)
    - fare_inr: Estimated fare in INR
    - stops_count: Number of intermediate stops/stations (default: 1)
    - time_of_day: 'morning_peak', 'evening_peak', 'afternoon', 'off_peak', 'weekend_day', 'weekend_night'
    - day_type: 'weekday' or 'weekend'
    - model_type: 'random_forest' or 'linear_regression'
    
    Returns:
    - Predicted travel time in minutes (float)
    """
    model = get_model(model_type)
    
    input_df = pd.DataFrame([{
        "distance_km": float(distance_km),
        "fare_inr": float(fare_inr),
        "stops_count": int(stops_count),
        "transport_type": str(transport_type).lower(),
        "time_of_day": str(time_of_day).lower(),
        "day_type": str(day_type).lower()
    }])

    pred = model.predict(input_df)[0]
    return round(float(pred), 2)


def enhance_route_with_ml(
    route_result: dict,
    model_type: str = "random_forest",
    time_of_day: str = "morning_peak",
    day_type: str = "weekday"
) -> dict:
    """
    Takes a route dictionary returned by router.find_route(...) and adds
    ML-predicted travel times alongside graph baseline times.
    """
    if route_result is None:
        return None

    model = get_model(model_type)
    enriched = dict(route_result)
    
    legs_inputs = []
    for leg in enriched.get("legs", []):
        legs_inputs.append({
            "distance_km": float(leg["distance_km"]),
            "fare_inr": float(leg.get("fare_inr", 0.0)),
            "stops_count": 1,
            "transport_type": str(leg["transport_type"]).lower(),
            "time_of_day": time_of_day,
            "day_type": day_type
        })
        
    if legs_inputs:
        preds = model.predict(pd.DataFrame(legs_inputs))
        ml_total_time = 0.0
        for leg, pred_time in zip(enriched["legs"], preds):
            pred_time = round(float(pred_time), 2)
            leg["ml_predicted_time_min"] = pred_time
            ml_total_time += pred_time
            
        enriched["ml_total_time_min"] = round(ml_total_time, 2)
        enriched["ml_model_used"] = model_type
        enriched["time_of_day"] = time_of_day
        enriched["day_type"] = day_type
        
    return enriched
