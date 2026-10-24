"""
Inference service for ML travel time models.
Provides simple functional APIs to predict travel times for single trips or complete routes.
"""
from pathlib import Path
import pandas as pd
from linear_regression_model import LinearRegressionTravelTimeModel
from random_forest_model import RandomForestTravelTimeModel
from dataset_builder import BUCKET_TO_HOUR, time_of_day_bucket

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


def _resolve_hour(departure_hour, time_of_day):
    if departure_hour is not None:
        return int(departure_hour) % 24
    return BUCKET_TO_HOUR.get(str(time_of_day or "morning_peak").lower(), 9)


def _feature_row(distance_km, transport_type, stops_count, hour, day_type):
    day_type = str(day_type or "weekday").lower()
    return {
        "distance_km": float(distance_km),
        "stops_count": max(1, int(stops_count)),
        "departure_hour": hour,
        "transport_type": str(transport_type).lower(),
        "time_of_day": time_of_day_bucket(hour, day_type),
        "day_type": day_type,
    }


def predict_travel_time(
    distance_km: float,
    transport_type: str,
    stops_count: int = 1,
    departure_hour: int = None,
    time_of_day: str = None,
    day_type: str = "weekday",
    model_type: str = "random_forest",
    fare_inr: float = None,  # accepted for backward compatibility; no longer a feature
) -> float:
    """
    Predict the travel time (minutes) of one continuous ride segment.

    - distance_km:    ride distance
    - transport_type: metro | rail | bus | walk | auto | cab
    - stops_count:    number of stops travelled on this ride (>= 1)
    - departure_hour: 0-23. If omitted, derived from time_of_day
                      ('morning_peak', 'evening_peak', 'afternoon', 'off_peak',
                       'weekend_day', 'weekend_night').
    - day_type:       'weekday' | 'weekend'
    """
    model = get_model(model_type)
    hour = _resolve_hour(departure_hour, time_of_day)
    row = _feature_row(distance_km, transport_type, stops_count, hour, day_type)
    return round(float(model.predict(pd.DataFrame([row]))[0]), 2)


def enhance_route_with_ml(
    route_result: dict,
    model_type: str = "random_forest",
    time_of_day: str = None,
    day_type: str = "weekday",
    departure_hour: int = None,
) -> dict:
    """
    Add ML-predicted times to a route returned by router.find_route().

    Predictions are made per *segment* (one continuous ride, the same unit the
    models were trained on). Each segment's prediction is then shared across
    its legs in proportion to their timetable times. Segment start times are
    chained, so a later segment uses the hour at which it actually starts.
    """
    if route_result is None:
        return None
    model = get_model(model_type)
    enriched = dict(route_result)
    hour0 = _resolve_hour(departure_hour, time_of_day)
    segments = enriched.get("segments", [])
    if not segments:
        return enriched

    elapsed = 0.0
    total = 0.0
    for seg in segments:
        hour = int(hour0 + elapsed / 60.0) % 24
        row = _feature_row(seg["distance_km"], seg["transport_type"], seg.get("num_stops", 1),
                           hour, day_type)
        pred = round(float(model.predict(pd.DataFrame([row]))[0]), 2)
        seg["ml_predicted_time_min"] = pred
        elapsed += pred
        total += pred

    # distribute segment predictions to legs (legs are ordered segment by segment)
    legs = enriched.get("legs", [])
    li = 0
    for seg in segments:
        n = seg.get("num_stops", 1)
        seg_legs = legs[li:li + n]
        li += n
        base = sum(l.get("time_min", 0.0) for l in seg_legs) or 1.0
        for l in seg_legs:
            l["ml_predicted_time_min"] = round(seg["ml_predicted_time_min"] * l.get("time_min", 0.0) / base, 2)

    enriched["ml_total_time_min"] = round(total, 2)
    enriched["ml_model_used"] = model_type
    enriched["departure_hour"] = hour0
    enriched["time_of_day"] = time_of_day_bucket(hour0, day_type)
    enriched["day_type"] = day_type
    return enriched


def predict_segments(rows: list, model_type: str = "random_forest"):
    """
    Batch prediction for many ride segments in one model call.
    Each row: distance_km, transport_type, stops_count, departure_hour, day_type.
    """
    if not rows:
        return []
    model = get_model(model_type)
    frame = pd.DataFrame([
        _feature_row(r["distance_km"], r["transport_type"], r.get("stops_count", 1),
                     int(r.get("departure_hour", 9)) % 24, r.get("day_type", "weekday"))
        for r in rows
    ])
    return model.predict(frame)
