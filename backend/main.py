
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
from pathlib import Path

from router import find_route, get_stops
from models.predictor import predict_travel_time, enhance_route_with_ml

app = FastAPI(title="Chennai Route Planner API", description="ML-Enhanced Multimodal Public Transportation Route Planner for Chennai")

# Allow requests from the React development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RouteRequest(BaseModel):
    start: str
    destination: str


class PredictionRequest(BaseModel):
    distance_km: float
    transport_type: str
    fare_inr: Optional[float] = 0.0
    stops_count: Optional[int] = 1
    time_of_day: Optional[str] = "morning_peak"
    day_type: Optional[str] = "weekday"
    model_type: Optional[str] = "random_forest"


@app.get("/")
def home():
    return {
        "message": "Chennai Multimodal Route Planner API is running",
        "ml_models": ["Multiple Linear Regression", "Random Forest Regressor"]
    }


@app.get("/stops")
def list_stops():
    return get_stops()


@app.post("/route")
def calculate_route(
    request: RouteRequest,
    preference: str = "fastest",
    ml_enhanced: bool = False,
    time_of_day: str = "morning_peak",
    model: str = "random_forest"
):
    try:
        result = find_route(
            request.start,
            request.destination,
            preference
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No route found between these locations"
        )

    if ml_enhanced:
        try:
            result = enhance_route_with_ml(
                result,
                model_type=model,
                time_of_day=time_of_day
            )
        except Exception as ex:
            # Fallback gracefully to graph baseline if ML model is unavailable
            result["ml_warning"] = f"ML enhancement failed: {str(ex)}"

    return result


@app.post("/predict-time")
def predict_time(req: PredictionRequest):
    """Predict travel time in minutes using ML models."""
    try:
        predicted_min = predict_travel_time(
            distance_km=req.distance_km,
            transport_type=req.transport_type,
            fare_inr=req.fare_inr,
            stops_count=req.stops_count,
            time_of_day=req.time_of_day,
            day_type=req.day_type,
            model_type=req.model_type
        )
        return {
            "distance_km": req.distance_km,
            "transport_type": req.transport_type,
            "time_of_day": req.time_of_day,
            "model_used": req.model_type,
            "predicted_travel_time_min": predicted_min
        }
    except Exception as ex:
        raise HTTPException(status_code=400, detail=str(ex))


@app.get("/models/metrics")
def get_metrics():
    """Return performance metrics for Linear Regression and Random Forest models."""
    metrics_path = Path(__file__).resolve().parent / "models" / "saved_models" / "model_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Model metrics file not found. Run training first.")
