"""
Chennai Multimodal Route Planner API.

Run:  uvicorn main:app --reload --port 8000      Docs: http://localhost:8000/docs
"""
import json
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from router import find_route
from route_engine import get_option, list_stops, plan_journeys, search_stops
from models.predictor import enhance_route_with_ml, predict_travel_time

app = FastAPI(
    title="Chennai Route Planner API",
    description="ML-enhanced multimodal public transport route planner for Chennai "
                "(Metro, Suburban Rail / MRTS, MTC Bus, walking, auto, cab).",
    version="2.0.0",
)

# React dev server (Vite 5173 / CRA 3000) on localhost or 127.0.0.1, any port.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

Preference = Literal["fastest", "cheapest", "least_walking", "balanced", "shortest"]
DayType = Literal["weekday", "weekend"]
ModelType = Literal["random_forest", "linear_regression"]
Mode = Literal["metro", "rail", "bus", "walk", "auto", "cab"]


# ---------------------------------------------------------------- schemas
class RouteRequest(BaseModel):
    start: str = Field(..., min_length=1, examples=["S001"])
    destination: str = Field(..., min_length=1, examples=["S058"])


class JourneyRequest(BaseModel):
    start: str = Field(..., min_length=1, description="stop_id (e.g. S001) or exact stop name",
                       examples=["S001"])
    destination: str = Field(..., min_length=1, examples=["S058"])
    preference: Preference = "balanced"
    departure_hour: int = Field(9, ge=0, le=23)
    day_type: DayType = "weekday"
    max_options: int = Field(3, ge=1, le=5)
    ml_enhanced: bool = True
    model: ModelType = "random_forest"

    @field_validator("start", "destination")
    @classmethod
    def strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


class PredictionRequest(BaseModel):
    distance_km: float = Field(..., gt=0, le=200)
    transport_type: Mode
    stops_count: int = Field(1, ge=1, le=60)
    departure_hour: Optional[int] = Field(None, ge=0, le=23)
    time_of_day: Optional[str] = "morning_peak"
    day_type: DayType = "weekday"
    model_type: ModelType = "random_forest"


# ---------------------------------------------------------------- endpoints
@app.get("/", tags=["meta"])
def home():
    return {
        "message": "Chennai Multimodal Route Planner API is running",
        "docs": "/docs",
        "ml_models": ["Multiple Linear Regression", "Random Forest Regressor"],
    }


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "stops": len(list_stops())}


@app.get("/stops", tags=["stops"])
def get_all_stops():
    """All stops with coordinates, modes served and interchange flag."""
    return list_stops()


@app.get("/stops/search", tags=["stops"])
def stop_search(q: str = Query(..., min_length=1, max_length=60), limit: int = Query(10, ge=1, le=50)):
    """Search stops by name, area or id (prefix matches first)."""
    return search_stops(q, limit)


@app.post("/routes", tags=["routing"])
def route_options(req: JourneyRequest):
    """
    Up to `max_options` distinct journeys between two stops, ranked by
    `preference`. Each option has total time (timetable and ML-predicted),
    fare, walking distance, transfers, mode changes, segments and legs.
    """
    try:
        result = plan_journeys(req.start, req.destination, req.preference, req.max_options,
                               req.departure_hour, req.day_type, req.ml_enhanced, req.model)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    if not result["options"]:
        raise HTTPException(status_code=404, detail="No route found between these stops")
    return result


@app.get("/routes/{option_id}", tags=["routing"])
def route_details(option_id: str):
    """Full details (segments, hop-by-hop legs, stop coordinates) of a returned option."""
    option = get_option(option_id)
    if option is None:
        raise HTTPException(status_code=404, detail="Route option not found or expired; search again")
    return option


@app.post("/route", tags=["routing"], deprecated=True)
def calculate_route(
    request: RouteRequest,
    preference: str = "fastest",
    ml_enhanced: bool = False,
    time_of_day: str = "morning_peak",
    departure_hour: Optional[int] = Query(None, ge=0, le=23),
    day_type: DayType = "weekday",
    model: ModelType = "random_forest",
):
    """Legacy single-route endpoint (kept for compatibility). Prefer POST /routes."""
    try:
        result = find_route(request.start, request.destination, preference)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    if result is None:
        raise HTTPException(status_code=404, detail="No route found between these locations")
    if ml_enhanced:
        try:
            result = enhance_route_with_ml(result, model_type=model, time_of_day=time_of_day,
                                           departure_hour=departure_hour, day_type=day_type)
        except Exception as ex:
            result["ml_warning"] = f"ML enhancement failed: {ex}"
    return result


@app.post("/predict-time", tags=["ml"])
def predict_time(req: PredictionRequest):
    """Predict the travel time of one continuous ride segment."""
    try:
        minutes = predict_travel_time(
            distance_km=req.distance_km, transport_type=req.transport_type,
            stops_count=req.stops_count, departure_hour=req.departure_hour,
            time_of_day=req.time_of_day, day_type=req.day_type, model_type=req.model_type,
        )
    except Exception as ex:
        raise HTTPException(status_code=400, detail=str(ex))
    return {**req.model_dump(), "model_used": req.model_type, "predicted_travel_time_min": minutes}


@app.get("/models/metrics", tags=["ml"])
def get_metrics():
    """Evaluation metrics for Linear Regression vs Random Forest (and timetable baseline)."""
    path = Path(__file__).resolve().parent / "models" / "saved_models" / "model_metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model metrics not found. Run models/train_models.py.")
    return json.loads(path.read_text())
