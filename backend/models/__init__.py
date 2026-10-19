"""
Machine Learning models package for Multimodal Travel Time Prediction in Chennai.
Exposes Multiple Linear Regression and Random Forest models and inference functions.
"""
from pathlib import Path
import sys

# Ensure current package path is in sys.path
_current_dir = str(Path(__file__).resolve().parent)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from linear_regression_model import LinearRegressionTravelTimeModel
from random_forest_model import RandomForestTravelTimeModel
from predictor import predict_travel_time, enhance_route_with_ml, get_model
from train_models import run_training_pipeline

__all__ = [
    "LinearRegressionTravelTimeModel",
    "RandomForestTravelTimeModel",
    "predict_travel_time",
    "enhance_route_with_ml",
    "get_model",
    "run_training_pipeline"
]
