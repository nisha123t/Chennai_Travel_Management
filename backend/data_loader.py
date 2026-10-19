"""
Data loader utilities for the Chennai Multimodal Transportation system.
Provides helper functions to load stops, routes, stop sequences, connections,
and machine learning datasets.
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
MODELS_DATA_DIR = Path(__file__).resolve().parent / "models" / "data"


def load_stops() -> pd.DataFrame:
    """Load all stops with coordinate and mode attributes."""
    return pd.read_csv(DATA_DIR / "stops.csv")


def load_routes() -> pd.DataFrame:
    """Load all canonical transport routes."""
    return pd.read_csv(DATA_DIR / "routes.csv")


def load_stop_sequences() -> pd.DataFrame:
    """Load ordered route stop sequences."""
    return pd.read_csv(DATA_DIR / "stop_sequences.csv")


def load_connections() -> pd.DataFrame:
    """Load directed connection edges with distances, times, and fares."""
    return pd.read_csv(DATA_DIR / "connections.csv")


def load_all_transit_data() -> dict:
    """Load all four transit CSVs into a single dictionary."""
    return {
        "stops": load_stops(),
        "routes": load_routes(),
        "sequences": load_stop_sequences(),
        "connections": load_connections()
    }


def load_ml_dataset() -> pd.DataFrame:
    """Load or generate the ML travel time training dataset."""
    ml_file = MODELS_DATA_DIR / "travel_time_dataset.csv"
    if ml_file.exists():
        return pd.read_csv(ml_file)
    from models.dataset_builder import generate_travel_time_dataset
    return generate_travel_time_dataset(output_path=ml_file)
