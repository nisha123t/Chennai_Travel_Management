"""
Random Forest Regressor Model for Multimodal Travel Time Prediction in Chennai.
Captures non-linear dynamics, mode-specific congestion impacts, and feature interactions.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path


class RandomForestTravelTimeModel:
    """
    Random Forest Regression Travel Time Model.
    Predicts travel_time_min capturing non-linear interactions across distance,
    speed, transit mode, and peak traffic intervals.
    """

    NUMERIC_FEATURES = ["distance_km", "stops_count", "departure_hour"]
    CATEGORICAL_FEATURES = ["transport_type", "time_of_day", "day_type"]
    TARGET = "travel_time_min"

    def __init__(self, n_estimators=200, max_depth=14, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.CATEGORICAL_FEATURES)
            ]
        )
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("regressor", RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    n_jobs=-1
                ))
            ]
        )
        self.is_fitted = False
        self.metrics = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the Random Forest regression model."""
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict travel time in minutes for input samples."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() or load().")
        preds = self.pipeline.predict(X)
        return np.maximum(1.0, preds)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluate performance metrics on test set."""
        preds = self.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        self.metrics = {
            "model_type": "Random Forest Regressor",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "mae": round(float(mae), 3),
            "mse": round(float(mse), 3),
            "rmse": round(float(rmse), 3),
            "r2_score": round(float(r2), 4)
        }
        return self.metrics

    def get_feature_importances(self) -> dict:
        """Extract top feature importances from the ensemble trees."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        
        regressor = self.pipeline.named_steps["regressor"]
        preprocessor = self.pipeline.named_steps["preprocessor"]
        
        num_names = self.NUMERIC_FEATURES
        cat_encoder = preprocessor.named_transformers_["cat"]
        cat_names = cat_encoder.get_feature_names_out(self.CATEGORICAL_FEATURES).tolist()
        all_features = num_names + cat_names

        importances = regressor.feature_importances_
        sorted_pairs = sorted(zip(all_features, importances), key=lambda x: x[1], reverse=True)
        return {feat: round(float(imp), 4) for feat, imp in sorted_pairs}

    def save(self, filepath: str):
        """Serialize model to file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pipeline": self.pipeline,
            "metrics": self.metrics,
            "is_fitted": self.is_fitted
        }, path)
        print(f"Random Forest model saved to {path}")

    @classmethod
    def load(cls, filepath: str):
        """Load serialized model from file."""
        data = joblib.load(filepath)
        instance = cls()
        instance.pipeline = data["pipeline"]
        instance.metrics = data.get("metrics", {})
        instance.is_fitted = data.get("is_fitted", True)
        return instance
