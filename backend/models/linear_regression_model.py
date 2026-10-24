"""
Multiple Linear Regression Model for Multimodal Travel Time Prediction in Chennai.
Uses scikit-learn Pipeline with ColumnTransformer (StandardScaler + OneHotEncoder).
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path


class LinearRegressionTravelTimeModel:
    """
    Multiple Linear Regression Travel Time Model.
    Predicts travel_time_min from distance_km, fare_inr, stops_count,
    transport_type, time_of_day, and day_type.
    """

    NUMERIC_FEATURES = ["distance_km", "stops_count", "departure_hour"]
    CATEGORICAL_FEATURES = ["transport_type", "time_of_day", "day_type"]
    TARGET = "travel_time_min"

    def __init__(self):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), self.CATEGORICAL_FEATURES)
            ]
        )
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("regressor", LinearRegression())
            ]
        )
        self.is_fitted = False
        self.metrics = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the multiple linear regression model."""
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict travel time in minutes for input samples."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() or load().")
        preds = self.pipeline.predict(X)
        # Travel time cannot be negative
        return np.maximum(1.0, preds)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluate performance metrics on test set."""
        preds = self.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        self.metrics = {
            "model_type": "Multiple Linear Regression",
            "mae": round(float(mae), 3),
            "mse": round(float(mse), 3),
            "rmse": round(float(rmse), 3),
            "r2_score": round(float(r2), 4)
        }
        return self.metrics

    def get_feature_coefficients(self) -> dict:
        """Extract learned coefficients for each one-hot and scaled feature."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        
        regressor = self.pipeline.named_steps["regressor"]
        preprocessor = self.pipeline.named_steps["preprocessor"]
        
        # Get transformed feature names
        num_names = self.NUMERIC_FEATURES
        cat_encoder = preprocessor.named_transformers_["cat"]
        cat_names = cat_encoder.get_feature_names_out(self.CATEGORICAL_FEATURES).tolist()
        all_features = num_names + cat_names

        coefs = regressor.coef_
        result = {
            feat: round(float(coef), 4)
            for feat, coef in zip(all_features, coefs)
        }
        result["intercept"] = round(float(regressor.intercept_), 4)
        return result

    def save(self, filepath: str):
        """Serialize model to file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pipeline": self.pipeline,
            "metrics": self.metrics,
            "is_fitted": self.is_fitted
        }, path, compress=3)  # compressed: keeps the file well under GitHub's 100 MB limit
        print(f"Linear Regression model saved to {path}")

    @classmethod
    def load(cls, filepath: str):
        """Load serialized model from file."""
        data = joblib.load(filepath)
        instance = cls()
        instance.pipeline = data["pipeline"]
        instance.metrics = data.get("metrics", {})
        instance.is_fitted = data.get("is_fitted", True)
        return instance
