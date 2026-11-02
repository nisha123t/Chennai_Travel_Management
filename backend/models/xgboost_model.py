"""
XGBoost Regressor Model for Multimodal Travel Time Prediction in Chennai.
Uses the scikit-learn compatible XGBRegressor with the same preprocessor pipeline
as the other travel-time models.
"""
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


class XGBoostTravelTimeModel:
    NUMERIC_FEATURES = ["distance_km", "stops_count", "departure_hour"]
    CATEGORICAL_FEATURES = ["transport_type", "time_of_day", "day_type"]
    TARGET = "travel_time_min"

    def __init__(self, n_estimators=300, max_depth=8, learning_rate=0.05,
                 subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                 random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.random_state = random_state

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.CATEGORICAL_FEATURES),
            ]
        )
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                (
                    "regressor",
                    XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        learning_rate=self.learning_rate,
                        subsample=self.subsample,
                        colsample_bytree=self.colsample_bytree,
                        reg_lambda=self.reg_lambda,
                        random_state=self.random_state,
                        n_jobs=-1,
                        verbosity=0,
                    ),
                ),
            ]
        )
        self.is_fitted = False
        self.metrics = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() or load().")
        preds = self.pipeline.predict(X)
        return np.maximum(1.0, preds)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        preds = self.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        self.metrics = {
            "model_type": "XGBoost Regressor",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "mae": round(float(mae), 3),
            "mse": round(float(mse), 3),
            "rmse": round(float(rmse), 3),
            "r2_score": round(float(r2), 4),
        }
        return self.metrics

    def get_feature_importances(self) -> dict:
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
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pipeline": self.pipeline,
            "metrics": self.metrics,
            "is_fitted": self.is_fitted,
        }, path, compress=3)
        print(f"XGBoost model saved to {path}")

    @classmethod
    def load(cls, filepath: str):
        data = joblib.load(filepath)
        instance = cls()
        instance.pipeline = data["pipeline"]
        instance.metrics = data.get("metrics", {})
        instance.is_fitted = data.get("is_fitted", True)
        return instance
