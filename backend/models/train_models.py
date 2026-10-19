"""
Training pipeline for Multiple Linear Regression and Random Forest travel time models.
Splits data, trains both models, evaluates performance, and saves serialized artifacts.
"""
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

from dataset_builder import generate_travel_time_dataset
from linear_regression_model import LinearRegressionTravelTimeModel
from random_forest_model import RandomForestTravelTimeModel

MODELS_DIR = Path(__file__).resolve().parent
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"
DATA_DIR = MODELS_DIR / "data"

def run_training_pipeline(test_size=0.2, random_state=42):
    print("==================================================")
    print("   MULTIMODAL TRAVEL TIME PREDICTION TRAINING     ")
    print("==================================================")

    # 1. Generate or load training data
    dataset_file = DATA_DIR / "travel_time_dataset.csv"
    print("\n[Step 1] Preparing dataset...")
    df = generate_travel_time_dataset(output_path=dataset_file, samples_per_connection=5, random_state=random_state)
    print(f"Total samples: {len(df)}")
    print("Modes distribution:")
    print(df["transport_type"].value_counts().to_string())

    feature_cols = ["distance_km", "fare_inr", "stops_count", "transport_type", "time_of_day", "day_type"]
    target_col = "travel_time_min"

    X = df[feature_cols]
    y = df[target_col]

    # 2. Train-test split
    print(f"\n[Step 2] Splitting dataset ({int((1-test_size)*100)}% Train / {int(test_size*100)}% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Multiple Linear Regression Model
    print("\n[Step 3] Training Multiple Linear Regression Model...")
    lr_model = LinearRegressionTravelTimeModel()
    lr_model.fit(X_train, y_train)
    lr_metrics = lr_model.evaluate(X_test, y_test)
    lr_model.save(SAVED_MODELS_DIR / "linear_regression.joblib")

    print("\n--- Linear Regression Performance ---")
    print(f"  MAE:      {lr_metrics['mae']:.3f} min")
    print(f"  RMSE:     {lr_metrics['rmse']:.3f} min")
    print(f"  R2 Score: {lr_metrics['r2_score']:.4f}")
    
    coefs = lr_model.get_feature_coefficients()
    print("  Top Learned Coefficients:")
    for k, v in list(coefs.items())[:8]:
        print(f"    * {k}: {v:+.4f}")

    # 4. Random Forest Regressor Model
    print("\n[Step 4] Training Random Forest Regressor Model...")
    rf_model = RandomForestTravelTimeModel(n_estimators=100, max_depth=12, random_state=random_state)
    rf_model.fit(X_train, y_train)
    rf_metrics = rf_model.evaluate(X_test, y_test)
    rf_model.save(SAVED_MODELS_DIR / "random_forest.joblib")

    print("\n--- Random Forest Regressor Performance ---")
    print(f"  MAE:      {rf_metrics['mae']:.3f} min")
    print(f"  RMSE:     {rf_metrics['rmse']:.3f} min")
    print(f"  R2 Score: {rf_metrics['r2_score']:.4f}")

    importances = rf_model.get_feature_importances()
    print("  Top Feature Importances:")
    for k, v in list(importances.items())[:6]:
        print(f"    * {k}: {v * 100:.2f}%")

    # 5. Save comparison metrics
    comparison = {
        "dataset_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "linear_regression": lr_metrics,
        "random_forest": rf_metrics
    }

    metrics_file = SAVED_MODELS_DIR / "model_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[Step 5] Metrics saved to {metrics_file}")

    print("\n==================================================")
    print("              MODEL COMPARISON SUMMARY            ")
    print("==================================================")
    print(f"{'Metric':<15} | {'Linear Regression':<20} | {'Random Forest':<20}")
    print("-" * 62)
    print(f"{'MAE (min)':<15} | {lr_metrics['mae']:<20.3f} | {rf_metrics['mae']:<20.3f}")
    print(f"{'RMSE (min)':<15} | {lr_metrics['rmse']:<20.3f} | {rf_metrics['rmse']:<20.3f}")
    print(f"{'R2 Score':<15} | {lr_metrics['r2_score']:<20.4f} | {rf_metrics['r2_score']:<20.4f}")
    print("==================================================")

    return lr_model, rf_model, comparison

if __name__ == "__main__":
    run_training_pipeline()
