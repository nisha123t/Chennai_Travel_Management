"""
Training + evaluation pipeline for the journey-time models.

Evaluation design (why the numbers can be trusted):
* Group split by route_id: every sample from a given route goes entirely to
  train OR test, so the test set measures performance on routes the model has
  never seen (no memorising of duplicated connections).
* 5-fold GroupKFold cross-validation on the training routes -> mean +/- std.
* A no-ML baseline (raw timetable time) so the models' gain is visible.
* Per-mode MAE, feature importances / coefficients, and a sample of test
  predictions for the frontend's predicted-vs-actual chart.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from dataset_builder import generate_travel_time_dataset
from linear_regression_model import LinearRegressionTravelTimeModel
from random_forest_model import RandomForestTravelTimeModel

MODELS_DIR = Path(__file__).resolve().parent
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"
DATA_DIR = MODELS_DIR / "data"

FEATURES = ["distance_km", "stops_count", "departure_hour", "transport_type", "time_of_day", "day_type"]
TARGET = "travel_time_min"


def _metrics(y, p):
    return {
        "mae": round(float(mean_absolute_error(y, p)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y, p))), 3),
        "r2_score": round(float(r2_score(y, p)), 4),
    }


def _per_mode(df, preds):
    out = {}
    for mode, idx in df.groupby("transport_type").groups.items():
        y = df.loc[idx, TARGET]
        out[mode] = {"n": int(len(idx)), "mae": round(float(mean_absolute_error(y, preds.loc[idx])), 3)}
    return out


def _cross_validate(make_model, X, y, groups, k=5):
    scores = {"mae": [], "rmse": [], "r2_score": []}
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        m = make_model().fit(X.iloc[tr], y.iloc[tr])
        for key, val in _metrics(y.iloc[te], m.predict(X.iloc[te])).items():
            scores[key].append(val)
    return {f"{key}_mean": round(float(np.mean(v)), 4) for key, v in scores.items()} | \
           {f"{key}_std": round(float(np.std(v)), 4) for key, v in scores.items()}


def run_training_pipeline(test_size=0.2, random_state=42):
    print("=" * 60)
    print("  JOURNEY-TIME MODELS: TRAINING & EVALUATION")
    print("=" * 60)

    df = generate_travel_time_dataset(output_path=DATA_DIR / "travel_time_dataset.csv",
                                      random_state=random_state)
    print(df["transport_type"].value_counts().to_string())

    # ---- group split by route --------------------------------------------
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_idx, te_idx = next(gss.split(df, groups=df["route_id"]))
    train, test = df.iloc[tr_idx], df.iloc[te_idx]
    X_tr, y_tr, X_te, y_te = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]
    print(f"\nTrain: {len(train)} samples / {train.route_id.nunique()} routes | "
          f"Test: {len(test)} samples / {test.route_id.nunique()} unseen routes")

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    makers = {
        "linear_regression": LinearRegressionTravelTimeModel,
        "random_forest": lambda: RandomForestTravelTimeModel(random_state=random_state),
    }
    labels = {"linear_regression": "Multiple Linear Regression", "random_forest": "Random Forest Regressor"}

    # ---- baseline: timetable time with no congestion model ----------------
    base_pred = test["timetable_time_min"].where(test.transport_type != "walk",
                                                 test.distance_km / 4.5 * 60)
    results = {"baseline_timetable": {"model_type": "Timetable baseline (no ML)",
                                      **_metrics(y_te, base_pred),
                                      "per_mode": _per_mode(test, base_pred)}}

    sample = test.sample(min(250, len(test)), random_state=random_state)
    scatter = {"actual": sample[TARGET].round(2).tolist(),
               "transport_type": sample["transport_type"].tolist()}

    for name, make in makers.items():
        model = make().fit(X_tr, y_tr)
        preds = np.asarray(model.predict(X_te))
        preds_s = pd.Series(preds, index=test.index)
        res = {"model_type": labels[name], **_metrics(y_te, preds),
               "per_mode": _per_mode(test, preds_s),
               "cv_5fold": _cross_validate(make, X_tr, y_tr, train["route_id"])}
        if name == "random_forest":
            res["feature_importances"] = dict(list(model.get_feature_importances().items())[:10])
            res["n_estimators"], res["max_depth"] = model.n_estimators, model.max_depth
        else:
            res["coefficients"] = model.get_feature_coefficients()
        model.metrics = res
        model.save(SAVED_MODELS_DIR / f"{name}.joblib")
        results[name] = res
        scatter[name] = np.round(model.predict(sample[FEATURES]), 2).tolist()

    comparison = {
        "dataset_samples": len(df),
        "ride_segments": int(df.groupby(["route_id", "from_stop", "to_stop"]).ngroups),
        "train_samples": len(train), "test_samples": len(test),
        "split": "GroupShuffleSplit by route_id (test routes unseen in training)",
        "target_source": "simulated from timetable times + time-of-day congestion profile "
                         "(no open Chennai trip-time data)",
        "features": FEATURES,
        **results,
        "test_predictions_sample": scatter,
    }
    with open(SAVED_MODELS_DIR / "model_metrics.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n{'Model':<28}{'MAE':>8}{'RMSE':>8}{'R2':>8}   CV MAE (5-fold)")
    for key in ("baseline_timetable", "linear_regression", "random_forest"):
        r = results[key]
        cv = r.get("cv_5fold")
        cv_s = f"{cv['mae_mean']:.3f} +/- {cv['mae_std']:.3f}" if cv else "-"
        print(f"{r['model_type']:<28}{r['mae']:>8.3f}{r['rmse']:>8.3f}{r['r2_score']:>8.3f}   {cv_s}")
    print("\nPer-mode MAE (test):")
    for mode in results["random_forest"]["per_mode"]:
        print(f"  {mode:<6} " + "  ".join(f"{k[:4]}={results[k]['per_mode'][mode]['mae']:.2f}"
                                         for k in ("baseline_timetable", "linear_regression", "random_forest")))
    return comparison


if __name__ == "__main__":
    run_training_pipeline()
