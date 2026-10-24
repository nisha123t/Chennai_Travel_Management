# Chennai Route Planner API

Base URL: `http://localhost:8000`. Interactive docs: `http://localhost:8000/docs`.

Stops can be given as a `stop_id` (`S001`) or an exact stop name (`"Chennai Central"`).

## Stops

### `GET /stops`
All 140 stops: `stop_id`, `stop_name`, `area`, `region`, `latitude`, `longitude`, `stop_type`, `modes` (list), `interchange` (bool).

### `GET /stops/search?q=tamb&limit=10`
Case-insensitive search on name, area or id. Name-prefix matches come first.

## Routing

### `POST /routes`: up to 3 distinct route options
```json
{
  "start": "S001",
  "destination": "S058",
  "preference": "balanced",     // fastest | cheapest | least_walking | balanced | shortest
  "departure_hour": 18,         // 0-23
  "day_type": "weekday",        // weekday | weekend
  "max_options": 3,             // 1-5
  "ml_enhanced": true,          // rank and display using ML-predicted ride times
  "model": "random_forest"      // random_forest | linear_regression
}
```
Response:
```json
{
  "start": {"stop_id": "S001", "stop_name": "Broadway", "latitude": 13.0874, "longitude": 80.2838, "area": "George Town"},
  "destination": {"stop_id": "S058", "stop_name": "Chennai Airport", "...": "..."},
  "preference": "balanced", "departure_hour": 18, "day_type": "weekday",
  "ml_enhanced": true, "model_used": "random_forest",
  "candidates_considered": 24,
  "options": [
    {
      "option_id": "3f9c1a7b2e4d", "rank": 1,
      "tags": ["fastest", "least_walking"],          // best-in-set badges
      "summary": "MTC Bus → Metro",
      "total_time_min": 54.1,          // timetable ride time + boarding waits
      "ml_total_time_min": 61.54,       // ML-predicted ride time + boarding waits
      "in_vehicle_time_min": 43.1, "wait_time_min": 11.0,
      "total_fare_inr": 45.0, "total_distance_km": 19.06,
      "walking_distance_km": 0.0, "transfers": 1, "mode_changes": 1,
      "modes_used": ["bus", "metro"],
      "balanced_score": 0.0944,          // only for preference=balanced (lower is better)
      "segments": [
        {"transport_type": "bus", "mode_label": "MTC Bus", "route_id": "R01",
         "route_name": "MTC 18A Broadway - Tambaram",
         "from_stop": "S001", "from_name": "Broadway", "to_stop": "S014", "to_name": "Chennai Central",
         "num_stops": 1, "distance_km": 1.34, "time_min": 4.6, "wait_min": 7.0,
         "fare_inr": 5.0, "shares_ticket_with_previous": false,
         "ml_predicted_time_min": 7.98}
      ],
      "legs":  [ {"from_stop": "S001", "to_stop": "S014", "transport_type": "bus", "route_id": "R01",
                  "time_min": 4.6, "distance_km": 1.34, "fare_inr": 5.0, "...": "..."} ],
      "stops": [ {"stop_id": "S001", "stop_name": "Broadway", "latitude": 13.0874, "longitude": 80.2838, "area": "..."} ]
    }
  ]
}
```
Errors: `400` unknown stop (with suggestions), same start/destination; `404` no route; `422` invalid field values.

### `GET /routes/{option_id}`
Full details of an option returned by `POST /routes` (kept in memory for the last 500 options). `404` if expired.

### `POST /route` (deprecated)
The original single-route endpoint, kept for compatibility.

## Machine learning

### `POST /predict-time`
```json
{"distance_km": 12.5, "transport_type": "bus", "stops_count": 8, "departure_hour": 18,
 "day_type": "weekday", "model_type": "random_forest"}
```
Returns the request fields plus `predicted_travel_time_min`. `time_of_day` (`morning_peak`, `evening_peak`, `afternoon`, `off_peak`, `weekend_day`, `weekend_night`) can be sent instead of `departure_hour`.

### `GET /models/metrics`
Test-set MAE / RMSE / R² for the timetable baseline, Linear Regression and Random Forest, plus 5-fold CV, per-mode MAE, feature importances / coefficients and a sample of test predictions.

## How routing works

1. **Route-expanded graph.** Nodes are stations and "vehicle on route R at stop S". Boarding costs the expected wait (metro 4, rail 6, bus 7, auto 3, cab 6 min), alighting is free, and walk links join stations. Changing vehicles always means alight + board, so every transfer is valid and costed.
2. **Candidates.** Yen's K-shortest simple paths (`networkx.shortest_simple_paths`) is run under four costs: time, fare proxy, walking, and time with a transfer penalty. 12 paths per cost gives a pool of roughly 20–30 journeys.
3. **Exact scoring.** Real ticket rules: one CMRL token for a continuous metro ride, one suburban ticket for continuous rail, one MTC stage fare per bus route, auto/cab formulas, walking free.
4. **Ranking and diversity.** The pool is sorted by the preference. Balanced = 0.45·time + 0.30·fare + 0.10·walking + 0.15·transfers, each min-max normalised. Options sharing more than 50% of their hops are skipped (relaxed to 80% if needed to fill 3 slots).

## ML model

Training samples are continuous ride segments (about 3,200 of them, 4 departure times each) built from `stop_sequences.csv` + `connections.csv`.
Features: `distance_km`, `stops_count`, `departure_hour`, `transport_type`, `time_of_day`, `day_type`.
The target (observed ride time) is **simulated** from timetable times with a time-of-day congestion profile, mode sensitivity, dwell time and noise, because no open Chennai trip-time data exists.
The test split is by route (`GroupShuffleSplit`), so test routes are unseen in training.

Random Forest hyperparameters are tuned with `GridSearchCV` (36 settings, 3-fold GroupKFold by route, training data only); results are in `rf_tuning` in `/models/metrics`.

Retrain: `cd backend/models && python train_models.py` (add `--no-tune` to skip tuning)
