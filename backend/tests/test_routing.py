"""
Automated tests for data integrity, fares, multi-route planning, ML and the API.

Run from backend/:   python -m pytest -q
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "models"))

import route_engine as eng  # noqa: E402
from router import calc_bus_fare, calc_metro_fare, calc_rail_fare  # noqa: E402

DATA = BACKEND / "data"


# ------------------------------------------------------------------ data
@pytest.fixture(scope="module")
def data():
    return {n: pd.read_csv(DATA / f"{n}.csv") for n in ("stops", "routes", "stop_sequences", "connections")}


def test_ids_unique(data):
    assert not data["stops"].stop_id.duplicated().any()
    assert not data["routes"].route_id.duplicated().any()
    assert not data["connections"].connection_id.duplicated().any()


def test_foreign_keys_valid(data):
    stop_ids, route_ids = set(data["stops"].stop_id), set(data["routes"].route_id)
    c = data["connections"]
    assert c.from_stop.isin(stop_ids).all() and c.to_stop.isin(stop_ids).all()
    assert c.route_id.isin(route_ids).all()
    assert data["stop_sequences"].stop_id.isin(stop_ids).all()


def test_values_sane(data):
    c, s = data["connections"], data["stops"]
    assert (c.distance_km > 0).all() and (c.estimated_time_min > 0).all()
    assert (c.fare_inr >= 0).all()
    assert c.transport_type.isin(["metro", "rail", "bus", "walk", "auto", "cab"]).all()
    # every stop lies inside greater Chennai
    assert s.latitude.between(12.6, 13.5).all() and s.longitude.between(79.8, 80.4).all()


def test_network_connected():
    import networkx as nx
    stations = [n for n in eng.GRAPH if n[0] == "S"]
    comp = max(nx.strongly_connected_components(eng.GRAPH), key=len)
    assert all(s in comp for s in stations), "some stops cannot reach / be reached from the rest"


# ------------------------------------------------------------------ fares
@pytest.mark.parametrize("km,fare", [(1.5, 10), (2.0, 10), (4.9, 20), (11, 30), (17.5, 40), (25, 50)])
def test_metro_fare_slabs(km, fare):
    assert calc_metro_fare(km) == fare


@pytest.mark.parametrize("km,fare", [(3, 5), (10, 10), (20, 15), (28, 20), (31.4, 25), (50, 30)])
def test_rail_fare_slabs(km, fare):
    assert calc_rail_fare(km) == fare


def test_bus_fare_monotonic_and_capped():
    fares = [calc_bus_fare(k) for k in range(1, 60)]
    assert fares[0] == 5 and max(fares) == 25
    assert all(a <= b for a, b in zip(fares, fares[1:]))


# ------------------------------------------------------------------ planner
@pytest.fixture(scope="module")
def plan():
    return eng.plan_journeys("S001", "S058", "balanced", ml_enhanced=False)


def test_returns_up_to_three_distinct_options(plan):
    opts = plan["options"]
    assert 1 <= len(opts) <= 3
    signatures = {tuple((s["transport_type"], s["route_id"], s["from_stop"]) for s in o["segments"]) for o in opts}
    assert len(signatures) == len(opts), "options must not repeat the same path"


def test_options_are_valid_journeys(plan):
    for o in plan["options"]:
        legs = o["legs"]
        assert legs[0]["from_stop"] == "S001" and legs[-1]["to_stop"] == "S058"
        for a, b in zip(legs, legs[1:]):
            assert a["to_stop"] == b["from_stop"], "legs must chain"
        # consistent totals
        assert o["total_fare_inr"] == pytest.approx(sum(s["fare_inr"] for s in o["segments"]), abs=0.05)
        assert o["total_distance_km"] == pytest.approx(sum(l["distance_km"] for l in legs), abs=0.05)
        assert o["walking_distance_km"] == pytest.approx(
            sum(s["distance_km"] for s in o["segments"] if s["transport_type"] == "walk"), abs=0.01)
        rides = [s for s in o["segments"] if s["transport_type"] != "walk"]
        assert o["transfers"] == max(0, len(rides) - 1)
        assert o["total_time_min"] == pytest.approx(o["in_vehicle_time_min"] + o["wait_time_min"], abs=0.05)


@pytest.mark.parametrize("pref,key", [("fastest", "total_time_min"), ("cheapest", "total_fare_inr"),
                                      ("least_walking", "walking_distance_km")])
def test_preference_puts_best_first(pref, key):
    r = eng.plan_journeys("S062", "S001", pref, ml_enhanced=False)
    vals = [o[key] for o in r["options"]]
    assert vals[0] == min(vals)


def test_single_ticket_for_continuous_metro():
    # Chennai Central -> Airport is one continuous metro ride: one CMRL token
    r = eng.plan_journeys("Chennai Central", "S058", "fastest", ml_enhanced=False, max_options=5)
    metro_only = [o for o in r["options"] if o["modes_used"] == ["metro"]]
    if metro_only:
        o = metro_only[0]
        assert o["total_fare_inr"] == calc_metro_fare(o["total_distance_km"])


def test_ml_enhanced_adds_predictions():
    r = eng.plan_journeys("S001", "S058", "fastest", departure_hour=18, ml_enhanced=True)
    for o in r["options"]:
        assert "ml_total_time_min" in o and o["ml_total_time_min"] > 0
        assert all("ml_predicted_time_min" in s for s in o["segments"])


def test_option_details_cached(plan):
    oid = plan["options"][0]["option_id"]
    assert eng.get_option(oid)["option_id"] == oid


@pytest.mark.parametrize("kwargs,msg", [
    (dict(start="S001", destination="S001"), "different"),
    (dict(start="S999", destination="S001"), "Unknown stop"),
    (dict(start="S001", destination="S002", preference="scenic"), "preference"),
    (dict(start="S001", destination="S002", departure_hour=25), "departure_hour"),
    (dict(start="S001", destination="S002", day_type="holiday"), "day_type"),
])
def test_invalid_input_raises(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        eng.plan_journeys(**kwargs)


def test_stop_search():
    hits = eng.search_stops("tamb")
    assert hits and hits[0]["stop_name"].lower().startswith("tamb")
    assert eng.search_stops("zzzz") == []
    assert eng.resolve_stop("Chennai Central") == eng.search_stops("Chennai Central")[0]["stop_id"]


# ------------------------------------------------------------------ ML
def test_peak_hour_slower_than_night_for_bus():
    from predictor import predict_travel_time
    peak = predict_travel_time(10, "bus", stops_count=6, departure_hour=18)
    night = predict_travel_time(10, "bus", stops_count=6, departure_hour=3)
    assert peak > night


def test_metrics_file_has_comparison():
    import json
    m = json.loads((BACKEND / "models" / "saved_models" / "model_metrics.json").read_text())
    for k in ("linear_regression", "random_forest", "baseline_timetable"):
        assert {"mae", "rmse", "r2_score"} <= set(m[k])
    assert m["random_forest"]["mae"] < m["baseline_timetable"]["mae"]


# ------------------------------------------------------------------ API
fastapi = pytest.importorskip("fastapi")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)


def test_api_routes(client):
    r = client.post("/routes", json={"start": "S001", "destination": "S058", "preference": "fastest"})
    assert r.status_code == 200
    body = r.json()
    assert 1 <= len(body["options"]) <= 3
    oid = body["options"][0]["option_id"]
    assert client.get(f"/routes/{oid}").status_code == 200


def test_api_errors(client):
    assert client.post("/routes", json={"start": "S001", "destination": "S001"}).status_code == 400
    assert client.post("/routes", json={"start": "nowhere", "destination": "S001"}).status_code == 400
    assert client.post("/routes", json={"start": "S001", "destination": "S002", "departure_hour": 30}).status_code == 422
    assert client.post("/routes", json={"start": "S001"}).status_code == 422
    assert client.get("/routes/doesnotexist").status_code == 404


def test_api_search_and_metrics(client):
    assert client.get("/stops/search", params={"q": "central"}).json()
    assert client.get("/stops").status_code == 200
    assert client.get("/models/metrics").status_code == 200
    r = client.post("/predict-time", json={"distance_km": 12.5, "transport_type": "bus", "departure_hour": 18})
    assert r.status_code == 200 and r.json()["predicted_travel_time_min"] > 0
