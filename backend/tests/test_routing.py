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
    # Include Chennai, Chengalpattu, Kanchipuram, and nearby ECR corridors.
    assert s.latitude.between(12.2, 13.6).all() and s.longitude.between(79.5, 80.8).all()


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


def test_freeform_place_name_geocodes_to_nearest_stop(monkeypatch):
    def fake_geocode(query):
        if "vit" in query.lower():
            return {"latitude": 12.849, "longitude": 80.161, "display_name": "VIT Chennai, Tamil Nadu"}
        return {"latitude": 13.078, "longitude": 80.261, "display_name": "Egmore Museum, Chennai"}

    monkeypatch.setattr(eng, "geocode_place", fake_geocode)
    resolved = eng.resolve_stop("VIT Chennai")
    assert resolved in eng.STOPS
    route = eng.plan_journeys("VIT Chennai", "Egmore Museum", ml_enhanced=False)
    assert route["start"]["stop_id"] == resolved
    assert route["destination"]["stop_id"] == "S025"


def test_same_station_place_destination_is_shown_as_last_mile_walk(monkeypatch):
    monkeypatch.setattr(
        eng,
        "geocode_place",
        lambda query: {"latitude": 13.064, "longitude": 80.243, "display_name": "Valluvar Kottam, Chennai"},
    )
    result = eng.plan_journeys("Nungambakkam", "Valluvar Kottam", ml_enhanced=False)
    option = result["options"][0]
    last_mile = option["segments"][-1]

    assert result["destination"]["stop_id"] == "S043"
    assert result["destination"]["access_mode"] == "walk"
    assert last_mile["access_leg"] is True
    assert last_mile["from_name"] == "Nungambakkam"
    assert last_mile["to_name"] == "Valluvar Kottam"
    assert last_mile["distance_km"] == pytest.approx(0.56)
    assert last_mile["wait_min"] == 0
    assert option["transfers"] == 0
    assert option["walking_distance_km"] == pytest.approx(0.56)


@pytest.mark.parametrize("distance,expected_mode", [(0.8, "walk"), (2.0, "auto"), (5.5, "cab")])
def test_geocoded_access_mode_uses_distance_bands(monkeypatch, distance, expected_mode):
    monkeypatch.setattr(eng, "geocode_place", lambda query: {"latitude": 13.0, "longitude": 80.0})
    monkeypatch.setattr(
        eng,
        "nearest_stop_for_coordinates",
        lambda latitude, longitude: {
            "stop_id": "S043", "stop_name": "Nungambakkam", "area": "Nungambakkam", "distance_km": distance,
        },
    )
    assert eng.resolve_location("Somewhere in Chennai")["access_mode"] == expected_mode


def test_manual_vandalur_kelambakkam_corridor_stops_are_present(data):
    required = {
        "Kelambakkam Road Junction",
        "Samiyar Pannai",
        "Chettinad Hospital",
        "Pudupakkam",
        "Anjaneyar Temple Pudupakkam",
        "Mambakkam Road Junction",
        "Melakottaiyur",
        "Kandigai",
        "Vengambakkam Road Junction",
        "Kolapakkam / Nallambakkam",
        "Anna Nagar Colony",
        "Vandalur Zoo",
        "VIT Chennai",
    }
    names = {stop["stop_name"] for stop in eng.list_stops()}
    assert required.issubset(names)
    assert eng.resolve_stop("VIT Chennai") in eng.STOPS

    sequences = data["stop_sequences"]
    outbound = sequences[sequences.route_id == "MTC_555G_MANUAL_K2V"].sort_values("sequence_no")
    inbound = sequences[sequences.route_id == "MTC_555G_MANUAL_V2K"].sort_values("sequence_no")
    expected = ["S141", "S142", "S148", "S143", "S149", "S144", "S147", "S150",
                "S145", "S146", "S151", "S152", "S136"]
    assert outbound.stop_id.tolist() == expected
    assert inbound.stop_id.tolist() == expected[::-1]


def test_vit_to_vandalur_uses_one_continuous_555g_ride():
    result = eng.plan_journeys("VIT Chennai", "Vandalur Zoo", "fastest", ml_enhanced=False)
    corridor_segments = [
        segment
        for option in result["options"]
        for segment in option["segments"]
        if segment["route_id"] == "MTC_555G_MANUAL_K2V"
    ]
    assert corridor_segments
    assert any(
        segment["from_stop"] == "S147"
        and segment["to_stop"] == "S136"
        and segment["num_stops"] == 6
        and segment["wait_min"] == eng.BOARDING_WAIT_MIN["bus"]
        for segment in corridor_segments
    )


def test_manual_vit_corridor_distances_match_stop_coordinates(data):
    corridor_route_ids = {"MTC_555G_MANUAL_K2V", "MTC_555G_MANUAL_V2K"}
    connections = data["connections"][data["connections"].route_id.isin(corridor_route_ids)]
    stops = data["stops"].set_index("stop_id")
    assert len(connections) == 24
    for connection in connections.itertuples(index=False):
        origin = stops.loc[connection.from_stop]
        destination = stops.loc[connection.to_stop]
        minimum_km = eng.haversine_km(
            origin.latitude, origin.longitude,
            destination.latitude, destination.longitude,
        )
        assert connection.distance_km >= minimum_km


# ------------------------------------------------------------------ ML
def test_peak_hour_slower_than_night_for_bus():
    from predictor import predict_travel_time
    peak = predict_travel_time(10, "bus", stops_count=6, departure_hour=18)
    night = predict_travel_time(10, "bus", stops_count=6, departure_hour=3)
    assert peak > night


def test_metrics_file_has_comparison():
    import json
    m = json.loads((BACKEND / "models" / "saved_models" / "model_metrics.json").read_text())
    for k in ("linear_regression", "random_forest", "xgboost", "baseline_timetable"):
        assert {"mae", "rmse", "r2_score"} <= set(m[k])
    assert m["random_forest"]["mae"] < m["baseline_timetable"]["mae"]


def test_xgboost_model_is_available():
    from predictor import predict_travel_time
    pred = predict_travel_time(10, "bus", stops_count=6, departure_hour=18, model_type="xgboost")
    assert pred > 0


def test_dynamic_model_selection_uses_supported_model():
    r = eng.plan_journeys("S001", "S058", "balanced", ml_enhanced=True, model_type="dynamic_model_selection")
    assert r["model_used"] in {"random_forest", "linear_regression", "xgboost"}
    assert r["journey_context"]["selection_mode"] == "dynamic_model_selection"
    assert "bandit_details" in r["journey_context"]
    assert "overhead_reduction_pct" in r["journey_context"]["bandit_details"]["legacy_comparison"]
    assert r["journey_context"]["bandit_details"]["legacy_comparison"]["overhead_reduction_pct"] > 50.0


def test_contextual_bandit_selection_logic():
    from predictor import select_model_for_context, get_bandit_stats
    model, details = select_model_for_context(15.0, "bus", departure_hour=18, day_type="weekday", weather="clear", return_details=True)
    assert model in {"random_forest", "xgboost", "linear_regression"}
    assert details["selected_model"] == model
    assert details["algorithm"] == "Contextual Multi-Armed Bandit (UCB1 + Domain Priors)"
    assert "arm_scores" in details
    assert "selection_reason" in details
    stats = get_bandit_stats()
    assert stats["total_selections"] > 0
    assert "benchmark_comparison" in stats



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


def test_weather_increases_journey_duration():
    dry = eng.plan_journeys("S001", "S058", "balanced", ml_enhanced=False, weather="clear")
    wet = eng.plan_journeys("S001", "S058", "balanced", ml_enhanced=False, weather="rain")
    assert wet["journey_context"]["weather"] == "rain"
    assert wet["options"][0]["total_time_min"] > dry["options"][0]["total_time_min"]


def test_api_accepts_weather_field(client):
    payload = {
        "start": "S001",
        "destination": "S058",
        "preference": "balanced",
        "weather": "rain",
    }
    r = client.post("/routes", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["journey_context"]["weather"] == "rain"


def test_api_errors(client):
    assert client.post("/routes", json={"start": "S001", "destination": "S001"}).status_code == 400
    assert client.post("/routes", json={"start": "nowhere", "destination": "S001"}).status_code == 400
    assert client.post("/routes", json={"start": "S001", "destination": "S002", "departure_hour": 30}).status_code == 422
    assert client.post("/routes", json={"start": "S001"}).status_code == 422
    assert client.get("/routes/doesnotexist").status_code == 404


def test_api_search_and_metrics(client):
    assert client.get("/stops/search", params={"q": "central"}).json()
    assert client.get("/stops").status_code == 200
    metrics = client.get("/models/metrics").json()
    assert "dynamic_selection" in metrics
    bandit = client.get("/models/bandit-stats").json()
    assert bandit["algorithm"].startswith("Contextual Multi-Armed Bandit")
    r = client.post("/predict-time", json={"distance_km": 12.5, "transport_type": "bus", "departure_hour": 18, "model_type": "dynamic_model_selection"})
    assert r.status_code == 200 and r.json()["predicted_travel_time_min"] > 0
    route_req = {
        "start": "S001",
        "destination": "S058",
        "preference": "fastest",
        "model": "dynamic_model_selection",
        "ml_enhanced": True,
    }
    route_res = client.post("/routes", json=route_req)
    assert route_res.status_code == 200
    body = route_res.json()
    assert body["journey_context"]["selection_mode"] == "dynamic_model_selection"
    assert "bandit_details" in body["journey_context"]

