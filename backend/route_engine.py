"""
Multi-route journey planner for the Chennai multimodal network.

How it works
------------
1. Route-expanded graph (valid multimodal transfers)
   Nodes are either a *station* ("S", stop) or a *vehicle on a route at a
   stop* ("R", stop, route_id).  Edges:
     station -> vehicle   board   (costs the expected wait for that mode)
     vehicle -> vehicle   ride    (along the route's own connection)
     vehicle -> station   alight  (free)
     station -> station   walk    (pedestrian transfer links)
   Because changing vehicle always means alight + board, a transfer has a real
   cost and a path can never "teleport" between routes.

2. Candidate generation - Yen's K-shortest simple paths
   (networkx.shortest_simple_paths) is run under three cost functions
   (time, fare proxy, walking), giving a diverse pool of feasible journeys.

3. Exact evaluation
   Every candidate is turned into rides/segments and scored with the real
   ticketing rules (one CMRL token for a continuous metro ride, one suburban
   ticket for a continuous rail ride, one MTC stage fare per bus route, auto /
   cab formulas), walking distance, waiting time, transfers and, optionally,
   ML-predicted ride times.

4. Ranking + diversity
   Candidates are ranked for the requested preference (fastest, cheapest,
   least_walking, balanced) and up to `max_options` *distinct* journeys are
   returned (near-duplicates sharing most of their hops are skipped).
"""
from __future__ import annotations

import itertools
import json
import math
import uuid
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import networkx as nx
import pandas as pd

from models.predictor import select_model_for_context
from router import calc_segment_fare

DATA = Path(__file__).parent / "data"

PREFERENCES = ("fastest", "cheapest", "least_walking", "balanced", "shortest")
RIDE_MODES = ("metro", "rail", "bus", "auto", "cab")
WEATHER_FACTORS = {
    "clear": 1.0,
    "cloudy": 1.05,
    "rain": 1.18,
    "storm": 1.30,
    "heat": 1.12,
    "fog": 1.08,
}

# Expected wait when boarding (approx. half the typical headway), minutes.
BOARDING_WAIT_MIN = {"metro": 4.0, "rail": 6.0, "bus": 7.0, "auto": 3.0, "cab": 6.0}
# Fare proxy used only to *generate* cheap candidates; exact fares are computed later.
FARE_PROXY_BOARD = {"metro": 10.0, "rail": 5.0, "bus": 5.0, "auto": 40.0, "cab": 100.0}
FARE_PROXY_PER_KM = {"metro": 2.0, "rail": 0.5, "bus": 0.7, "auto": 18.0, "cab": 20.0}

CANDIDATES_PER_CRITERION = 12
# Jaccard hop overlap allowed between returned journeys: strict first, relaxed if
# the strict pass cannot fill all slots.
DIVERSITY_PASSES = (0.5, 0.8)

MODE_LABEL = {"metro": "Metro", "rail": "Suburban Rail", "bus": "MTC Bus",
              "walk": "Walk", "auto": "Auto", "cab": "Cab"}


# --------------------------------------------------------------------------
# Data + graph
# --------------------------------------------------------------------------
stops_df = pd.read_csv(DATA / "stops.csv")
routes_df = pd.read_csv(DATA / "routes.csv")
connections_df = pd.read_csv(DATA / "connections.csv")

STOPS = {r.stop_id: r._asdict() for r in stops_df.itertuples(index=False)}
ROUTE_NAMES = dict(zip(routes_df.route_id, routes_df.route_name))


def _build_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    for sid in STOPS:
        g.add_node(("S", sid))

    for c in connections_df.itertuples(index=False):
        mode = str(c.transport_type).lower()
        hop = {"mode": mode, "route_id": c.route_id, "from_stop": c.from_stop,
               "to_stop": c.to_stop, "distance_km": float(c.distance_km),
               "time_min": float(c.estimated_time_min)}
        if mode == "walk":
            g.add_edge(("S", c.from_stop), ("S", c.to_stop), kind="walk", hop=hop,
                       w_time=hop["time_min"], w_xfer=hop["time_min"], w_fare=0.05 * hop["time_min"],
                       w_walk=hop["distance_km"] * 100 + hop["time_min"])
            continue

        u, v = ("R", c.from_stop, c.route_id), ("R", c.to_stop, c.route_id)
        wait = BOARDING_WAIT_MIN.get(mode, 5.0)
        for stop, node in ((c.from_stop, u), (c.to_stop, v)):
            if not g.has_edge(("S", stop), node):
                g.add_edge(("S", stop), node, kind="board", mode=mode,
                           w_time=wait, w_xfer=wait + 15.0, w_fare=FARE_PROXY_BOARD.get(mode, 10.0) + 0.05 * wait,
                           w_walk=wait)
                g.add_edge(node, ("S", stop), kind="alight", w_time=0.0, w_xfer=0.0, w_fare=0.0, w_walk=0.0)
        g.add_edge(u, v, kind="ride", hop=hop, w_time=hop["time_min"], w_xfer=hop["time_min"],
                   w_fare=FARE_PROXY_PER_KM.get(mode, 1.0) * hop["distance_km"] + 0.05 * hop["time_min"],
                   w_walk=hop["time_min"])
    return g


GRAPH = _build_graph()


# --------------------------------------------------------------------------
# Stop helpers
# --------------------------------------------------------------------------
def stop_info(stop_id: str) -> dict:
    s = STOPS[stop_id]
    return {"stop_id": stop_id, "stop_name": s["stop_name"], "area": s["area"],
            "latitude": float(s["latitude"]), "longitude": float(s["longitude"])}


def list_stops() -> list[dict]:
    out = []
    for sid, s in STOPS.items():
        out.append({**stop_info(sid), "region": s["region"], "stop_type": s["stop_type"],
                    "modes": str(s["modes"]).split(";"), "interchange": s["interchange"] == "yes"})
    return out


def search_stops(query: str, limit: int = 10) -> list[dict]:
    """Case-insensitive search on stop name / area / id. Prefix matches rank first."""
    q = query.strip().lower()
    if not q:
        return []
    scored = []
    for item in list_stops():
        name, area = item["stop_name"].lower(), str(item["area"]).lower()
        if name.startswith(q):
            rank = 0
        elif any(w.startswith(q) for w in name.split()):
            rank = 1
        elif q in name:
            rank = 2
        elif q in area or q == item["stop_id"].lower():
            rank = 3
        else:
            continue
        scored.append((rank, len(name), item))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in scored[:limit]]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def geocode_place(query: str) -> dict | None:
    """Geocode a free-form place query to latitude / longitude using OpenStreetMap Nominatim."""
    q = (query or "").strip()
    if not q:
        return None
    params = {
        "q": f"{q}, Chennai, Tamil Nadu, India",
        "format": "jsonv2",
        "limit": 5,
        "addressdetails": 1,
        "bounded": 1,
        "viewbox": "79.5,13.6,80.8,12.2",
    }
    req = Request(
        f"https://nominatim.openstreetmap.org/search?{urlencode(params)}",
        headers={"User-Agent": "ChennaiRoutePlanner/1.0"},
    )
    try:
        with urlopen(req, timeout=8) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    if not body:
        return None
    first = body[0]
    try:
        return {
            "latitude": float(first["lat"]),
            "longitude": float(first["lon"]),
            "display_name": first.get("display_name", q),
        }
    except (KeyError, TypeError, ValueError):
        return None


def nearest_stop_for_coordinates(latitude: float, longitude: float) -> dict | None:
    best = None
    best_distance = float("inf")
    for sid, s in STOPS.items():
        stop_lat = float(s["latitude"])
        stop_lon = float(s["longitude"])
        distance_km = haversine_km(latitude, longitude, stop_lat, stop_lon)
        if distance_km < best_distance:
            best_distance = distance_km
            best = {"stop_id": sid, "stop_name": s["stop_name"], "area": s["area"], "distance_km": distance_km}
    return best


def resolve_stop(value: str) -> str:
    """Accept a stop_id, exact name, or a free-form place name resolved to the nearest stop."""
    return resolve_location(value)["stop_id"]


def resolve_location(value: str) -> dict:
    """Resolve a free-form place string to a stop plus the mode/distance for the last-mile access leg."""
    if value in STOPS:
        return {"stop_id": value, "location_type": "stop", "access_mode": None, "access_distance_km": 0.0}

    v = value.strip().lower()
    exact = [sid for sid, s in STOPS.items() if s["stop_name"].lower() == v]
    if len(exact) == 1:
        return {"stop_id": exact[0], "location_type": "stop", "access_mode": None, "access_distance_km": 0.0}

    hits = search_stops(value, limit=5)
    if len(hits) == 1:
        sid = hits[0]["stop_id"]
        return {"stop_id": sid, "location_type": "stop", "access_mode": None, "access_distance_km": 0.0}

    geocode = geocode_place(value)
    if geocode is not None:
        nearest = nearest_stop_for_coordinates(geocode["latitude"], geocode["longitude"])
        if nearest is not None:
            dist = nearest["distance_km"]
            if dist < 1.0:
                access_mode = "walk"
            elif dist <= 5.0:
                access_mode = "auto"
            else:
                access_mode = "cab"
            return {
                "stop_id": nearest["stop_id"],
                "location_type": "geocoded",
                "access_mode": access_mode,
                "access_distance_km": round(dist, 2),
                "source_query": value,
                "display_name": geocode.get("display_name", value),
                "access_latitude": geocode["latitude"],
                "access_longitude": geocode["longitude"],
            }

    hint = ", ".join(f"{h['stop_name']} ({h['stop_id']})" for h in hits)
    raise ValueError(f"Unknown stop '{value}'." + (f" Did you mean: {hint}?" if hint else ""))


# --------------------------------------------------------------------------
# Path -> journey
# --------------------------------------------------------------------------
def _path_to_rides(path: list) -> list[dict]:
    """Collapse a node path into rides: consecutive hops on one vehicle (or one walk)."""
    rides, current = [], None
    for a, b in zip(path, path[1:]):
        e = GRAPH.edges[a, b]
        if e["kind"] not in ("ride", "walk"):
            if e["kind"] == "board":
                current = None  # next ride hop starts a new vehicle
            continue
        hop = e["hop"]
        if current is None or e["kind"] == "walk" or current["mode"] == "walk":
            current = {"mode": hop["mode"], "route_id": hop["route_id"], "hops": []}
            rides.append(current)
        current["hops"].append(hop)
    return rides


def _fare_groups(rides: list[dict]) -> list[list[dict]]:
    """Group rides that share one ticket: continuous metro or continuous rail."""
    groups = []
    for r in rides:
        if groups and r["mode"] in ("metro", "rail") and groups[-1][-1]["mode"] == r["mode"]:
            groups[-1].append(r)
        else:
            groups.append([r])
    return groups


def _evaluate(path: list) -> dict:
    rides = _path_to_rides(path)
    # merge consecutive walk rides into one walking segment
    merged = []
    for r in rides:
        if merged and r["mode"] == "walk" and merged[-1]["mode"] == "walk":
            merged[-1]["hops"].extend(r["hops"])
        else:
            merged.append(r)
    rides = merged

    segments, legs = [], []
    total_fare = wait_total = 0.0
    for group in _fare_groups(rides):
        group_dist = sum(h["distance_km"] for r in group for h in r["hops"])
        mode = group[0]["mode"]
        ticket = calc_segment_fare(mode, group_dist)
        total_fare += ticket
        for idx, r in enumerate(group):
            dist = sum(h["distance_km"] for h in r["hops"])
            time = sum(h["time_min"] for h in r["hops"])
            wait = 0.0 if mode == "walk" else BOARDING_WAIT_MIN.get(mode, 5.0)
            wait_total += wait
            seg_fare = ticket * (dist / group_dist) if group_dist else ticket / len(group)
            seg = {
                "transport_type": mode,
                "mode_label": MODE_LABEL.get(mode, mode.title()),
                "route_id": r["route_id"],
                "route_name": "Walk" if mode == "walk" else ROUTE_NAMES.get(r["route_id"], r["route_id"]),
                "from_stop": r["hops"][0]["from_stop"],
                "from_name": STOPS[r["hops"][0]["from_stop"]]["stop_name"],
                "to_stop": r["hops"][-1]["to_stop"],
                "to_name": STOPS[r["hops"][-1]["to_stop"]]["stop_name"],
                "num_stops": len(r["hops"]),
                "distance_km": round(dist, 2),
                "time_min": round(time, 2),
                "wait_min": wait,
                "fare_inr": round(seg_fare, 2),
                "shares_ticket_with_previous": idx > 0,
            }
            segments.append(seg)
            for h in r["hops"]:
                legs.append({
                    "from_stop": h["from_stop"], "from_name": STOPS[h["from_stop"]]["stop_name"],
                    "to_stop": h["to_stop"], "to_name": STOPS[h["to_stop"]]["stop_name"],
                    "transport_type": mode, "route_id": h["route_id"],
                    "time_min": h["time_min"], "distance_km": h["distance_km"],
                    "fare_inr": round(seg_fare * (h["distance_km"] / dist), 2) if dist else 0.0,
                })

    vehicle_segments = [s for s in segments if s["transport_type"] != "walk"]
    mode_seq = [s["transport_type"] for s in segments]
    stop_ids = [segments[0]["from_stop"]] + [l["to_stop"] for l in legs] if segments else []
    in_vehicle = sum(s["time_min"] for s in segments)
    return {
        "segments": segments,
        "legs": legs,
        "stops": [stop_info(s) for s in stop_ids],
        "modes_used": list(dict.fromkeys(m for m in mode_seq)),
        "total_time_min": round(in_vehicle + wait_total, 2),
        "in_vehicle_time_min": round(in_vehicle, 2),
        "wait_time_min": round(wait_total, 2),
        "total_fare_inr": round(total_fare, 2),
        "total_distance_km": round(sum(s["distance_km"] for s in segments), 2),
        "walking_distance_km": round(sum(s["distance_km"] for s in segments if s["transport_type"] == "walk"), 2),
        "transfers": max(0, len(vehicle_segments) - 1),
        "mode_changes": sum(1 for a, b in zip(mode_seq, mode_seq[1:]) if a != b),
        "_hops": {(l["from_stop"], l["to_stop"]) for l in legs},
        "_signature": tuple((s["transport_type"], s["route_id"], s["from_stop"], s["to_stop"]) for s in segments),
    }


def _access_leg(meta: dict, stop_id: str, endpoint: str) -> tuple[dict, dict] | None:
    mode = meta.get("access_mode")
    if meta.get("location_type") != "geocoded" or mode not in ("walk", "auto", "cab"):
        return None

    distance = float(meta["access_distance_km"])
    place_id = f"__place_{endpoint}__"
    stop_name = STOPS[stop_id]["stop_name"]
    place_name = meta.get("source_query") or meta.get("display_name") or "Your destination"
    is_origin = endpoint == "start"
    from_stop, to_stop = (place_id, stop_id) if is_origin else (stop_id, place_id)
    from_name, to_name = (place_name, stop_name) if is_origin else (stop_name, place_name)
    speed_kmh = {"walk": 5.0, "auto": 18.0, "cab": 25.0}[mode]
    travel_time = distance / speed_kmh * 60.0
    wait_time = 0.0 if mode == "walk" else BOARDING_WAIT_MIN[mode]
    fare = calc_segment_fare(mode, distance)
    place = {
        "stop_id": place_id,
        "stop_name": place_name,
        "area": "Place",
        "latitude": float(meta["access_latitude"]),
        "longitude": float(meta["access_longitude"]),
        "is_place": True,
    }
    segment = {
        "transport_type": mode,
        "mode_label": MODE_LABEL[mode],
        "route_id": None,
        "route_name": f"Last-mile {MODE_LABEL[mode].lower()}",
        "from_stop": from_stop,
        "from_name": from_name,
        "to_stop": to_stop,
        "to_name": to_name,
        "num_stops": 1,
        "distance_km": round(distance, 2),
        "time_min": round(travel_time, 2),
        "wait_min": wait_time,
        "fare_inr": round(fare, 2),
        "shares_ticket_with_previous": False,
        "access_leg": True,
        "access_direction": endpoint,
    }
    leg = {
        "from_stop": from_stop,
        "from_name": from_name,
        "to_stop": to_stop,
        "to_name": to_name,
        "transport_type": mode,
        "route_id": None,
        "time_min": segment["time_min"],
        "distance_km": segment["distance_km"],
        "fare_inr": segment["fare_inr"],
    }
    return segment, {"leg": leg, "place": place}


def _add_access_legs(journey: dict, start_meta: dict, dest_meta: dict, start: str, destination: str) -> dict:
    start_access = _access_leg(start_meta, start, "start")
    dest_access = _access_leg(dest_meta, destination, "destination")
    accesses = [("start", start_access), ("destination", dest_access)]
    accesses = [(direction, access) for direction, access in accesses if access is not None]
    if not accesses:
        return journey

    start_segment = next((segment for direction, (segment, _) in accesses if direction == "start"), None)
    dest_segment = next((segment for direction, (segment, _) in accesses if direction == "destination"), None)
    start_details = next((details for direction, (_, details) in accesses if direction == "start"), None)
    dest_details = next((details for direction, (_, details) in accesses if direction == "destination"), None)

    journey["segments"] = ([start_segment] if start_segment else []) + journey["segments"] + ([dest_segment] if dest_segment else [])
    journey["legs"] = ([start_details["leg"]] if start_details else []) + journey["legs"] + ([dest_details["leg"]] if dest_details else [])
    if start_details:
        journey["stops"].insert(0, start_details["place"])
    if dest_details:
        journey["stops"].append(dest_details["place"])

    for segment in (start_segment, dest_segment):
        if segment is None:
            continue
        journey["total_time_min"] += segment["time_min"] + segment["wait_min"]
        journey["in_vehicle_time_min"] += segment["time_min"]
        journey["wait_time_min"] += segment["wait_min"]
        journey["total_fare_inr"] += segment["fare_inr"]
        journey["total_distance_km"] += segment["distance_km"]
        if segment["transport_type"] == "walk":
            journey["walking_distance_km"] += segment["distance_km"]

    for key in ("total_time_min", "in_vehicle_time_min", "wait_time_min", "total_fare_inr",
                "total_distance_km", "walking_distance_km"):
        journey[key] = round(journey[key], 2)
    journey["modes_used"] = list(dict.fromkeys(segment["transport_type"] for segment in journey["segments"]))
    journey["mode_changes"] = sum(
        first["transport_type"] != second["transport_type"]
        for first, second in zip(journey["segments"], journey["segments"][1:])
    )
    journey["transfers"] = max(0, sum(segment["transport_type"] != "walk" for segment in journey["segments"]) - 1)
    return journey


def _empty_journey(stop_id: str) -> dict:
    return {
        "segments": [], "legs": [], "stops": [stop_info(stop_id)], "modes_used": [],
        "total_time_min": 0.0, "in_vehicle_time_min": 0.0, "wait_time_min": 0.0,
        "total_fare_inr": 0.0, "total_distance_km": 0.0, "walking_distance_km": 0.0,
        "transfers": 0, "mode_changes": 0, "_hops": set(), "_signature": (),
    }


def _resolved_endpoint(meta: dict, stop_id: str, endpoint: str) -> dict:
    result = stop_info(stop_id)
    if meta.get("location_type") == "geocoded":
        result.update({
            "access_mode": meta.get("access_mode"),
            "access_distance_km": meta.get("access_distance_km"),
            "access_query": meta.get("source_query"),
            "access_display_name": meta.get("display_name"),
            "access_place_id": f"__place_{endpoint}__",
            "access_latitude": meta.get("access_latitude"),
            "access_longitude": meta.get("access_longitude"),
        })
    return result


# --------------------------------------------------------------------------
# ML integration
# --------------------------------------------------------------------------
def _apply_ml(journeys: list[dict], departure_hour: int, day_type: str, model_type: str, weather: str = "clear"):
    """Batch-predict every ride segment of every journey (one model call)."""
    from models.predictor import predict_segments, select_model_for_context, update_dynamic_model_reward

    if model_type in ("dynamic_model_selection", "dynamic"):
        sample = journeys[0] if journeys else None
        if sample and sample.get("segments"):
            segment = sample["segments"][0]
            model_type = select_model_for_context(
                segment["distance_km"],
                segment["transport_type"],
                departure_hour,
                day_type,
                weather,
            )
        else:
            model_type = "random_forest"

    rows, refs = [], []
    for j in journeys:
        elapsed = 0.0
        for seg in j["segments"]:
            elapsed += seg["wait_min"]
            rows.append({"distance_km": seg["distance_km"], "transport_type": seg["transport_type"],
                         "stops_count": seg["num_stops"],
                         "departure_hour": int(departure_hour + elapsed / 60.0) % 24,
                         "day_type": day_type})
            refs.append(seg)
            elapsed += seg["time_min"]
    preds = predict_segments(rows, model_type=model_type)
    for seg, p in zip(refs, preds):
        seg["ml_predicted_time_min"] = round(float(p), 2)
    for j in journeys:
        ml_ride = sum(s["ml_predicted_time_min"] for s in j["segments"])
        j["ml_total_time_min"] = round(ml_ride + j["wait_time_min"], 2)

    if model_type in {"random_forest", "linear_regression", "xgboost"} and rows:
        ctx = rows[0]
        update_dynamic_model_reward(
            float(ctx["distance_km"]),
            ctx["transport_type"],
            int(ctx["departure_hour"]),
            ctx["day_type"],
            weather,
            model_type,
        )


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------
def _time(j):
    return j.get("ml_total_time_min", j["total_time_min"])


def _weather_multiplier(weather: str) -> float:
    return WEATHER_FACTORS.get(weather, 1.0)


def _apply_weather_delay(journey: dict, weather: str) -> float:
    multiplier = _weather_multiplier(weather)
    journey["_base_total_time_min"] = journey.get("_base_total_time_min", journey["total_time_min"])
    if "ml_total_time_min" in journey:
        journey["_base_ml_total_time_min"] = journey.get("_base_ml_total_time_min", journey["ml_total_time_min"])
    if multiplier == 1.0:
        journey["weather_delay_min"] = 0.0
        return 0.0

    extra = journey["total_time_min"] * (multiplier - 1.0)
    journey["total_time_min"] = round(journey["total_time_min"] * multiplier, 2)
    journey["in_vehicle_time_min"] = round(journey["in_vehicle_time_min"] * multiplier, 2)
    journey["wait_time_min"] = round(journey["wait_time_min"] * multiplier, 2)
    journey["weather_delay_min"] = round(extra, 2)
    if "ml_total_time_min" in journey:
        journey["ml_total_time_min"] = round(journey["ml_total_time_min"] * multiplier, 2)
    return journey["weather_delay_min"]


def _base_time(j):
    if "ml_total_time_min" in j and "_base_ml_total_time_min" in j:
        return j["_base_ml_total_time_min"]
    return j.get("_base_total_time_min", j["total_time_min"])


def _rank(journeys: list[dict], preference: str) -> list[dict]:
    if preference == "fastest":
        key = lambda j: (_base_time(j), j["total_fare_inr"])
    elif preference == "cheapest":
        key = lambda j: (j["total_fare_inr"], _base_time(j))
    elif preference == "least_walking":
        key = lambda j: (j["walking_distance_km"], _base_time(j))
    elif preference == "shortest":
        key = lambda j: (j["total_distance_km"], _base_time(j))
    else:  # balanced: min-max normalised weighted score
        def norm(vals):
            lo, hi = min(vals), max(vals)
            return [(v - lo) / (hi - lo) if hi > lo else 0.0 for v in vals]
        t = norm([_base_time(j) for j in journeys])
        f = norm([j["total_fare_inr"] for j in journeys])
        w = norm([j["walking_distance_km"] for j in journeys])
        x = norm([j["transfers"] for j in journeys])
        for j, a, b, c, dval in zip(journeys, t, f, w, x):
            j["balanced_score"] = round(0.40 * a + 0.25 * b + 0.10 * c + 0.25 * dval, 4)
        key = lambda j: (j["balanced_score"], _base_time(j))
    return sorted(journeys, key=key)


def _distinct(ranked: list[dict], k: int) -> list[dict]:
    """Best-ranked journey first, then the best ones that are genuinely different."""
    chosen = []
    for limit in DIVERSITY_PASSES:
        for j in ranked:
            if len(chosen) == k:
                return chosen
            if j not in chosen and all(_overlap(j, c) <= limit for c in chosen):
                chosen.append(j)
    return chosen


def _overlap(a, b):
    inter = len(a["_hops"] & b["_hops"])
    union = len(a["_hops"] | b["_hops"]) or 1
    return inter / union


def _tag(options: list[dict]):
    """Badge the best option for each criterion among the returned set."""
    if not options:
        return
    for name, fn in (("fastest", _time), ("cheapest", lambda j: j["total_fare_inr"]),
                     ("least_walking", lambda j: j["walking_distance_km"]),
                     ("fewest_transfers", lambda j: j["transfers"])):
        best = min(fn(o) for o in options)
        for o in options:
            if fn(o) == best:
                o["tags"].append(name)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_CACHE_SIZE = 500


def plan_journeys(start: str, destination: str, preference: str = "balanced",
                  max_options: int = 3, departure_hour: int = 9, day_type: str = "weekday",
                  ml_enhanced: bool = True, model_type: str = "random_forest",
                  weather: str = "clear") -> dict:
    if preference not in PREFERENCES:
        raise ValueError(f"preference must be one of {', '.join(PREFERENCES)}")
    if day_type not in ("weekday", "weekend"):
        raise ValueError("day_type must be 'weekday' or 'weekend'")
    if not 0 <= int(departure_hour) <= 23:
        raise ValueError("departure_hour must be between 0 and 23")
    if not 1 <= int(max_options) <= 5:
        raise ValueError("max_options must be between 1 and 5")
    if weather not in WEATHER_FACTORS:
        raise ValueError(f"weather must be one of {', '.join(WEATHER_FACTORS)}")
    start_meta = resolve_location(start)
    dest_meta = resolve_location(destination)
    start, destination = start_meta["stop_id"], dest_meta["stop_id"]
    if start == destination:
        if not start_meta.get("access_mode") and not dest_meta.get("access_mode"):
            raise ValueError("Start and destination must be different stops")

    src, dst = ("S", start), ("S", destination)
    seen, pool = set(), []
    if start == destination:
        pool = [_empty_journey(start)]
    else:
        for weight in ("w_time", "w_fare", "w_walk", "w_xfer"):
            try:
                gen = nx.shortest_simple_paths(GRAPH, src, dst, weight=weight)
                for path in itertools.islice(gen, CANDIDATES_PER_CRITERION):
                    j = _evaluate(path)
                    if j["_signature"] not in seen:
                        seen.add(j["_signature"])
                        pool.append(j)
            except nx.NetworkXNoPath:
                break

    for journey in pool:
        _add_access_legs(journey, start_meta, dest_meta, start, destination)
        _apply_weather_delay(journey, weather)

    if not pool:
        return {
            "start": _resolved_endpoint(start_meta, start, "start"),
            "destination": _resolved_endpoint(dest_meta, destination, "destination"),
            "preference": preference,
            "options": [],
            "candidates_considered": 0,
            "journey_context": {"weather": weather}
        }

    ml_warning = None
    selected_model = model_type
    bandit_details = None
    if ml_enhanced:
        try:
            if model_type in ("dynamic_model_selection", "dynamic"):
                first_segment = pool[0]["segments"][0] if pool and pool[0].get("segments") else None
                if first_segment is not None:
                    selected_model, bandit_details = select_model_for_context(
                        float(first_segment["distance_km"]),
                        first_segment["transport_type"],
                        int(departure_hour),
                        day_type,
                        weather,
                        return_details=True,
                    )
                else:
                    selected_model = "random_forest"
            _apply_ml(pool, int(departure_hour), day_type, selected_model, weather)
        except Exception as ex:  # fall back to timetable times
            ml_warning = f"ML prediction unavailable, using timetable times: {ex}"

    options = _distinct(_rank(pool, preference), int(max_options))
    for i, o in enumerate(options, 1):
        o["option_id"] = uuid.uuid4().hex[:12]
        o["rank"] = i
        o["tags"] = []
        o["summary"] = " → ".join(s["mode_label"] for s in o["segments"])
        if ml_enhanced and ml_warning is None:
            o["ml_model_used"] = selected_model
            if bandit_details:
                o["bandit_details"] = bandit_details
    _tag(options)

    clean = [{k: v for k, v in o.items() if not k.startswith("_")} for o in options]
    for o in clean:
        _CACHE[o["option_id"]] = o
        _CACHE.move_to_end(o["option_id"])
    while len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)

    journey_ctx = {
        "weather": weather,
        "selection_mode": model_type,
        "selected_model": selected_model if ml_enhanced and ml_warning is None else None,
    }
    if bandit_details:
        journey_ctx["bandit_details"] = bandit_details

    result = {
        "start": _resolved_endpoint(start_meta, start, "start"),
        "destination": _resolved_endpoint(dest_meta, destination, "destination"),
        "preference": preference, "departure_hour": int(departure_hour), "day_type": day_type,
        "ml_enhanced": ml_enhanced and ml_warning is None,
        "model_used": selected_model if ml_enhanced and ml_warning is None else None,
        "candidates_considered": len(pool),
        "journey_context": journey_ctx,
        "options": clean,
    }
    if ml_warning:
        result["ml_warning"] = ml_warning
    return result


def get_option(option_id: str) -> dict | None:
    return _CACHE.get(option_id)
