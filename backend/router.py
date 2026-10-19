
import pandas as pd
import networkx as nx
from pathlib import Path

DATA = Path(__file__).parent / "data"

stops_df = pd.read_csv(DATA / "stops.csv")
connections_df = pd.read_csv(DATA / "connections.csv")

# Build a directed graph
graph = nx.MultiDiGraph()

# Add all stops
for _, stop in stops_df.iterrows():
    graph.add_node(
        stop["stop_id"],
        name=stop["stop_name"],
        area=stop["area"]
    )

# Add transport connections
for _, conn in connections_df.iterrows():
    graph.add_edge(
        conn["from_stop"],
        conn["to_stop"],
        transport_type=conn["transport_type"],
        time=float(conn["estimated_time_min"]),
        distance=float(conn["distance_km"]),
        fare=float(conn["fare_inr"]),
        route_id=conn["route_id"]
    )



def calc_metro_fare(distance_km: float) -> float:
    """Official CMRL distance-slab token fare chart."""
    d = float(distance_km)
    if d <= 2.0:
        return 10.0
    elif d <= 5.0:
        return 20.0
    elif d <= 12.0:
        return 30.0
    elif d <= 21.0:
        return 40.0
    else:
        return 50.0


def calc_rail_fare(distance_km: float) -> float:
    """Official Indian Railways Chennai Suburban & MRTS fare structure."""
    d = float(distance_km)
    if d <= 5.0:
        return 5.0
    elif d <= 15.0:
        return 10.0
    elif d <= 25.0:
        return 15.0
    elif d <= 30.0:
        return 20.0
    elif d <= 45.0:
        return 25.0
    else:
        return 30.0


def calc_bus_fare(distance_km: float) -> float:
    """Official MTC Chennai stage-wise fare structure (~2km per stage)."""
    import math
    d = float(distance_km)
    stage = math.ceil(d / 2.0)
    if stage <= 1:
        return 5.0
    elif stage == 2:
        return 6.0
    elif stage == 3:
        return 7.0
    elif stage == 4:
        return 8.0
    elif stage == 5:
        return 9.0
    elif stage == 6:
        return 10.0
    elif stage == 7:
        return 11.0
    elif stage == 8:
        return 12.0
    elif stage == 9:
        return 13.0
    elif stage == 10:
        return 14.0
    elif stage <= 12:
        return 15.0
    elif stage <= 14:
        return 16.0
    elif stage <= 16:
        return 17.0
    elif stage <= 18:
        return 18.0
    elif stage <= 21:
        return 19.0
    elif stage <= 24:
        return 21.0
    elif stage <= 27:
        return 23.0
    else:
        return 25.0


def calc_segment_fare(transport_type: str, distance_km: float) -> float:
    """Calculate the overall fare for a continuous journey on a given mode."""
    d = float(distance_km)
    mode = str(transport_type).lower()
    if mode == "walk":
        return 0.0
    elif mode == "metro":
        return calc_metro_fare(d)
    elif mode in ("rail", "suburban_rail", "mrts"):
        return calc_rail_fare(d)
    elif mode == "bus":
        return calc_bus_fare(d)
    elif mode == "auto":
        # Tamil Nadu regulated auto fare: Rs 40 for first 1.8km, Rs 18/km after
        return round(40.0 + max(0.0, d - 1.8) * 18.0, 2)
    elif mode == "cab":
        # Standard taxi rate: Rs 100 base + Rs 20/km
        return round(100.0 + d * 20.0, 2)
    else:
        return 10.0


def find_route(start, destination, preference="fastest"):
    if start not in graph:
        raise ValueError(f"Unknown start stop: {start}")

    if destination not in graph:
        raise ValueError(f"Unknown destination stop: {destination}")

    weights = {
        "fastest": "time",
        "cheapest": "fare",
        "shortest": "distance"
    }

    if preference not in weights:
        raise ValueError("Preference must be fastest, cheapest, or shortest")

    weight = weights[preference]

    try:
        path = nx.shortest_path(
            graph,
            source=start,
            target=destination,
            weight=weight
        )
    except nx.NetworkXNoPath:
        return None

    raw_legs = []
    total_time = 0.0
    total_distance = 0.0

    for u, v in zip(path, path[1:]):
        edges = graph.get_edge_data(u, v)
        edge = min(edges.values(), key=lambda e: e[weight])

        total_time += edge["time"]
        total_distance += edge["distance"]

        raw_legs.append({
            "from_stop": u,
            "from_name": graph.nodes[u]["name"],
            "to_stop": v,
            "to_name": graph.nodes[v]["name"],
            "transport_type": edge["transport_type"],
            "route_id": edge["route_id"],
            "time_min": edge["time"],
            "distance_km": edge["distance"]
        })

    # Group continuous transit rides into segments to compute overall ride fares
    segments = []
    current_seg = None

    for leg in raw_legs:
        mode = leg["transport_type"]
        rid = leg["route_id"]

        same_seg = False
        if current_seg is not None:
            if mode == current_seg["transport_type"]:
                # Continuous metro or continuous suburban/MRTS rail share network ticketing
                if mode in ("metro", "rail", "suburban_rail", "mrts", "walk"):
                    same_seg = True
                # Bus is continuous if on the same bus route (changing routes requires a new ticket)
                elif mode == "bus" and rid == current_seg["route_id"]:
                    same_seg = True
                elif mode in ("auto", "cab") and rid == current_seg["route_id"]:
                    same_seg = True

        if same_seg:
            current_seg["legs"].append(leg)
            current_seg["to_stop"] = leg["to_stop"]
            current_seg["to_name"] = leg["to_name"]
            current_seg["distance_km"] += leg["distance_km"]
            current_seg["time_min"] += leg["time_min"]
        else:
            if current_seg is not None:
                segments.append(current_seg)
            current_seg = {
                "transport_type": mode,
                "route_id": rid,
                "from_stop": leg["from_stop"],
                "from_name": leg["from_name"],
                "to_stop": leg["to_stop"],
                "to_name": leg["to_name"],
                "distance_km": leg["distance_km"],
                "time_min": leg["time_min"],
                "legs": [leg]
            }

    if current_seg is not None:
        segments.append(current_seg)

    # Calculate overall fare per segment and allocate proportionally to individual legs
    total_fare = 0.0
    final_legs = []

    for seg in segments:
        seg_dist = seg["distance_km"]
        seg_fare = calc_segment_fare(seg["transport_type"], seg_dist)
        seg["fare_inr"] = round(seg_fare, 2)
        seg["num_stops"] = len(seg["legs"])
        total_fare += seg_fare

        rem_fare = seg_fare
        num_legs = len(seg["legs"])
        for idx, leg in enumerate(seg["legs"]):
            if idx == num_legs - 1:
                leg_fare = round(rem_fare, 2)
            else:
                ratio = (leg["distance_km"] / seg_dist) if seg_dist > 0 else (1.0 / num_legs)
                leg_fare = round(seg_fare * ratio, 2)
                rem_fare -= leg_fare
            leg["fare_inr"] = max(0.0, leg_fare)
            final_legs.append(leg)

    return {
        "start": graph.nodes[start]["name"],
        "destination": graph.nodes[destination]["name"],
        "preference": preference,
        "stops": [
            {"stop_id": stop, "stop_name": graph.nodes[stop]["name"]}
            for stop in path
        ],
        "legs": final_legs,
        "segments": [
            {
                "transport_type": seg["transport_type"],
                "route_id": seg["route_id"],
                "from_stop": seg["from_stop"],
                "from_name": seg["from_name"],
                "to_stop": seg["to_stop"],
                "to_name": seg["to_name"],
                "num_stops": seg["num_stops"],
                "distance_km": round(seg["distance_km"], 2),
                "time_min": round(seg["time_min"], 2),
                "fare_inr": round(seg["fare_inr"], 2)
            }
            for seg in segments
        ],
        "total_time_min": round(total_time, 2),
        "total_distance_km": round(total_distance, 2),
        "total_fare_inr": round(total_fare, 2)
    }


def get_stops():
    return stops_df[
        ["stop_id", "stop_name", "area", "region", "stop_type"]
    ].fillna("").to_dict(orient="records")

