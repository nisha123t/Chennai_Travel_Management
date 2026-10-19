"""
Comprehensive Validation Script for Candidate Multimodal Dataset in backend/staging_data/
Validates:
1. Duplicate IDs
2. Missing / null values
3. Coordinate boundary sanity
4. Referential integrity (FK references)
5. Graph connectivity, in/out degrees, strongly connected components, reachability
6. Route planner simulation across multiple inter-region journeys
"""
import pandas as pd
import networkx as nx
from pathlib import Path

STAGING = Path(__file__).parent / "staging_data"

stops = pd.read_csv(STAGING / "stops.csv")
routes = pd.read_csv(STAGING / "routes.csv")
sequences = pd.read_csv(STAGING / "stop_sequences.csv")
connections = pd.read_csv(STAGING / "connections.csv")

print("==================================================")
print("     CANDIDATE DATASET COMPREHENSIVE VALIDATION   ")
print("==================================================")

errors = []

# 1. DUPLICATE ID CHECKS
dup_stops = stops["stop_id"].duplicated().sum()
dup_routes = routes["route_id"].duplicated().sum()
dup_conns = connections["connection_id"].duplicated().sum()
dup_seqs = sequences.duplicated(subset=["route_id", "sequence_no"]).sum()

print(f"[1] Duplicate stop IDs: {dup_stops}")
print(f"    Duplicate route IDs: {dup_routes}")
print(f"    Duplicate connection IDs: {dup_conns}")
print(f"    Duplicate (route, sequence) pairs: {dup_seqs}")

if dup_stops > 0: errors.append(f"Found {dup_stops} duplicate stop IDs")
if dup_routes > 0: errors.append(f"Found {dup_routes} duplicate route IDs")
if dup_conns > 0: errors.append(f"Found {dup_conns} duplicate connection IDs")
if dup_seqs > 0: errors.append(f"Found {dup_seqs} duplicate sequence numbers")

# 2. MISSING VALUES CHECK
print("\n[2] Missing Values:")
for name, df in [("Stops", stops), ("Routes", routes), ("Sequences", sequences), ("Connections", connections)]:
    null_count = df.isnull().sum().sum()
    print(f"    {name} missing values: {null_count}")
    if null_count > 0:
        print(f"    Columns with nulls in {name}:", df.columns[df.isnull().any()].tolist())
        errors.append(f"{name} has {null_count} null values")

# 3. COORDINATE BOUNDARY CHECK (Chennai Metro Region: 12.6 to 13.5 Lat, 79.8 to 80.4 Lon)
print("\n[3] Coordinate Boundaries:")
lat_min, lat_max = stops["latitude"].min(), stops["latitude"].max()
lon_min, lon_max = stops["longitude"].min(), stops["longitude"].max()
print(f"    Latitude range:  {lat_min:.4f} to {lat_max:.4f}")
print(f"    Longitude range: {lon_min:.4f} to {lon_max:.4f}")

invalid_coords = stops[(stops["latitude"] < 12.6) | (stops["latitude"] > 13.5) | 
                       (stops["longitude"] < 79.8) | (stops["longitude"] > 80.4)]
print(f"    Invalid coordinate count: {len(invalid_coords)}")
if len(invalid_coords) > 0:
    errors.append(f"Found {len(invalid_coords)} stops with out-of-bounds coordinates")

# 4. REFERENTIAL INTEGRITY (FOREIGN KEYS)
print("\n[4] Referential Integrity:")
stop_ids = set(stops["stop_id"])
route_ids = set(routes["route_id"])

inv_conn_from = (~connections["from_stop"].isin(stop_ids)).sum()
inv_conn_to = (~connections["to_stop"].isin(stop_ids)).sum()
inv_conn_route = (~connections["route_id"].isin(route_ids)).sum()

inv_seq_route = (~sequences["route_id"].isin(route_ids)).sum()
inv_seq_stop = (~sequences["stop_id"].isin(stop_ids)).sum()

inv_route_src = (~routes["source_stop"].isin(stop_ids)).sum()
inv_route_dst = (~routes["destination_stop"].isin(stop_ids)).sum()

print(f"    Invalid connection origins:      {inv_conn_from}")
print(f"    Invalid connection destinations: {inv_conn_to}")
print(f"    Invalid connection routes:       {inv_conn_route}")
print(f"    Invalid sequence routes:         {inv_seq_route}")
print(f"    Invalid sequence stops:          {inv_seq_stop}")
print(f"    Invalid route source stops:      {inv_route_src}")
print(f"    Invalid route destination stops: {inv_route_dst}")

if inv_conn_from + inv_conn_to + inv_conn_route + inv_seq_route + inv_seq_stop + inv_route_src + inv_route_dst > 0:
    errors.append("Foreign key referential integrity failures detected")

# 5. GRAPH CONNECTIVITY & REACHABILITY
print("\n[5] Network Graph Connectivity:")
G = nx.MultiDiGraph()
for _, stop in stops.iterrows():
    G.add_node(stop["stop_id"], name=stop["stop_name"], area=stop["area"])
for _, conn in connections.iterrows():
    G.add_edge(conn["from_stop"], conn["to_stop"],
               transport_type=conn["transport_type"],
               time=float(conn["estimated_time_min"]),
               distance=float(conn["distance_km"]),
               fare=float(conn["fare_inr"]),
               route_id=conn["route_id"])

is_scc = nx.is_strongly_connected(G)
scc_count = nx.number_strongly_connected_components(G)
sccs = list(nx.strongly_connected_components(G))
scc_sizes = sorted([len(c) for c in sccs], reverse=True)

zero_in = [n for n in G.nodes if G.in_degree(n) == 0]
zero_out = [n for n in G.nodes if G.out_degree(n) == 0]

print(f"    Total stops in graph: {G.number_of_nodes()}")
print(f"    Total edges in graph: {G.number_of_edges()}")
print(f"    Is strongly connected: {is_scc}")
print(f"    Strongly connected components count: {scc_count}")
print(f"    SCC size distribution: {scc_sizes}")
print(f"    Zero in-degree stops (unreachable):  {len(zero_in)} {zero_in}")
print(f"    Zero out-degree stops (dead ends):   {len(zero_out)} {zero_out}")

if not is_scc:
    errors.append(f"Graph is not strongly connected. Number of SCCs: {scc_count}")
if len(zero_in) > 0:
    errors.append(f"{len(zero_in)} stops have 0 in-degree (cannot be reached)")
if len(zero_out) > 0:
    errors.append(f"{len(zero_out)} stops have 0 out-degree (dead ends)")

# 6. ROUTE SIMULATION TEST
print("\n[6] Route Planning Simulation across Key Chennai Corridors:")
test_pairs = [
    ("S001", "S065", "Broadway (North) -> Kilambakkam KCBT (South)"),
    ("S075", "S058", "Wimco Nagar Depot (North) -> Airport (South)"),
    ("S098", "S014", "Gummidipoondi (Far North) -> Central (Central)"),
    ("S110", "S124", "Tiruvallur (Far West) -> Besant Nagar (South Coast)"),
    ("S132", "S129", "Poonamallee (West) -> Siruseri IT Park (OMR South)"),
    ("S119", "S062", "Red Hills (North) -> Tambaram (South)"),
    ("S067", "S003", "Velachery MRTS (South) -> Chennai Beach (North)")
]

for src, dst, label in test_pairs:
    try:
        path = nx.shortest_path(G, source=src, target=dst, weight="time")
        dist = sum(min(G.get_edge_data(u, v).values(), key=lambda e: e['time'])['distance'] for u, v in zip(path, path[1:]))
        time = sum(min(G.get_edge_data(u, v).values(), key=lambda e: e['time'])['time'] for u, v in zip(path, path[1:]))
        fare = sum(min(G.get_edge_data(u, v).values(), key=lambda e: e['time'])['fare'] for u, v in zip(path, path[1:]))
        modes = [min(G.get_edge_data(u, v).values(), key=lambda e: e['time'])['transport_type'] for u, v in zip(path, path[1:])]
        distinct_modes = list(dict.fromkeys(modes))
        print(f"    PASS: {label}")
        print(f"          Stops: {len(path)} | Time: {time:.1f} min | Dist: {dist:.1f} km | Fare: Rs {fare:.0f} | Modes: {' -> '.join(distinct_modes)}")
    except Exception as ex:
        print(f"    FAIL: {label} - {ex}")
        errors.append(f"Routing failed for {label}: {ex}")

print("\n==================================================")
if len(errors) == 0:
    print("      ALL VALIDATION CHECKS PASSED (0 ERRORS)     ")
else:
    print(f"      VALIDATION FAILED WITH {len(errors)} ERRORS:")
    for err in errors:
        print(f"      - {err}")
print("==================================================")
