
import pandas as pd
from pathlib import Path

DATA = Path(__file__).parent / "data"

stops = pd.read_csv(DATA / "stops.csv")
routes = pd.read_csv(DATA / "routes.csv")
sequences = pd.read_csv(DATA / "stop_sequences.csv")
connections = pd.read_csv(DATA / "connections.csv")

# Check duplicate IDs
print("Duplicate stop IDs:", stops["stop_id"].duplicated().sum())
print("Duplicate route IDs:", routes["route_id"].duplicated().sum())

# Check missing values
for name, df in [
    ("Stops", stops),
    ("Routes", routes),
    ("Sequences", sequences),
    ("Connections", connections)
]:
    print(f"{name} missing values:", df.isnull().sum().sum())

# Check invalid stop references
stop_ids = set(stops["stop_id"])

invalid_from = ~connections["from_stop"].isin(stop_ids)
invalid_to = ~connections["to_stop"].isin(stop_ids)

print("Invalid connection origins:", invalid_from.sum())
print("Invalid connection destinations:", invalid_to.sum())

# Check invalid route references
route_ids = set(routes["route_id"])

print(
    "Invalid sequence route IDs:",
    (~sequences["route_id"].isin(route_ids)).sum()
)
print(
    "Invalid sequence stop IDs:",
    (~sequences["stop_id"].isin(stop_ids)).sum()
)

# Coordinate boundary check
invalid_coords = stops[
    (stops["latitude"] < 12.6) | (stops["latitude"] > 13.5) |
    (stops["longitude"] < 79.8) | (stops["longitude"] > 80.4)
]
print("Invalid coordinate count:", len(invalid_coords))

# Network connectivity check
import networkx as nx
graph = nx.MultiDiGraph()
for _, stop in stops.iterrows():
    graph.add_node(stop["stop_id"], name=stop["stop_name"])
for _, conn in connections.iterrows():
    graph.add_edge(conn["from_stop"], conn["to_stop"], transport_type=conn["transport_type"])

print("Graph nodes count:", graph.number_of_nodes())
print("Graph edges count:", graph.number_of_edges())
print("Is strongly connected:", nx.is_strongly_connected(graph))
print("Strongly connected components count:", nx.number_strongly_connected_components(graph))
print("Unreachable stops (zero in-degree):", sum(1 for n in graph.nodes if graph.in_degree(n) == 0))
print("Dead-end stops (zero out-degree):", sum(1 for n in graph.nodes if graph.out_degree(n) == 0))

total_records = len(stops) + len(routes) + len(sequences) + len(connections)
print(f"Total dataset records: {total_records} (Stops: {len(stops)}, Routes: {len(routes)}, Sequences: {len(sequences)}, Connections: {len(connections)})")

print("\nValidation check completed.")
