import pandas as pd
import networkx as nx
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

stops = pd.read_csv(DATA_DIR / "stops.csv")
routes = pd.read_csv(DATA_DIR / "routes.csv")
sequences = pd.read_csv(DATA_DIR / "stop_sequences.csv")
connections = pd.read_csv(DATA_DIR / "connections.csv")

print("=== EXISTING DATASET SUMMARY ===")
print(f"Stops count: {len(stops)}")
print(f"Routes count: {len(routes)}")
print(f"Stop sequences count: {len(sequences)}")
print(f"Connections count: {len(connections)}")
print(f"Total records: {len(stops) + len(routes) + len(sequences) + len(connections)}")

print("\n--- Stops by Mode ---")
print(stops['stop_type'].value_counts())

print("\n--- Stops by Region ---")
print(stops['region'].value_counts())

print("\n--- Routes by Mode ---")
print(routes['transport_type'].value_counts())

print("\n--- Graph Connectivity Check ---")
G = nx.MultiDiGraph()
for _, stop in stops.iterrows():
    G.add_node(stop["stop_id"], name=stop["stop_name"])
for _, conn in connections.iterrows():
    G.add_edge(conn["from_stop"], conn["to_stop"], transport_type=conn["transport_type"])

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Is strongly connected:", nx.is_strongly_connected(G))
print("Strongly connected components count:", nx.number_strongly_connected_components(G))
sccs = list(nx.strongly_connected_components(G))
print("Sizes of SCCs:", sorted([len(c) for c in sccs], reverse=True))

zero_in = [n for n in G.nodes if G.in_degree(n) == 0]
zero_out = [n for n in G.nodes if G.out_degree(n) == 0]
print(f"Zero in-degree nodes ({len(zero_in)}):", [(n, G.nodes[n]['name']) for n in zero_in])
print(f"Zero out-degree nodes ({len(zero_out)}):", [(n, G.nodes[n]['name']) for n in zero_out])

central_id = "S014" # Chennai Central
unreachable_from_central = [n for n in G.nodes if not nx.has_path(G, central_id, n)]
cannot_reach_central = [n for n in G.nodes if not nx.has_path(G, n, central_id)]
print(f"Nodes unreachable FROM Chennai Central ({len(unreachable_from_central)}):", [(n, G.nodes[n]['name']) for n in unreachable_from_central])
print(f"Nodes that CANNOT REACH Chennai Central ({len(cannot_reach_central)}):", [(n, G.nodes[n]['name']) for n in cannot_reach_central])
