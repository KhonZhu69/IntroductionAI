import streamlit as st
from routefinder import PathFinder
from predicted_volumes_gru import predicted_volumes as gru_volumes
from predicted_volumes_lstm import predicted_volumes as lstm_volumes
from predicted_volumes_svr import predicted_volumes as svr_volumes
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import copy

st.set_page_config(page_title="Traffic-Based Route Guidance System (TBRGS)")
st.title("Traffic-Based Route Guidance System (TBRGS)")

st.markdown("""
Upload a map, select a Machine Learning model, choose a search method,
and get up to 5 optimal routes with estimated travel times.
""")

map_file = st.file_uploader("Upload Map File (`.txt`)", type=["txt"])

if map_file is not None:
    map_path = Path("uploaded_map.txt")
    map_path.write_bytes(map_file.read())

    model_choice = st.selectbox("Choose a Machine Learning Model", ["GRU", "LSTM", "SVR"])
    predicted_volumes = {
        "GRU": gru_volumes,
        "LSTM": lstm_volumes,
        "SVR": svr_volumes
    }[model_choice]

    search_method = st.selectbox(
        "Choose Search Algorithm",
        ["dfs", "bfs", "gbfs", "as", "cus1", "cus2"],
        help="dfs=Depth-First, bfs=Breadth-First, gbfs=Greedy Best-First, as=A*, cus1=UCS, cus2=Bidirectional"
    )

    finder = PathFinder(str(map_path))
    finder.apply_predicted_volumes(predicted_volumes)

    all_nodes = sorted(finder.nodes.keys())
    origin = st.selectbox("📍 Select Origin Node (SCATS Site ID)", all_nodes)
    destination = st.selectbox("🏁 Select Destination Node (SCATS Site ID)", all_nodes)

    finder.set_origin_and_destinations(origin, [destination])

    if st.button("Show Routes"):
        route_limit = 5 if search_method == "as" else 1  # Only A* gets 5 routes
        count = 0
        visited_paths = set()
        base_graph = copy.deepcopy(finder.graph)

        st.markdown(f"### 🚦 Calculating route(s) from {origin} to {destination} using `{search_method.upper()}` and `{model_choice}` prediction...")

        while count < route_limit:
            finder.graph = copy.deepcopy(base_graph)

            # Increase weights of previous path edges for A* to find alternatives
            if search_method == "as":
                for path in visited_paths:
                    for i in range(len(path) - 1):
                        for j, (nbr, cost) in enumerate(finder.graph[path[i]]):
                            if nbr == path[i + 1]:
                                finder.graph[path[i]][j] = (nbr, cost + 10)

            # Run selected method
            if search_method == "as":
                goal, created, path = finder.astar()
            elif search_method == "dfs":
                goal, created, path = finder.dfs()
            elif search_method == "bfs":
                goal, created, path = finder.bfs()
            elif search_method == "gbfs":
                goal, created, path = finder.gbfs()
            elif search_method == "cus1":
                goal, created, path = finder.ucs()
            elif search_method == "cus2":
                goal, created, path = finder.bidirectional_search()
            else:
                st.error("Unsupported method.")
                break

            if not path or tuple(path) in visited_paths:
                break

            visited_paths.add(tuple(path))

            # Calculate travel time
            travel_time = 0
            for i in range(len(path) - 1):
                to_node = path[i + 1]
                flow = predicted_volumes.get(to_node, 1000)
                distance_km = 1.0
                try:
                    inner = 1 - 0.00066666 * flow
                    speed = 32.0001 * (inner ** 0.5) + 32.0001 if inner > 0 else 5
                    travel_time += (distance_km / speed) * 60 + 0.5
                except:
                    travel_time += 10.0

            st.subheader(f"Route {count + 1}")
            st.write("Path:", " → ".join(map(str, path)))
            st.write(f"Estimated Travel Time: {travel_time:.2f} minutes")
            st.write(f"Nodes Explored: {created}")

            # Visualization
            G = nx.DiGraph()
            for node in finder.graph:
                for neighbor, cost in finder.graph[node]:
                    G.add_edge(node, neighbor, weight=cost)

            pos = finder.nodes
            plt.figure(figsize=(8, 6))
            nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=800, font_size=10)
            nx.draw_networkx_edges(G, pos, edgelist=list(zip(path, path[1:])), edge_color="red", width=3)
            st.pyplot(plt.gcf())

            count += 1
