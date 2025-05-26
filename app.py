
import streamlit as st
from routefinder import PathFinder
from predicted_volumes_gru import predicted_volumes as gru_volumes
from predicted_volumes_lstm import predicted_volumes as lstm_volumes
from predicted_volumes_svr import predicted_volumes as svr_volumes
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import copy
import pandas as pd
import numpy as np

st.set_page_config(page_title="Traffic-Based Route Guidance System (TBRGS)")
st.title("Traffic-Based Route Guidance System (TBRGS)")

st.markdown("""
Upload a map, select a Machine Learning model, choose a search method,
and view the optimal route with estimated travel time and node exploration details.
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

    if st.button("Show Route"):
        st.markdown(f"### 🚦 Calculating route from {origin} to {destination} using `{search_method.upper()}` and `{model_choice}` prediction...")

        base_graph = copy.deepcopy(finder.graph)
        goal, created, path = None, 0, []

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

        if not path:
            st.error("No path found.")
        else:
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

            st.success(f"Path: {' → '.join(map(str, path))}")
            st.info(f"Estimated Travel Time: {travel_time:.2f} minutes")
            st.info(f"Nodes Explored: {created}")

            G = nx.DiGraph()
            for node in finder.graph:
                for neighbor, cost in finder.graph[node]:
                    G.add_edge(node, neighbor, weight=cost)

            pos = finder.nodes
            plt.figure(figsize=(8, 6))
            nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=800, font_size=10)
            nx.draw_networkx_edges(G, pos, edgelist=list(zip(path, path[1:])), edge_color="red", width=3)
            st.pyplot(plt.gcf())

    # Heatmap toggle
    if st.checkbox("Show Heatmap of Traffic Zones (from CSV data)"):
        try:
            df = pd.read_csv("cleaned_SData_Oct2006.csv")
            volume_columns = [col for col in df.columns if col.startswith('v') and col[1:].isdigit()]
            coord_columns = ['nb_latitude', 'nb_longitude']
            df = df.dropna(subset=coord_columns + volume_columns)
            df['avg_volume'] = df[volume_columns].mean(axis=1)
            df_grouped = df.groupby(coord_columns)['avg_volume'].mean().reset_index()
            heatmap_data = df_grouped[['nb_latitude', 'nb_longitude', 'avg_volume']].values.tolist()

            from folium.plugins import HeatMap
            import folium
            from streamlit_folium import st_folium

            avg_lat = np.mean(df_grouped['nb_latitude'])
            avg_lon = np.mean(df_grouped['nb_longitude'])
            m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)
            HeatMap(heatmap_data, radius=10, blur=6, max_zoom=13).add_to(m)
            st_folium(m, width=700, height=500)
        except Exception as e:
            st.error(f"Heatmap error: {e}")
