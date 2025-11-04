#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 07:15:28 2025

@author: samkirsh
"""

import streamlit as st
import pandas as pd
import networkx as nx
import folium
from streamlit_folium import st_folium
import pathlib
from zipfile import ZipFile

st.set_page_config(page_title="Rail Network Path Mapper", layout="wide")

# Define data paths
@st.cache_resource
def create_or_load_graph(nodes, edges):
    """Create or load a cached NetworkX graph"""
    DATA_DIR = pathlib.Path(__file__).parent / "data"
    GRAPH_PATH = DATA_DIR / "graph.gpickle"

    # ✅ If the graph already exists, load it directly
    if GRAPH_PATH.exists():
        st.info("Loading cached graph from disk...")
        G = nx.read_gpickle(GRAPH_PATH)
        return G

    # ✅ Otherwise, build it from CSVs and save
    st.warning("Graph not found on disk — building new graph (this may take a few minutes)...")
    G = nx.Graph()
    for _, row in nodes.iterrows():
        G.add_node(str(row["FRANODEID"]), state=row.get("STATE", ""), pos=(row["x"], row["y"]))

    for _, row in edges.iterrows():
        ownership = row.get("TRKRGHTS1")
        if pd.isna(ownership):
            ownership = ""
        else:
            ownership = str(ownership).strip()
        G.add_edge(str(row["FRFRANODE"]), str(row["TOFRANODE"]), weight=row.get("MILES", 1), ownership=ownership)

    # ✅ Save prebuilt graph for faster future loads
    nx.write_gpickle(G, GRAPH_PATH)
    st.success(f"Graph cached at {GRAPH_PATH}")
    return G


# --- Load data and graph ---
nodes, edges = load_data()
G = create_or_load_graph(nodes, edges)


#Collect the track rights owners together
owner_col = [c for c in edges.columns if "TRK" in c.upper() or "RGHTS" in c.upper()]

@st.cache_resource
def plot_paths(G, base_path, diversion_path):
    """Plot base and diversion paths on a Folium map"""
    if not base_path:
        st.error("No base path found.")
        return None

    m = folium.Map(location=[45, -95], zoom_start=5, tiles="CartoDB positron")

    def node_coords(node):
        data = G.nodes.get(node, {})
        if "pos" in data:
            x, y = data["pos"]
            return (y, x)  # folium wants (lat, lon)
        return None

    # Base path (blue)
    base_coords = [node_coords(n) for n in base_path if node_coords(n)]
    folium.PolyLine(base_coords, color="blue", weight=5, tooltip="Base Path").add_to(m)

    # Diversion path (red)
    if diversion_path:
        div_coords = [node_coords(n) for n in diversion_path if node_coords(n)]
        folium.PolyLine(div_coords, color="red", weight=4, tooltip="Diversion Path").add_to(m)

    return m

# --- Streamlit UI ---
st.title("🚆 North American Rail Network Path Mapper")


st.success(f"Loaded {len(edges)} edges and {len(nodes)} nodes.")

# Sidebar controls
st.sidebar.header("Path Configuration")
start_node = st.sidebar.text_input("Start Node (6-digit ID)", "")
end_node = st.sidebar.text_input("End Node (6-digit ID)", "")
avoid_nodes_input = st.sidebar.text_input("Nodes to avoid (comma-separated 6-digit IDs)", "")
avoid_nodes = [n.strip() for n in avoid_nodes_input.split(",") if n.strip()]

# --- Allowed Owner Selection ---
trk_cols = [f"TRKRGHTS{i}" for i in range(1, 10)]
unique_owners = pd.unique(edges[trk_cols].values.ravel())
unique_owners = [o for o in unique_owners if isinstance(o, str) and o.strip()]
unique_owners.sort()

allowed_owner = st.sidebar.selectbox("Allowed Owner (Railroad)", ["All"] + unique_owners)
    
# Remove avoided nodes
for node in avoid_nodes:
    if node in G:
        G.remove_node(node)

# Compute and plot
if st.sidebar.button("Compute Paths"):
    if start_node and end_node:
        # --- Create working copy of graph ---
        G_temp = G.copy()

        # --- Apply allowed owner filter ---
        if allowed_owner != "All":
            edges_to_remove = []
            for u, v, data in G_temp.edges(data=True):
                # Collect all trackage rights for this edge
                row = edges[
                    (edges["FRFRANODE"] == u) &
                    (edges["TOFRANODE"] == v)
                ]
                if not row.empty:
                    trk_values = [str(row.iloc[0].get(col, "")).strip() for col in trk_cols]
                    if allowed_owner not in trk_values:
                        edges_to_remove.append((u, v))
            G_temp.remove_edges_from(edges_to_remove)

        # --- Remove avoided nodes ---
        if avoid_nodes:
            avoid_list = [int(n.strip()) for n in avoid_nodes.split(",") if n.strip().isdigit()]
            G_temp.remove_nodes_from(avoid_list)

        # --- Compute base path ---
        try:
            base_path = nx.shortest_path(G_temp, start_node, end_node, weight="weight")
            base_distance = nx.shortest_path_length(G_temp, start_node, end_node, weight="weight")
        except nx.NetworkXNoPath:
            st.error("No base path found for the selected owner or nodes.")
            base_path, base_distance = None, None

        # --- Compute diversion path ---
        diversion_path, diversion_distance = None, None
        if base_path and len(base_path) > 2:
            G_div = G_temp.copy()
            i = len(base_path) // 2
            u, v = base_path[i - 1], base_path[i]
            if G_div.has_edge(u, v):
                G_div.remove_edge(u, v)
                try:
                    diversion_path = nx.shortest_path(G_div, start_node, end_node, weight="weight")
                    diversion_distance = nx.shortest_path_length(G_div, start_node, end_node, weight="weight")
                except nx.NetworkXNoPath:
                    diversion_path, diversion_distance = None, None
                    
        # --- Display distances ---
        st.subheader("📏 Path Distances")
        st.write(f"**Base path distance:** {base_distance:.2f} miles")
        if diversion_distance:
            st.write(f"**Diversion path distance:** {diversion_distance:.2f} miles")
            st.write(f"**Additional distance:** {diversion_distance - base_distance:.2f} miles")
        else:
            st.write("No valid diversion path found.")
            
        #Plotting
        if base_path:
            m = plot_paths(G, base_path, diversion_path)
            if m:
                st_folium(m, width=1200, height=700)
        else:
            st.error("No base path found between selected nodes.")
    else:
        st.warning("Please enter both start and end nodes.")
