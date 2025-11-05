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
import pickle

st.set_page_config(page_title="Rail Network Path Mapper", layout="wide")
@st.cache_resource
def get_allowed_edges(edges, allowed_owner, trk_cols):
    """Return a set of allowed edges for a given railroad owner"""
    if allowed_owner == "All":
        return None  # no filtering needed

    # normalize once for reliability
    edges = edges.copy()
    edges["FR_str"] = edges["FRFRANODE"].astype(str).str.strip()
    edges["TO_str"] = edges["TOFRANODE"].astype(str).str.strip()
    for c in trk_cols:
        edges[c] = edges[c].fillna("").astype(str).str.strip()

    # rows where any TRKRGHTS col matches this owner
    mask = edges[trk_cols].apply(
        lambda row: any((str(x).strip() == str(allowed_owner)) for x in row), axis=1
    )
    subset = edges[mask]

    # build set of (u,v) and (v,u) for undirected matching
    allowed_edges = set()
    for _, r in subset.iterrows():
        u, v = r["FR_str"], r["TO_str"]
        allowed_edges.add((u, v))
        allowed_edges.add((v, u))
    return allowed_edges
    
# Define data paths
@st.cache_data 
def load_data(): 
    """Load edges and nodes from local data folder""" 
    DATA_DIR = pathlib.Path(__file__).parent / "data" 
    EDGES_PATH = DATA_DIR / "Edges.csv.gz" 
    NODES_PATH = DATA_DIR / "Nodes.csv.gz" 
    edges = pd.read_csv(EDGES_PATH, compression='gzip') 
    nodes = pd.read_csv(NODES_PATH, compression='gzip') 
    return nodes, edges


@st.cache_resource
def create_or_load_graph(nodes, edges):
    """Create or load a cached NetworkX graph"""
    DATA_DIR = pathlib.Path(__file__).parent / "data"
    GRAPH_PATH = DATA_DIR / "graph.gpickle"

    # ✅ If the graph already exists, load it directly
    if GRAPH_PATH.exists():
        #st.info("Loading cached graph from disk...")
        with open(GRAPH_PATH, "rb") as f:
            G = pickle.load(f)
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
   
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    st.success(f"Graph cached at {GRAPH_PATH}")
    return G


# --- Load data and graph ---
nodes, edges = load_data()
G = create_or_load_graph(nodes, edges)


#Collect the track rights owners together
owner_col = [c for c in edges.columns if "TRK" in c.upper() or "RGHTS" in c.upper()]


def plot_paths(_G, base_path, diversion_path):
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
base_speed = st.sidebar.number_input("Base Speed (mph)")
div_speed = st.sidebar.number_input("Diversion Speed (mph)")
fuel_cost_per_mile =st.sidebar.number_input("Fuel Cost per Mile")
labor_cost_per_mile=st.sidebar.number_input("Labor Cost per Mile")
edges_to_remove =[]
# --- Allowed Owner Selection ---
trk_cols = [f"TRKRGHTS{i}" for i in range(1, 10)]
unique_owners = pd.unique(edges[trk_cols].values.ravel())
unique_owners = [o for o in unique_owners if isinstance(o, str) and o.strip()]
unique_owners.sort()

allowed_owner = st.sidebar.multiselect("Allowed Owner (Railroad)", ["All"] + unique_owners)
    
# Remove avoided nodes
for node in avoid_nodes:
    if node in G:
        G.remove_node(node)

# Compute and plot
# --- Compute and plot ---
if st.sidebar.button("Compute Paths"):
    if start_node and end_node:
        # --- Create working copy of graph ---
        G_temp = G.copy()

        # --- Apply allowed owner filter with caching ---
        if allowed_owner != "All":
            allowed_edges = get_allowed_edges(edges, allowed_owner, trk_cols)
            if not allowed_edges:
                st.warning(f"No edges found for owner {allowed_owner}.")
            else:
                edges_to_remove = [
                    (u, v) for u, v in list(G_temp.edges())
                    if (str(u).strip(), str(v).strip()) not in allowed_edges
                ]
                G_temp.remove_edges_from(edges_to_remove)
                st.session_state["allowed_edges"] = allowed_edges
        else:
            st.session_state["allowed_edges"] = None

        # Store filtered graph in session state
        st.session_state["filtered_graph"] = G_temp
        st.write(f"Ownership filter: removed {len(edges_to_remove)} edges for {allowed_owner}.")

        # --- Compute base path on filtered graph ---
        try:
            base_path = nx.shortest_path(G_temp, start_node, end_node, weight="weight")
            base_distance = nx.shortest_path_length(G_temp, start_node, end_node, weight="weight")
        
        except nx.NetworkXNoPath:
            st.warning("⚠️ Base path cannot be computed based on ownership restriction.")
            base_path, base_distance = None, None
        
        except nx.NodeNotFound as e:
            st.error(f"❌ Invalid node: {e}. Please check your start and end node IDs.")
            base_path, base_distance = None, None
        
        except Exception as e:
            st.error(f"Unexpected error computing base path: {e}")
            base_path, base_distance = None, None

        # --- Compute diversion path only if avoid nodes provided ---
        diversion_path, diversion_distance = None, None
        if avoid_nodes:
            G_div = G_temp.copy()
            avoid_list = [n.strip() for n in avoid_nodes if n.strip().isdigit()]
            G_div.remove_nodes_from(avoid_list)
            try:
                diversion_path = nx.shortest_path(G_div, start_node, end_node, weight="weight")
                diversion_distance = nx.shortest_path_length(G_div, start_node, end_node, weight="weight")
            except nx.NetworkXNoPath:
                diversion_path, diversion_distance = None, None

        # --- Compute and display results ---
        if base_path:
            # Validate speeds
            if base_speed <= 0 or (avoid_nodes and div_speed <= 0):
                st.error("Please enter valid (positive) speeds for all paths before computing.")
            else:
                base_time = base_distance / base_speed
                base_fuel = base_distance * fuel_cost_per_mile
                base_labor = base_distance * labor_cost_per_mile

                div_time = div_fuel = div_labor = 0
                if diversion_distance:
                    div_time = diversion_distance / div_speed
                    div_fuel = diversion_distance * fuel_cost_per_mile
                    div_labor = diversion_distance * labor_cost_per_mile

                # --- Plot and store results ---
                m = plot_paths(G, base_path, diversion_path)
                if m:
                    st.session_state["results"] = {
                        "base": {
                            "distance": base_distance,
                            "speed": base_speed,
                            "time": base_time,
                            "fuel": base_fuel,
                            "labor": base_labor,
                        },
                        "diversion": {
                            "distance": diversion_distance,
                            "speed": div_speed,
                            "time": div_time,
                            "fuel": div_fuel,
                            "labor": div_labor,
                        },
                    }
                    st.session_state["map"] = m

    else:
        st.warning("Please enter both start and end nodes.")


# --- Always display last computed results if they exist ---

if "filtered_graph" in st.session_state:
    G_temp = st.session_state["filtered_graph"]
    allowed_edges = st.session_state.get("allowed_edges")

    if allowed_edges is not None:
        st.info(f"Ownership filter active: {len(allowed_edges)} allowed edges for {allowed_owner}.")
    else:
        st.info("No ownership filter applied (All).")
        
if "results" in st.session_state:
    res = st.session_state["results"]
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="border:1px solid black; border-radius:8px; padding:10px;">
        <h4 style="text-align:center;">Base Path</h4>
        """, unsafe_allow_html=True)
        st.markdown(f"**Distance:** {res['base']['distance']:.2f} miles")
        st.markdown(f"**Avg Speed:** {res['base']['speed']:.1f} mph")
        st.markdown(f"**Travel Time:** {res['base']['time']:.2f} hours")
        st.markdown(f"**Fuel Cost:** ${res['base']['fuel']:,.2f}")
        st.markdown(f"**Labor Cost:** ${res['base']['labor']:,.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="border:1px solid black; border-radius:8px; padding:10px;">
        <h4 style="text-align:center;">Diversion Path</h4>
        """, unsafe_allow_html=True)
        if res["diversion"]["distance"]:
            st.markdown(f"**Distance:** {res['diversion']['distance']:.2f} miles")
            st.markdown(f"**Avg Speed:** {res['diversion']['speed']:.1f} mph")
            st.markdown(f"**Travel Time:** {res['diversion']['time']:.2f} hours")
            st.markdown(f"**Fuel Cost:** ${res['diversion']['fuel']:,.2f}")
            st.markdown(f"**Labor Cost:** ${res['diversion']['labor']:,.2f}")
        else:
            st.markdown("No valid diversion path found.")
        st.markdown("</div>", unsafe_allow_html=True)
    # --- Add a Clear Results button ---
    st.markdown("---")
    if st.button("🧹 Clear Results"):
        for key in ["results", "map"]:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()
# --- Display the map persistently ---
if "map" in st.session_state:
    st_folium(st.session_state["map"], width=1200, height=700)
