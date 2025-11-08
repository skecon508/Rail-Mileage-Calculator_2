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
import io
import pandas as pd
from openpyxl import Workbook
#from openpyxl.drawing.image import Image as XLImage
#import wkhtmltopdf

st.set_page_config(page_title="Rail Network Path Mapper", layout="wide")

@st.cache_resource
def get_allowed_edges(edges, allowed_owners, trk_cols):
    """Return set of allowed edges for one or more selected owners."""
    # If "All" selected, skip filtering
    if "All" in allowed_owners or not allowed_owners:
        return None

    allowed_owners = [str(o).strip() for o in allowed_owners]

    edges = edges.copy()
    edges["FR_str"] = edges["FRFRANODE"].astype(str).str.strip()
    edges["TO_str"] = edges["TOFRANODE"].astype(str).str.strip()

    for c in trk_cols:
        edges[c] = edges[c].fillna("").astype(str).str.strip()

    # Row is allowed if ANY trackage rights col matches ANY selected owner
    mask = edges[trk_cols].apply(lambda row: any(x in allowed_owners for x in row), axis=1)
    subset = edges[mask]

    allowed_edges = set()
    for _, r in subset.iterrows():
        u, v = r["FR_str"], r["TO_str"]
        allowed_edges.add((u, v))
        allowed_edges.add((v, u))  # undirected edges

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

#Define plotting function
def plot_paths(G_in, base_path, diversion_path, show_network=False):
    """Plot base + diversion paths on a Folium map, optionally showing full network."""
    if not base_path:
        st.error("No base path found.")
        return None

    m = folium.Map(location=[45, -95], zoom_start=5, tiles="CartoDB positron")

    # Helper to extract coordinates
    def node_coords(node):
        data = G_in.nodes.get(node, {})
        if "pos" in data:
            x, y = data["pos"]
            return (y, x)  # lat, lon
        return None

    # Optional: Light-grey background network
    if show_network:
        for u, v in G_in.edges():
            p1 = node_coords(u)
            p2 = node_coords(v)
            if p1 and p2:
                folium.PolyLine([p1, p2], color="#C4C4C4", weight=1, opacity=0.7).add_to(m)

    # Base path (blue)
    base_coords = [node_coords(n) for n in base_path if node_coords(n)]
    folium.PolyLine(base_coords, color="blue", weight=4, tooltip="Base Path").add_to(m)

    # Diversion path (red)
    if diversion_path:
        div_coords = [node_coords(n) for n in diversion_path if node_coords(n)]
        folium.PolyLine(div_coords, color="red", weight=4, tooltip="Diversion Path").add_to(m)

    return m


# --- Plot underlying network ----
def plot_full_network(G):
    """Return a folium map with the full rail network drawn in light grey."""
    m = folium.Map(location=[45, -95], zoom_start=5, tiles="CartoDB positron")

    for u, v, data in G.edges(data=True):
        u_pos = G.nodes[u].get("pos")
        v_pos = G.nodes[v].get("pos")
        if u_pos and v_pos:
            (x1, y1) = u_pos
            (x2, y2) = v_pos
            folium.PolyLine([(y1, x1), (y2, x2)],
                            color="#CCCCCC",
                            weight=1,
                            opacity=0.5).add_to(m)
    return m


# --- Streamlit UI ---
st.title("🚆 North American Rail Network Path Mapper")

st.success(f"Loaded {len(edges)} edges and {len(nodes)} nodes.")

# Sidebar controls
st.sidebar.header("Path Configuration")
start_node = st.sidebar.text_input("Start Node (6-digit ID)", value=st.session_state.get("start_node", ""))
end_node = st.sidebar.text_input("End Node (6-digit ID)", value=st.session_state.get("end_node", ""))

#FRA Links
st.sidebar.markdown("---")
st.sidebar.markdown("### 📎 Reference Maps")
show_network = st.sidebar.checkbox("Show Full Network", value=True)
st.sidebar.markdown(
    "[FRA Nodes Map](https://geodata.bts.gov/datasets/54f40ddee0844fb285dafb13a6b2f623_0/explore?location=32.691740%2C-108.315141%2C3.98)",
    unsafe_allow_html=True
)

st.sidebar.markdown(
    "[FRA Edges (Rail Lines) Map](https://geodata.bts.gov/datasets/e143f436d4774402aa8cca1e663b1d24_0/explore?location=41.945498%2C-72.474621%2C9.00)",
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

# --- Node Lookup Panel ---
st.sidebar.markdown("### 🔍 Node Lookup")
with st.expander("Lookup Nodes by Filters"):

    # Determine available filter columns
    lookup_states = sorted(nodes["STATE"].dropna().unique())
    state_sel = st.selectbox("State", ["All"] + lookup_states)

    # Optional filters (only shown if column exists)
    div_sel = None
    if "DIVISION" in edges.columns:
        lookup_divs = sorted(edges["DIVISION"].dropna().unique())
        div_sel = st.selectbox("Division", ["All"] + lookup_divs)

    subdiv_sel = None
    if "SUBDIV" in edges.columns:
        lookup_subdivs = sorted(edges["SUBDIV"].dropna().unique())
        subdiv_sel = st.selectbox("Subdivision", ["All"] + lookup_subdivs)

    # Apply filters
    filtered_nodes = nodes.copy()

    if state_sel != "All":
        filtered_nodes = filtered_nodes[filtered_nodes["STATE"] == state_sel]

    if div_sel and div_sel != "All":
        edge_nodes_div = edges.loc[edges["DIVISION"] == div_sel, ["FRFRANODE", "TOFRANODE"]]
        valid_node_ids_div = set(edge_nodes_div["FRFRANODE"].astype(str)) | set(edge_nodes_div["TOFRANODE"].astype(str))
        filtered_nodes = filtered_nodes[filtered_nodes["FRANODEID"].astype(str).isin(valid_node_ids_div)]

    if subdiv_sel and subdiv_sel != "All":
        # Nodes must appear in edges with matching SUBDIV
        edge_nodes = edges.loc[edges["SUBDIV"] == subdiv_sel, ["FRFRANODE", "TOFRANODE"]]
        valid_node_ids = set(edge_nodes["FRFRANODE"].astype(str)) | set(edge_nodes["TOFRANODE"].astype(str))
        filtered_nodes = filtered_nodes[filtered_nodes["FRANODEID"].astype(str).isin(valid_node_ids)]

    st.write(f"**Matches:** {len(filtered_nodes)}")
    st.dataframe(filtered_nodes[["FRANODEID", "STATE", "x", "y"]], use_container_width=True)

    # Allow user to copy a node into input
    selected_node = st.selectbox("Select node to copy:", filtered_nodes["FRANODEID"].astype(str).tolist())

    colA, colB = st.columns(2)
    with colA:
        if st.button("Copy to Start Node"):
            st.session_state["start_node"] = selected_node
    with colB:
        if st.button("Copy to End Node"):
            st.session_state["end_node"] = selected_node


avoid_nodes_input = st.sidebar.text_input("Nodes to avoid (comma-separated 6-digit IDs)", "")
avoid_nodes = [n.strip() for n in avoid_nodes_input.split(",") if n.strip()]
base_speed = st.sidebar.number_input("Base Speed (mph)")
div_speed = st.sidebar.number_input("Diversion Speed (mph)")
fuel_cost_per_mile =st.sidebar.number_input("Fuel Cost per Mile")
labor_cost_per_mile=st.sidebar.number_input("Labor Cost per Mile")

#edges_to_remove =[]

# --- Allowed Owner Selection ---
trk_cols = [f"TRKRGHTS{i}" for i in range(1, 10)]
unique_owners = pd.unique(edges[trk_cols].values.ravel())
unique_owners = [o for o in unique_owners if isinstance(o, str) and o.strip()]
unique_owners.sort()

allowed_owner = st.sidebar.multiselect("Allowed Owner (Railroad)", ["All"] + unique_owners)

# Compute and plot
# --- Compute and plot ---
if st.sidebar.button("Compute Paths"):
    if not (start_node and end_node):
        st.warning("Please enter both start and end nodes.")
    else:
        # --- Create working copy of graph (preserve cached G) ---
        G_temp = G.copy()


        if not show_network and allowed_owner != "All":
            allowed_edges = get_allowed_edges(edges, allowed_owner, trk_cols)
            if allowed_edges:
                edges_to_remove = [
                    (u, v) for u, v in G_temp.edges()
                    if (str(u).strip(), str(v).strip()) not in allowed_edges
                ]
                G_temp.remove_edges_from(edges_to_remove)

        # --- Apply allowed owner filter (cached via get_allowed_edges) ---
        allowed_edges = get_allowed_edges(edges, allowed_owner, trk_cols)  # accepts list or "All"
        if allowed_edges is not None:
            # remove any edge in G_temp that is NOT in allowed_edges
            removed = []
            for u, v in list(G_temp.edges()):
                if (str(u).strip(), str(v).strip()) not in allowed_edges:
                    removed.append((u, v))
            if removed:
                G_temp.remove_edges_from(removed)
            st.info(f"Ownership filter applied: removed {len(removed)} edges for {allowed_owner}.")
        else:
            st.info("No ownership filter applied (All).")

        # Save filtered graph to session (for debugging / display)
        st.session_state["filtered_graph"] = G_temp
        st.session_state["allowed_edges"] = allowed_edges

        # --- Compute base path on filtered graph (IGNORE avoid nodes for base) ---
        try:
            base_path = nx.shortest_path(G_temp, str(start_node).strip(), str(end_node).strip(), weight="weight")
            base_distance = nx.shortest_path_length(G_temp, str(start_node).strip(), str(end_node).strip(), weight="weight")
            st.session_state["base_path"] = base_path
            st.session_state["base_distance"] = base_distance
        except nx.NodeNotFound as e:
            st.error(f"Invalid start/end node: {e}. Make sure node IDs exist in the network (after owner filter).")
            base_path = base_distance = None
        except nx.NetworkXNoPath:
            st.warning("⚠️ Base path cannot be computed on the filtered network (ownership restriction).")
            base_path = base_distance = None
        except Exception as e:
            st.error(f"Unexpected error computing base path: {e}")
            base_path = base_distance = None

        # --- Compute diversion path ONLY if user provided avoid nodes ---
        diversion_path = None
        diversion_distance = None
        if avoid_nodes:
            # normalize avoid node ids to strings and strip
            avoid_list = [str(n).strip() for n in avoid_nodes_input.split(",") if str(n).strip()]
            # check existence in the filtered graph BEFORE removal
            missing = [n for n in avoid_list if n not in G_temp.nodes()]
            if missing:
                st.warning(f"The following avoid-node(s) are not present in the filtered graph and cannot be removed: {missing}")
                # We continue: we will remove only nodes that exist
                avoid_list = [n for n in avoid_list if n in G_temp.nodes()]

            if avoid_list:
                G_div = G_temp.copy()
                G_div.remove_nodes_from(avoid_list)
                st.info(f"Removed avoid-nodes from diversion graph: {avoid_list}")
                # Now attempt shortest path on G_div
                try:
                    diversion_path = nx.shortest_path(G_div, str(start_node).strip(), str(end_node).strip(), weight="weight")
                    diversion_distance = nx.shortest_path_length(G_div, str(start_node).strip(), str(end_node).strip(), weight="weight")
                    st.session_state["diversion_path"] = diversion_path
                    st.session_state["diversion_distance"] = diversion_distance
                except nx.NodeNotFound as e:
                    st.error(f"Invalid start/end node for diversion: {e}")
                    diversion_path = diversion_distance = None
                except nx.NetworkXNoPath:
                    st.warning("No diversion path found after removing the avoided node(s).")
                    diversion_path = diversion_distance = None
                except Exception as e:
                    st.error(f"Unexpected error computing diversion path: {e}")
                    diversion_path = diversion_distance = None
            else:
                # avoid_list ended up empty (no nodes removed)
                st.info("No avoidable nodes were present in the filtered graph; diversion not computed.")
                diversion_path = diversion_distance = None
        else:
            st.info("No avoid nodes provided; diversion will not be computed.")
            diversion_path = diversion_distance = None

        # --- Validate speeds before cost/time calculations ---
        if base_path and (base_speed is None or base_speed <= 0):
            st.error("Please enter a positive Base Speed (mph) to compute times/costs.")
        elif avoid_nodes and (diversion_distance is not None) and (div_speed is None or div_speed <= 0):
            st.error("Please enter a positive Diversion Speed (mph) to compute diversion times/costs.")
        else:
            # --- Compute costs & times (if values exist) ---
            if base_path:
                base_time = base_distance / base_speed
                base_fuel = base_distance * fuel_cost_per_mile
                base_labor = base_distance * labor_cost_per_mile
            else:
                base_time = base_fuel = base_labor = 0

            if diversion_distance:
                div_time = diversion_distance / div_speed
                div_fuel = diversion_distance * fuel_cost_per_mile
                div_labor = diversion_distance * labor_cost_per_mile
            else:
                div_time = div_fuel = div_labor = 0

            # --- Build display results and store in session state ---
            # --- Plot and store results ---
            
            
            # If the user wants to see the full network, draw it first
            if show_network:
                m = plot_full_network(G)   # Use filtered graph version
            else:
                m = folium.Map(location=[45, -95], zoom_start=5, tiles="CartoDB positron")
            
            # Then overlay the actual calculated paths
            m = plot_paths(G_temp, base_path, diversion_path, show_network)
            
            # Store in session_state so it stays visible after refresh
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
                }
            }
            st.session_state["map"] = m


        # Debug: show session keys (temporary - remove later)
        st.debug = getattr(st, "debug", None)
        st.write("Session state keys:", list(st.session_state.keys()))


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
        st.rerun()
# --- Display the map persistently ---
if "map" in st.session_state:
    st_folium(st.session_state["map"], width=1200, height=700)

# --- Export to Excel ---
if "results" in st.session_state and "map" in st.session_state and st.button("Export Results to Excel"):
    res = st.session_state["results"]
    base_path = st.session_state.get("base_path", [])
    diversion_path = st.session_state.get("diversion_path", [])

    # --- Create Excel workbook in memory ---
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # 1️⃣ Summary sheet
        summary_data = {
            "Path Type": ["Base", "Diversion"],
            "Distance (miles)": [
                res["base"]["distance"],
                res["diversion"]["distance"],
            ],
            "Average Speed (mph)": [
                res["base"]["speed"],
                res["diversion"]["speed"],
            ],
            "Travel Time (hours)": [
                res["base"]["time"],
                res["diversion"]["time"],
            ],
            "Fuel Cost ($)": [
                res["base"]["fuel"],
                res["diversion"]["fuel"],
            ],
            "Labor Cost ($)": [
                res["base"]["labor"],
                res["diversion"]["labor"],
            ],
        }
        pd.DataFrame(summary_data).to_excel(writer, index=False, sheet_name="Summary")

        # --- Convert node list to dataframe with coordinates ---
        def path_to_df(path, nodes_df):
            if not path:
                return pd.DataFrame(columns=["NodeID", "Latitude", "Longitude"])
        
            # Ensure string match
            nodes_df = nodes_df.copy()
            nodes_df["FRANODEID"] = nodes_df["FRANODEID"].astype(str).str.strip()
        
            rows = []
            for node in path:
                row = nodes_df[nodes_df["FRANODEID"] == str(node)]
                if not row.empty:
                    lat, lon = row.iloc[0]["y"], row.iloc[0]["x"]
                    rows.append([node, lat, lon])
                else:
                    rows.append([node, None, None])
        
            return pd.DataFrame(rows, columns=["NodeID", "Latitude", "Longitude"])
        
        
        # Create one sheet per path
        base_df = path_to_df(base_path, nodes)
        base_df.to_excel(writer, index=False, sheet_name="Base_Path")
        
        if diversion_path:
            diversion_df = path_to_df(diversion_path, nodes)
            diversion_df.to_excel(writer, index=False, sheet_name="Diversion_Path")


        # 3️⃣ Optional — Map image
        try:
            m = st.session_state["map"]
            m.save("temp_map.html")
            import tempfile, imgkit

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                imgkit.from_file("temp_map.html", tmp_img.name)
                ws = writer.book.create_sheet("Map")
                img = XLImage(tmp_img.name)
                ws.add_image(img, "A1")
        except Exception as e:
            st.warning(f"Map export skipped (wkhtmltopdf/imgkit not installed): {e}")

    output.seek(0)

    # --- Provide download button ---
    st.download_button(
        label="📊 Download Excel File",
        data=output,
        file_name="RailMileageResults.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
