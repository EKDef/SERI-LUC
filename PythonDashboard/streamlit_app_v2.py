# streamlit_app_v4.py — Fast MSOA explorer using GeoParquet
# =============================================================================
# Streamlit + Pydeck choropleth with AgGrid table. Performance tweaks:
#   • Uses GeoParquet for much faster data loading
#   • Each numeric column is colourised ONCE and cached with `@st.cache_data`
#   • Variable switch = replace layer only (no Python loop)
#
# Launch:
#   streamlit run streamlit_app_v4.py
#
# Requirements:
#   pip install streamlit pandas geopandas pydeck shapely pyarrow geoparquet streamlit-aggrid>=1.1.3
# =============================================================================

import json
from pathlib import Path

import pandas as pd
import geopandas as gpd
import pydeck as pdk
import streamlit as st

# ── 1. Page config (must be first) ────────────────────────────────────────────
st.set_page_config(page_title="MSOA Explorer", layout="wide")

# Initialize global variable for MSOA selection
if 'selected_msoa_code' not in st.session_state:
    st.session_state.selected_msoa_code = None

# ── 2. Optional AgGrid import ────────────────────────────────────────────────
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

    AGGRID_OK = True
except Exception:
    AGGRID_OK = False

# ── 3. File paths (edit as needed) ───────────────────────────────────────────
DATA_DIR = Path("./PythonDashboard")
PARQUET_PATH = DATA_DIR / "msoa_2021_data.parquet"  # Using geoparquet instead of separate files


# ── 4. Load data from GeoParquet (cached) ───────────────────────────────────────────────
@st.cache_data(show_spinner="Loading MSOA data from GeoParquet…")
def load_geoparquet() -> gpd.GeoDataFrame:
    """Load the GeoParquet file containing both geometry and attributes."""
    return gpd.read_parquet(PARQUET_PATH)


# Load data once - much faster with GeoParquet
GDF = load_geoparquet()

# Create attribute dataframe without geometry for table view
attr_df = GDF.drop(columns=['geometry'])

# Calculate map center
MINX, MINY, MAXX, MAXY = GDF.total_bounds
CENTRE_LON, CENTRE_LAT = (MINX + MAXX) / 2, (MINY + MAXY) / 2

# ── 5. Colour ramp helper (five‑class YlOrRd) ────────────────────────────────
COLORS = [
    [255, 255, 204], [255, 237, 160], [254, 217, 118], [253, 141, 60], [189, 0, 38]
]


def ramp(val: float, vmin: float, vmax: float):
    if pd.isna(val):
        return [200, 200, 200]
    t = (val - vmin) / (vmax - vmin + 1e-9)
    idx = max(0, min(4, int(t * 4)))
    return COLORS[idx]


# ── 6. Cache colourised GeoJSON per variable ───────────────────────────────
@st.cache_data(show_spinner=False)
def colourised_geojson(_gdf, col_name: str):
    """Return (geojson_dict, vmin, vmax) for the chosen variable.
    Using _gdf parameter with leading underscore to prevent hashing this unhashable argument."""
    vmin = float(_gdf[col_name].min())
    vmax = float(_gdf[col_name].max())

    # Convert directly from GDF to avoid property loss
    gj = json.loads(_gdf.to_json())

    # Process all features to add color
    for f in gj["features"]:
        val = f["properties"].get(col_name, float("nan"))
        f["properties"]["_color"] = ramp(val, vmin, vmax)

        # Format the value for display
        if not pd.isna(val):
            f["properties"][f"{col_name}_formatted"] = f"{val:.2f}"

    return gj, vmin, vmax


# ── 7. Sidebar controls ──────────────────────────────────────────────────────
st.sidebar.header("Controls")

# Map variable selection
NUMERIC_COLS = attr_df.select_dtypes("number").columns.tolist()
MAP_VAR = st.sidebar.selectbox("Variable for map", NUMERIC_COLS)

# MSOA search functionality - simple implementation that works
st.sidebar.markdown("### Search MSOAs")
search_term = st.sidebar.text_input("Search by name or code:")

# Process search
filtered_msoas = None
if search_term:  # Execute search on text input
    filter_cond = (
            GDF["MSOA21NM"].str.contains(search_term, case=False) |
            GDF["MSOA21CD"].str.contains(search_term, case=False)
    )
    filtered_msoas = GDF[filter_cond]

    # Show count of matches
    st.sidebar.write(f"Found {len(filtered_msoas)} matching MSOAs")

    # Allow selection from filtered list
    if not filtered_msoas.empty:
        msoa_options = filtered_msoas["MSOA21NM"].tolist()
        selected_msoa_name = st.sidebar.selectbox("Select MSOA:", ["None"] + msoa_options)

        if selected_msoa_name != "None":
            # Get the MSOA code for the selected name
            st.session_state.selected_msoa_code = filtered_msoas[
                filtered_msoas["MSOA21NM"] == selected_msoa_name
                ]["MSOA21CD"].iloc[0]

            # Highlight in info box
            st.sidebar.success(f"Selected: {selected_msoa_name}")
            st.sidebar.info(f"Code: {st.session_state.selected_msoa_code}")
        else:
            st.session_state.selected_msoa_code = None
    elif search_term:
        st.sidebar.warning("No matching MSOAs found.")

# Clear selection button
if st.session_state.selected_msoa_code:
    if st.sidebar.button("Clear Selection"):
        st.session_state.selected_msoa_code = None
        st.experimental_rerun()

# Table controls
st.sidebar.markdown("### Table Options")
ALL_COLS = attr_df.columns.tolist()

# Add a "Select All Columns" checkbox
select_all_cols = st.sidebar.checkbox("Select All Columns", value=False)

# Use a key variable to force multiselect to update when select_all changes
multiselect_key = "column_select_" + ("all" if select_all_cols else "custom")

if select_all_cols:
    # When "Select All" is checked, pre-select all columns but allow deselection
    SHOWN_COLS = st.sidebar.multiselect(
        "Deselect columns to hide",
        options=ALL_COLS,
        default=ALL_COLS,
        key=multiselect_key
    )
    remaining_count = len(SHOWN_COLS)
    total_count = len(ALL_COLS)
    st.sidebar.write(f"Showing {remaining_count} of {total_count} columns")
else:
    # Default selection when not selecting all
    default_cols = ["MSOA21CD", "MSOA21NM"] + NUMERIC_COLS[:3]
    # Make sure default columns exist in the data
    default_cols = [col for col in default_cols if col in ALL_COLS]

    SHOWN_COLS = st.sidebar.multiselect(
        "Columns visible in table",
        options=ALL_COLS,
        default=default_cols,
        key=multiselect_key
    )

# Map style selector
st.sidebar.markdown("### Map Options")
map_style = st.sidebar.selectbox(
    "Map style",
    ["mapbox://styles/mapbox/light-v9", "mapbox://styles/mapbox/dark-v9", None],
    format_func=lambda x: "Light" if x == "mapbox://styles/mapbox/light-v9" else
    "Dark" if x == "mapbox://styles/mapbox/dark-v9" else "No basemap",
    index=2  # Default to no basemap for speed
)

# Add performance mode option
perf_mode = st.sidebar.checkbox("High Performance Mode", value=True,
                                help="Simplifies rendering for better performance")

# ── 8. Prepare colourised data & legend stats ────────────────────────────────
geo_src, VMIN, VMAX = colourised_geojson(GDF, MAP_VAR)
Q1 = float(GDF[MAP_VAR].quantile(0.25))
MEDIAN = float(GDF[MAP_VAR].median())
MEAN = float(GDF[MAP_VAR].mean())
Q3 = float(GDF[MAP_VAR].quantile(0.75))

# ── 9. Layout (65% table | 35% map) ───────────────────────────────────────
left, right = st.columns([65, 35], gap="small")

# ----- TABLE --------------------------------------------------------------
with left:
    st.subheader("MSOA data table")

    # If there are search results, provide an option to filter the table
    display_df = attr_df
    if filtered_msoas is not None and not filtered_msoas.empty and search_term:
        show_filtered = st.checkbox("Show only search results in table", value=True)
        if show_filtered:
            display_df = attr_df[attr_df["MSOA21CD"].isin(filtered_msoas["MSOA21CD"])]

    if AGGRID_OK:
        # Generate a unique key for the table to avoid duplicates
        table_key = f"table_{len(display_df)}_{len(SHOWN_COLS)}_{hash(tuple(SHOWN_COLS))}"

        gb = GridOptionsBuilder.from_dataframe(display_df[SHOWN_COLS])
        gb.configure_default_column(
            resizable=True, filter=True, sortable=True,
            wrapHeaderText=True, autoHeaderHeight=True,
        )
        gb.configure_selection("single")
        table_resp = AgGrid(
            display_df[SHOWN_COLS],
            gridOptions=gb.build(),
            height=800,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            theme="alpine",
            key=table_key
        )
        sel_rows = table_resp["selected_rows"]
        # Only override selected_msoa_code if something is selected in the table
        if sel_rows and "MSOA21CD" in sel_rows[0]:
            st.session_state.selected_msoa_code = sel_rows[0]["MSOA21CD"]
    else:
        st.info("AgGrid not installed – showing basic table.")
        # Simple selectbox for MSOA selection
        table_selection = st.selectbox("Highlight MSOA", [None] + display_df["MSOA21CD"].tolist())
        if table_selection:
            st.session_state.selected_msoa_code = table_selection

        st.dataframe(display_df[SHOWN_COLS], height=800)

# ----- MAP ----------------------------------------------------------------
with right:
    st.subheader("Choropleth map")

    # Build layers (highlight optional)
    layer_settings = {
        "get_fill_color": "properties._color",
        "get_line_color": [0, 0, 0, 40],
        "line_width_min_pixels": 0.5,
        "pickable": True,
    }

    # Add performance optimizations if high performance mode is enabled
    if perf_mode:
        layer_settings.update({
            "opacity": 0.9
        })

    layers = [
        pdk.Layer(
            "GeoJsonLayer",
            geo_src,
            **layer_settings
        )
    ]

    # Add highlight layer if MSOA is selected
    center_lon, center_lat = CENTRE_LON, CENTRE_LAT
    map_zoom = 5

    if st.session_state.selected_msoa_code:
        sel = GDF[GDF["MSOA21CD"] == st.session_state.selected_msoa_code]
        if not sel.empty:
            # Create a more efficient highlight layer
            highlight_data = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": json.loads(sel.iloc[0]["geometry"].to_json()),
                    "properties": {}
                }]
            }

            # Add highlight layer
            layers.append(
                pdk.Layer(
                    "GeoJsonLayer",
                    highlight_data,
                    get_fill_color=[0, 0, 0, 0],
                    get_line_color=[0, 255, 255],
                    line_width_min_pixels=3,
                    pickable=False
                )
            )

            # Update map center to focus on the selected MSOA
            minx, miny, maxx, maxy = sel.total_bounds
            center_lon, center_lat = (minx + maxx) / 2, (miny + maxy) / 2
            map_zoom = 9

            # Add info about selected MSOA
            msoa_name = sel["MSOA21NM"].iloc[0]
            msoa_value = sel[MAP_VAR].iloc[0]

            st.info(f"Selected: {msoa_name} ({st.session_state.selected_msoa_code})\n\n"
                    f"{MAP_VAR}: {msoa_value:.2f}")

    # Tooltip with formatted value for better display
    tooltip_html = f"""
    <div style="background-color: rgba(42, 42, 42, 0.95); color: white; padding: 10px; border-radius: 3px;">
      <b>{{MSOA21NM}}</b><br/>
      <b>Code:</b> {{MSOA21CD}}<br/>
      <b>{MAP_VAR}:</b> {{{MAP_VAR}_formatted}}
    </div>
    """

    # Create deck with optimized settings - using only supported parameters
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            longitude=center_lon,
            latitude=center_lat,
            zoom=map_zoom,
            pitch=0,
            bearing=0
        ),
        map_style=map_style,
        tooltip={"html": tooltip_html, "style": {"z-index": "1"}}
    )

    # Display the map
    st.pydeck_chart(deck, use_container_width=True, height=800)

    # ----- Legend below map ---------------------------------------------
    legend_blocks = "".join(
        f"<div style='flex:1;background:rgb({r},{g},{b});'></div>" for r, g, b in COLORS
    )
    labels_row = (
        f"<div style='display:flex;justify-content:space-between;font-size:0.8em'>"
        f"<span>{VMIN:.1f}</span><span>{Q1:.1f}</span><span>{MEDIAN:.1f}</span><span>{Q3:.1f}</span><span>{VMAX:.1f}</span></div>"
    )
    legend_html = (
            "<div style='width:100%;margin-top:4px'>"
            "<div style='display:flex;height:16px;border:1px solid #ccc'>" + legend_blocks + "</div>" +
            labels_row +
            f"<div style='font-size:0.8em;margin-top:2px;'>Mean: {MEAN:.1f}</div>" +
            "</div>"
    )
    st.markdown(legend_html, unsafe_allow_html=True)

    # Performance info
    with st.expander("Performance Information", expanded=False):
        st.write("### Data Source")
        st.success("✅ Using GeoParquet for optimal performance")
        st.write(f"• Features: {len(GDF)} MSOAs")
        st.write(f"• Attributes: {len(ALL_COLS)} columns")

        if perf_mode:
            st.success("✅ Using high performance rendering mode")
        else:
            st.warning("⚠️ Standard rendering mode (enable high performance mode for faster map)")