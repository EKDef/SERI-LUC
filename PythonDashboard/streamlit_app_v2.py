# streamlit_app_v5.py — Fast MSOA explorer using GeoParquet with Map Toggle
# =============================================================================
# Streamlit + Pydeck choropleth with AgGrid table. Performance tweaks:
#   • Uses GeoParquet for much faster data loading
#   • Supports pre-computed LOD versions for even faster loading
#   • Each numeric column is colourised ONCE and cached with `@st.cache_data`
#   • Variable switch = replace layer only (no Python loop)
#   • Added map toggle functionality to only render map when needed
#   • Added layout slider to resize table vs map proportions
#   • Fixed LOD (Level of Detail) selection with direct manual control
#   • Special coloring for outliers to better visualize data distribution
#
# File structure:
#   PythonDashboard/
#   ├── parquet/               # Main parquet files directory
#   │   ├── dataset1.parquet
#   │   └── dataset2.parquet
#   └── lod_versions/          # Pre-computed LOD versions
#       ├── dataset1/
#       │   ├── medium.parquet
#       │   └── low.parquet
#       └── dataset2/
#           ├── medium.parquet
#           └── low.parquet
#
# Launch:
#   streamlit run streamlit_app_v5.py
#
# Requirements:
#   pip install streamlit pandas geopandas pydeck shapely pyarrow geoparquet streamlit-aggrid>=1.1.3
# =============================================================================

import json
import os
from pathlib import Path

import pandas as pd
import geopandas as gpd
import pydeck as pdk
import streamlit as st

# ── 1. Page config (must be first) ────────────────────────────────────────────
# Must be the very first Streamlit command
st.set_page_config(page_title="Geographic Data Explorer", layout="wide")

# ── 2. Define data directory and list datasets ────────────────────────────────
BASE_DIR = Path("./PythonDashboard")
DATA_DIR = BASE_DIR / "parquet"
LOD_DIR = BASE_DIR / "lod_versions"


# Get dataset files without using cache_data (since it would be a Streamlit command)
def list_geoparquet_files(directory):
    """List all .parquet files in the specified directory."""
    if not directory.exists():
        return []
    parquet_files = list(directory.glob("*.parquet"))
    return [file.name for file in parquet_files]


# Check if a dataset has LOD versions available
def has_lod_versions(dataset_name):
    """Check if the dataset has LOD versions available in the lod_versions folder."""
    dataset_lod_dir = LOD_DIR / dataset_name.replace(".parquet", "")
    print(f"Checking for LOD versions at: {dataset_lod_dir}")

    # First check if the directory exists
    if not dataset_lod_dir.exists():
        print(f"LOD directory doesn't exist: {dataset_lod_dir}")
        return False

    # Check for the specific files with the correct naming pattern
    medium_exists = (dataset_lod_dir / "medium_detail.parquet").exists()
    low_exists = (dataset_lod_dir / "low_detail.parquet").exists()
    print(f"medium_detail.parquet exists: {medium_exists}, low_detail.parquet exists: {low_exists}")

    # List available files
    try:
        files = list(dataset_lod_dir.glob("*"))
        print(f"Files in {dataset_lod_dir}: {[f.name for f in files]}")
    except Exception as e:
        print(f"Error listing files: {e}")

    # Only return True if at least one of the LOD files exists
    return medium_exists or low_exists


# Get available datasets
dataset_options = list_geoparquet_files(DATA_DIR)

# Add indicators for datasets with LOD versions
dataset_options_with_lod = []
for dataset in dataset_options:
    if has_lod_versions(dataset):
        dataset_options_with_lod.append(f"{dataset} ✓")
    else:
        dataset_options_with_lod.append(dataset)

# Default to the first dataset if available, or use a placeholder
default_dataset = "msoa_2021_data.parquet"
default_index = 0

# Find default dataset in the list if it exists
if dataset_options:
    for i, filename in enumerate(dataset_options):
        if filename == default_dataset:
            default_index = i
            break
else:
    # Add a placeholder option if no datasets found
    dataset_options = ["No datasets found"]
    dataset_options_with_lod = ["No datasets found"]

# Initialize global variables in session state
if 'selected_msoa_code' not in st.session_state:
    st.session_state.selected_msoa_code = None

if 'show_map' not in st.session_state:
    st.session_state.show_map = True  # Default to showing map

if 'search_term' not in st.session_state:
    st.session_state.search_term = ""

if 'turn_map_on' not in st.session_state:
    st.session_state.turn_map_on = False

if 'layout_ratio' not in st.session_state:
    st.session_state.layout_ratio = 65  # Default split (65% table | 35% map)

if 'prev_dataset' not in st.session_state:
    st.session_state.prev_dataset = dataset_options[default_index]

# Initialize map state to track view state and user's previous zoom level
if 'map_view_state' not in st.session_state:
    st.session_state.map_view_state = {
        'zoom': 5,
        'latitude': 54.5,  # Center of UK approximately
        'longitude': -3.0,  # Center of UK approximately
        'pitch': 0,
        'bearing': 0
    }

# Initialize prev_zoom to track zoom level changes
if 'prev_zoom' not in st.session_state:
    st.session_state.prev_zoom = 5  # Default zoom


# Function to clear selection
def clear_selection():
    st.session_state.selected_msoa_code = None
    st.session_state.search_term = ""
    # Don't change map visibility
    st.rerun()


# Function to toggle map on
def show_map_on():
    # Instead of directly modifying show_map, we'll set a flag to update it
    st.session_state.turn_map_on = True


# Function to handle map view state changes
def handle_view_state_change(view_state):
    """Update map view state when it changes."""
    # Check if view_state is a valid dict with expected keys
    if isinstance(view_state, dict) and 'zoom' in view_state:
        # Update the session state with the new view state
        st.session_state.map_view_state = view_state
        # Also update prev_zoom after we've used it
        st.session_state.prev_zoom = view_state.get('zoom', 5)


# Function to determine which detail level to use based on zoom
def get_detail_level_for_zoom(zoom, selected_detail="Auto"):
    """Return the appropriate detail level based on zoom and user settings."""
    # Convert zoom to float for safety
    try:
        zoom_level = float(zoom)
    except (ValueError, TypeError):
        zoom_level = 5.0  # Default if conversion fails

    # Directly map detail level selections to the appropriate level
    # This function is kept for backward compatibility but simplified
    if selected_detail == "High":
        return "original"
    elif selected_detail == "Medium":
        return "medium"
    elif selected_detail == "Low":
        return "low"
    else:
        return "medium"  # Default to medium detail


# ── 3. Optional AgGrid import ────────────────────────────────────────────────
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

    AGGRID_OK = True
except Exception:
    AGGRID_OK = False


# ── 4. Load data from GeoParquet with simplified versions (cached) ───────────────────────
@st.cache_data(show_spinner="Loading geographic data from GeoParquet…")
def load_geoparquet_with_simplifications(file_path) -> dict:
    """Load the GeoParquet file and create simplified versions for different zoom levels.

    This function first checks if pre-computed LOD versions exist in the lod_versions folder.
    If they do, it loads them directly. Otherwise, it computes them on the fly.
    """
    try:
        # Determine the base dataset name (without extension)
        dataset_name = Path(file_path).name
        dataset_base = dataset_name.replace(".parquet", "")
        lod_dir = LOD_DIR / dataset_base

        # Debug info about paths
        print(f"Loading dataset: {dataset_name}")
        print(f"Full path: {file_path}")
        print(f"Looking for LOD versions in: {lod_dir}")

        # Create a dictionary to store different detail levels
        detail_levels = {}

        # First, load the original (high detail) version from the main folder
        original_gdf = gpd.read_parquet(file_path)
        detail_levels["original"] = original_gdf

        # Check if LOD versions are available in the lod_versions folder
        if lod_dir.exists():
            # Check for and load medium detail - using correct filename
            medium_path = lod_dir / "medium_detail.parquet"
            print(f"Checking for medium LOD at: {medium_path}")
            if medium_path.exists() and medium_path.is_file():
                try:
                    detail_levels["medium"] = gpd.read_parquet(medium_path)
                except Exception as e:
                    st.sidebar.error(f"Error loading medium LOD: {e}")
                    detail_levels["medium"] = original_gdf
            else:
                detail_levels["medium"] = original_gdf
                st.sidebar.warning(f"Medium detail version not found at {medium_path}, using original")

                # List available files in the LOD directory
                try:
                    if lod_dir.exists():
                        files = list(lod_dir.glob("*"))
                        st.sidebar.info(f"Available files in LOD dir: {[f.name for f in files]}")
                except Exception as e:
                    st.sidebar.warning(f"Error listing files: {e}")

            # Check for and load low detail - using correct filename
            low_path = lod_dir / "low_detail.parquet"
            print(f"Checking for low LOD at: {low_path}")
            if low_path.exists() and low_path.is_file():
                try:
                    detail_levels["low"] = gpd.read_parquet(low_path)
                except Exception as e:
                    st.sidebar.error(f"Error loading low LOD: {e}")
                    detail_levels["low"] = original_gdf
            else:
                detail_levels["low"] = original_gdf
                st.sidebar.warning(f"Low detail version not found at {low_path}, using original")

            # If at least one LOD file was loaded, consider it a success
            if (detail_levels["medium"] is not original_gdf) or (detail_levels["low"] is not original_gdf):
                st.sidebar.success(f"Pre-computed LOD for {dataset_name}")
            else:
                # Create simplified versions if no LOD files were found
                st.sidebar.info(f"No valid LOD files found for {dataset_name}, generating on-the-fly")
                if len(original_gdf) > 100:
                    try:
                        # Medium detail (good for mid-level zoom)
                        medium_gdf = original_gdf.copy()
                        medium_gdf['geometry'] = medium_gdf['geometry'].simplify(tolerance=0.001,
                                                                                 preserve_topology=True)
                        detail_levels["medium"] = medium_gdf

                        # Low detail (good for far zoom)
                        low_gdf = original_gdf.copy()
                        low_gdf['geometry'] = low_gdf['geometry'].simplify(tolerance=0.005, preserve_topology=True)
                        detail_levels["low"] = low_gdf

                        st.sidebar.success(f"✅ On-the-fly multi-resolution rendering enabled")
                    except Exception as e:
                        st.sidebar.warning(f"Could not create simplified geometries: {e}")
                        detail_levels["medium"] = original_gdf
                        detail_levels["low"] = original_gdf
                else:
                    detail_levels["medium"] = original_gdf
                    detail_levels["low"] = original_gdf

            # Calculate and display size reductions
            try:
                orig_size = len(original_gdf.geometry.to_wkt().sum())
                med_size = len(detail_levels["medium"].geometry.to_wkt().sum())
                low_size = len(detail_levels["low"].geometry.to_wkt().sum())

                print(f"Original size: {orig_size}, Medium: {med_size}, Low: {low_size}")
                print(
                    f"Reduction - Medium: {(1 - med_size / orig_size) * 100:.1f}%, Low: {(1 - low_size / orig_size) * 100:.1f}%")

            except Exception as e:
                st.sidebar.warning(f"Could not calculate size reductions: {e}")

        else:
            # Fall back to on-the-fly generation if no LOD versions are available
            st.sidebar.info(f"No pre-computed LOD versions for {dataset_name}, generating on-the-fly")

            # Create simplified versions if the dataset is large enough
            if len(original_gdf) > 100:  # Only create simplified versions for larger datasets
                try:
                    # Medium detail (good for mid-level zoom)
                    medium_gdf = original_gdf.copy()
                    medium_gdf['geometry'] = medium_gdf['geometry'].simplify(tolerance=0.001, preserve_topology=True)
                    detail_levels["medium"] = medium_gdf

                    # Low detail (good for far zoom)
                    low_gdf = original_gdf.copy()
                    low_gdf['geometry'] = low_gdf['geometry'].simplify(tolerance=0.005, preserve_topology=True)
                    detail_levels["low"] = low_gdf

                    # Print/log actual simplification results
                    orig_size = len(original_gdf.geometry.to_wkt().sum())
                    med_size = len(medium_gdf.geometry.to_wkt().sum())
                    low_size = len(low_gdf.geometry.to_wkt().sum())

                    st.sidebar.success(f"✅ On-the-fly multi-resolution rendering enabled")
                    print(f"Original size: {orig_size}, Medium: {med_size}, Low: {low_size}")
                    print(
                        f"Reduction - Medium: {(1 - med_size / orig_size) * 100:.1f}%, Low: {(1 - low_size / orig_size) * 100:.1f}%")

                except Exception as e:
                    st.sidebar.warning(f"Could not create simplified geometries: {e}")
                    # If simplification fails, just use the original for all levels
                    detail_levels["medium"] = original_gdf
                    detail_levels["low"] = original_gdf
            else:
                # For small datasets, use the original for all detail levels
                detail_levels["medium"] = original_gdf
                detail_levels["low"] = original_gdf

        return detail_levels

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        # Return empty GeoDataFrames if there's an error
        empty_gdf = gpd.GeoDataFrame()
        return {"original": empty_gdf, "medium": empty_gdf, "low": empty_gdf}


# ── 5. Colour ramp helper (Viridis - colorblind friendly palette) ────────────────────────────────
# Viridis-inspired color palette (accessible, colorblind-friendly)
COLORS = [
    [68, 1, 84, 240],  # Dark purple (with transparency)
    [59, 82, 139, 240],  # Blue-purple
    [33, 144, 141, 240],  # Teal
    [93, 201, 99, 240],  # Green
    [253, 231, 37, 240]  # Yellow
]

# Special colors for outliers - using more distinct colors
OUTLIER_COLORS = {
    "low": [150, 0, 0, 240],  # Dark red for bottom 1%
    "high": [255, 165, 0, 240]  # Orange for top 1%
}


def ramp(val: float, vmin: float, vmax: float, p01: float, p99: float):
    """
    Return color for value, with special colors for outliers.

    Args:
        val: The value to color
        vmin: The minimum value in the dataset
        vmax: The maximum value in the dataset
        p01: The 1st percentile value
        p99: The 99th percentile value
    """
    if pd.isna(val):
        return [200, 200, 200, 100]  # Light gray with transparency for missing values

    # Check for outliers
    if val <= p01:
        return OUTLIER_COLORS["low"]  # Bottom 1% outlier
    elif val >= p99:
        return OUTLIER_COLORS["high"]  # Top 1% outlier

    # For the middle 98%, use the regular color ramp
    # Normalize the value between p01 and p99 (rather than min and max)
    t = (val - p01) / (p99 - p01 + 1e-9)
    t = max(0, min(1, t))  # Clamp to [0, 1]
    idx = int(t * 4)  # Map to one of the 5 colors (0-4)
    idx = max(0, min(4, idx))  # Safety check
    return COLORS[idx]


# ── 6. Cache colourised GeoJSON per variable ───────────────────────────────
@st.cache_data(show_spinner=False)
def colourised_geojson(_gdf, col_name: str, detail_level: str):
    """Return (geojson_dict, vmin, vmax) for the chosen variable.
    Using _gdf parameter with leading underscore to prevent hashing this unhashable argument.
    Added detail_level parameter to ensure cache is unique per detail level."""
    # Calculate statistics including outlier thresholds
    vmin = float(_gdf[col_name].min())
    vmax = float(_gdf[col_name].max())
    p01 = float(_gdf[col_name].quantile(0.01))  # 1st percentile
    p99 = float(_gdf[col_name].quantile(0.99))  # 99th percentile

    # Calculate representative statistics excluding outliers
    # This makes the color ramp more representative of the majority of data
    trimmed_series = _gdf[col_name][(_gdf[col_name] > p01) & (_gdf[col_name] < p99)]
    trimmed_min = float(trimmed_series.min())
    trimmed_max = float(trimmed_series.max())
    trimmed_q1 = float(trimmed_series.quantile(0.25))
    trimmed_median = float(trimmed_series.median())
    trimmed_mean = float(trimmed_series.mean())
    trimmed_q3 = float(trimmed_series.quantile(0.75))

    # Calculate quartile boundaries for color transitions
    # These values represent where the color changes in the Viridis palette
    color_bounds = []
    for i in range(5):  # 5 colors = 4 boundaries
        q_val = float(trimmed_series.quantile(0.25 * i))
        color_bounds.append(q_val)

    # Convert directly from GDF to avoid property loss
    gj = json.loads(_gdf.to_json())

    # Process all features to add color
    for f in gj["features"]:
        val = f["properties"].get(col_name, float("nan"))
        f["properties"]["_color"] = ramp(val, vmin, vmax, p01, p99)

        # Format the value for display
        if not pd.isna(val):
            f["properties"][f"{col_name}_formatted"] = f"{val:.2f}"

        # Add outlier flag for display in tooltip
        if not pd.isna(val):
            if val <= p01:
                f["properties"]["_outlier_text"] = "<br/><b style='color: #bb0000'>⚠️ Outlier (bottom 1%)</b>"
            elif val >= p99:
                f["properties"]["_outlier_text"] = "<br/><b style='color: orange'>⚠️ Outlier (top 1%)</b>"
            else:
                f["properties"]["_outlier_text"] = ""

    # Return full stats dictionary to use in the legend and elsewhere
    stats = {
        "vmin": vmin,
        "vmax": vmax,
        "p01": p01,
        "p99": p99,
        "trimmed_min": trimmed_min,
        "trimmed_max": trimmed_max,
        "trimmed_q1": trimmed_q1,
        "trimmed_median": trimmed_median,
        "trimmed_mean": trimmed_mean,
        "trimmed_q3": trimmed_q3,
        "color_bounds": color_bounds
    }

    return gj, stats


# ── 7. Sidebar controls ──────────────────────────────────────────────────────
st.sidebar.header("Dataset Selection")
selected_dataset_with_lod = st.sidebar.selectbox(
    "Choose a dataset:",
    options=dataset_options_with_lod,
    index=default_index
)

# Strip the LOD indicator if present
selected_dataset = selected_dataset_with_lod.split(" ✓")[
    0] if " ✓" in selected_dataset_with_lod else selected_dataset_with_lod

# Handle case where no datasets are found
if selected_dataset == "No datasets found":
    st.error("No geoparquet datasets found in the directory.")
    st.info(f"Please add .parquet files to the {DATA_DIR} directory.")
    st.stop()

# Get the full path for the selected dataset
PARQUET_PATH = DATA_DIR / selected_dataset

# Reset selection when dataset changes
if st.session_state.prev_dataset != selected_dataset:
    st.session_state.selected_msoa_code = None
    st.session_state.search_term = ""
    st.session_state.prev_dataset = selected_dataset
    # Always reset to Low detail when changing datasets
    st.session_state.default_detail_level = "Low"

# Load data with multiple detail levels - much faster with GeoParquet
# FIXED: Changed function name to match the function we defined
detail_levels = load_geoparquet_with_simplifications(PARQUET_PATH)

# Handle empty dataframe case (if file couldn't be loaded)
if detail_levels["original"].empty:
    st.error(f"Could not load the selected dataset: {selected_dataset}")
    st.info("Please select a different dataset from the sidebar.")
    # Stop execution with a graceful error message
    st.stop()

# For convenience, store the original GDF
GDF = detail_levels["original"]

# Create attribute dataframe without geometry for table view
attr_df = GDF.drop(columns=['geometry'])

# Calculate map center
MINX, MINY, MAXX, MAXY = GDF.total_bounds
CENTRE_LON, CENTRE_LAT = (MINX + MAXX) / 2, (MINY + MAXY) / 2

# Update map center in view state
if 'map_view_state' not in st.session_state:
    st.session_state.map_view_state = {
        'zoom': 5,
        'latitude': CENTRE_LAT,
        'longitude': CENTRE_LON,
        'pitch': 0,
        'bearing': 0
    }

# Store default detail level in session state
if 'default_detail_level' not in st.session_state:
    st.session_state.default_detail_level = "Low"  # Default to Low detail for maximum performance

# Extract geographic ID and name columns - try to detect standard patterns
# Check for common column name patterns for ID and name fields
id_column = None
name_column = None

# Try to auto-detect ID column
for possible_id in ["MSOA21CD", "MSOA11CD", "LSOA21CD", "LSOA11CD", "LAD22CD", "LAD21CD", "LAD20CD", "OA21CD",
                    "OA11CD"]:
    if possible_id in GDF.columns:
        id_column = possible_id
        break

# Try to auto-detect name column
for possible_name in ["MSOA21NM", "MSOA11NM", "LSOA21NM", "LSOA11NM", "LAD22NM", "LAD21NM", "LAD20NM", "OA21NM",
                      "OA11NM"]:
    if possible_name in GDF.columns:
        name_column = possible_name
        break

# If no standard columns found, use the first string column as the name and first column as ID
if id_column is None:
    id_column = GDF.columns[0]

if name_column is None:
    # Find the first string column for the name
    for col in GDF.columns:
        if GDF[col].dtype == 'object' and col != id_column:
            name_column = col
            break
    # If still not found, just use the ID column
    if name_column is None:
        name_column = id_column

# Update title based on selected dataset
dataset_title = selected_dataset.replace(".parquet", "").replace("_", " ").title()
st.sidebar.title(f"{dataset_title} Explorer")

# Map variable selection
NUMERIC_COLS = attr_df.select_dtypes("number").columns.tolist()
MAP_VAR_key = f"map_var_{selected_dataset}"  # Use unique key for this dataset
MAP_VAR = st.sidebar.selectbox("Variable for map", NUMERIC_COLS, key=MAP_VAR_key)

# Reset detail level selection when variable changes
if 'prev_map_var' not in st.session_state:
    st.session_state.prev_map_var = MAP_VAR
elif st.session_state.prev_map_var != MAP_VAR:
    # Reset to default detail level (Low) when variable changes
    st.session_state.default_detail_level = "Low"
    st.session_state.prev_map_var = MAP_VAR

# MSOA search functionality - simple implementation that works
st.sidebar.markdown("### Search Areas")
search_term = st.sidebar.text_input("Search by name or code:", value=st.session_state.search_term, key="search_input")

# Process search
filtered_msoas = None
if search_term:  # Execute search on text input
    # Update the session state with current search term
    st.session_state.search_term = search_term
    try:
        filter_cond = (
                GDF[name_column].str.contains(search_term, case=False) |
                GDF[id_column].str.contains(search_term, case=False)
        )
        filtered_msoas = GDF[filter_cond]

        # Show count of matches
        st.sidebar.write(f"Found {len(filtered_msoas)} matching areas")

        # Allow selection from filtered list
        if not filtered_msoas.empty:
            area_options = filtered_msoas[name_column].tolist()
            selected_area_name = st.sidebar.selectbox("Select area:", ["None"] + area_options)

            if selected_area_name != "None":
                # Get the code for the selected name
                st.session_state.selected_msoa_code = filtered_msoas[
                    filtered_msoas[name_column] == selected_area_name
                    ][id_column].iloc[0]

                # Highlight in info box
                st.sidebar.success(f"Selected: {selected_area_name}")
                st.sidebar.info(f"Code: {st.session_state.selected_msoa_code}")
            else:
                st.session_state.selected_msoa_code = None
        elif search_term:
            st.sidebar.warning("No matching areas found.")
    except Exception as e:
        st.sidebar.error(f"Search error: {e}")
        st.sidebar.info("Try a different search term or check if the dataset has text columns.")

# Clear selection button
if st.session_state.selected_msoa_code:
    if st.sidebar.button("Clear Selection"):
        clear_selection()

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

# Layout controls in sidebar
st.sidebar.markdown("### Layout Settings")
# Use a simpler approach without columns for the layout slider
st.sidebar.write("📋 Table vs Map 🗺️ balance:")
# Slider for adjusting the layout
st.session_state.layout_ratio = st.sidebar.slider(
    "Drag to adjust layout proportion",
    min_value=30, max_value=85,
    value=st.session_state.layout_ratio,
    help="Drag left for more map space, right for more table space"
)

# Map controls
st.sidebar.markdown("### Map Options")


# Define callback for the checkbox
def map_toggle_callback():
    # The checkbox will automatically update st.session_state.show_map
    pass


show_map = st.sidebar.checkbox("Show Map", value=st.session_state.show_map,
                               key="show_map",
                               on_change=map_toggle_callback,
                               help="Toggle map visibility to improve performance")

# Only show these options if map is enabled
if show_map:
    # Map style options - add an option specifically for boundary focus
    map_style = st.sidebar.selectbox(
        "Map style",
        ["mapbox://styles/mapbox/light-v9", "mapbox://styles/mapbox/streets-v11", "mapbox://styles/mapbox/dark-v9",
         None],
        format_func=lambda x: "Light" if x == "mapbox://styles/mapbox/light-v9" else
        "Streets" if x == "mapbox://styles/mapbox/streets-v11" else
        "Dark" if x == "mapbox://styles/mapbox/dark-v9" else "No basemap",
        index=0  # Default to light map for better visibility
    )

    # Add detail level selector - simplified to just direct selection
    # Use a unique key to force re-rendering when MAP_VAR changes
    detail_level_key = f"detail_level_{MAP_VAR}"
    detail_level = st.sidebar.radio(
        "Map detail level",
        options=["High", "Medium", "Low"],
        index=2,  # Default to Low (index 2) for better performance
        horizontal=True,
        key=detail_level_key,
        help="Higher detail is more accurate but slower."
    )

    # Direct mapping to the detail levels
    detail_map = {
        "High": "original",
        "Medium": "medium",
        "Low": "low"
    }

    # Set current detail directly based on selection
    current_detail_name = detail_map.get(detail_level, "medium")

    # Show effective simplification
    if current_detail_name in detail_levels:
        original_size = len(detail_levels["original"].geometry.to_wkt().sum())
        current_size = len(detail_levels[current_detail_name].geometry.to_wkt().sum())
        if original_size > 0:
            reduction = (1 - current_size / original_size) * 100
            st.sidebar.caption(f"Geometry size: {reduction:.1f}% reduced from original")

    # Add performance mode option
    perf_mode = st.sidebar.checkbox(
        "High Performance Mode",
        value=True,
        help="Simplifies rendering for better performance"
    )

    # Add boundary emphasis control
    border_emphasis = st.sidebar.slider(
        "Boundary emphasis",
        min_value=0,
        max_value=100,
        value=50,
        help="Adjust how visible boundaries are"
    )

# ── 8. Prepare colourised data & legend stats (only when map is shown) ────────────────
if show_map:
    # Directly use the selected detail level
    current_detail = detail_map.get(detail_level, "medium")

    # Debug info
    st.sidebar.write(f"Debug: Using detail level: {current_detail}")
    st.sidebar.write(f"Available levels: {list(detail_levels.keys())}")

    # Use the appropriate GeoDataFrame based on detail level
    display_gdf = detail_levels[current_detail]

    # Debug info about GeoDataFrame
    st.sidebar.write(f"GeoDataFrame empty? {display_gdf.empty}")
    st.sidebar.write(f"GeoDataFrame shape: {display_gdf.shape}")

    # Generate colourised GeoJSON for the current detail level - include detail_level in cache key
    geo_src, stats = colourised_geojson(display_gdf, MAP_VAR, detail_level)

    # Debug info about GeoJSON
    st.sidebar.write(f"GeoJSON created: {geo_src is not None}")

    # Extract statistics from the returned dictionary
    VMIN = stats["vmin"]
    VMAX = stats["vmax"]
    P01 = stats["p01"]
    P99 = stats["p99"]
    Q1 = stats["trimmed_q1"]
    MEDIAN = stats["trimmed_median"]
    MEAN = stats["trimmed_mean"]
    Q3 = stats["trimmed_q3"]

# Create columns with the dynamic ratio from session state
left, right = st.columns([st.session_state.layout_ratio, 100 - st.session_state.layout_ratio], gap="small")

# ----- TABLE --------------------------------------------------------------
with left:
    st.subheader("Data Table")

    # If there are search results, provide an option to filter the table
    display_df = attr_df
    if filtered_msoas is not None and not filtered_msoas.empty and search_term:
        show_filtered = st.checkbox("Show only search results in table", value=True)
        if show_filtered:
            display_df = attr_df[attr_df[id_column].isin(filtered_msoas[id_column])]

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
        if sel_rows and id_column in sel_rows[0]:
            st.session_state.selected_msoa_code = sel_rows[0][id_column]
    else:
        st.info("AgGrid not installed – showing basic table.")
        # Simple selectbox for MSOA selection
        table_selection = st.selectbox("Highlight area", [None] + display_df[id_column].tolist())
        if table_selection:
            st.session_state.selected_msoa_code = table_selection

        st.dataframe(display_df[SHOWN_COLS], height=800)

# ----- MAP or MSOA INFO ----------------------------------------------------------------
with right:
    if not show_map:
        st.subheader("Map View Disabled")
        st.info(
            "Map is currently hidden to improve performance. Enable it from the sidebar to view the choropleth visualization.")

        # Show statistics about dataset
        st.write("### Dataset Summary")
        st.write(f"• Total areas: {len(GDF)}")
        st.write(f"• Available attributes: {len(ALL_COLS)}")

        # If an MSOA is selected, show its details
        if st.session_state.selected_msoa_code:
            sel = GDF[GDF[id_column] == st.session_state.selected_msoa_code]
            if not sel.empty:
                msoa_name = sel[name_column].iloc[0]

                st.write(f"### Selected Area Details")
                st.success(f"**{msoa_name}** ({st.session_state.selected_msoa_code})")

                # Create metrics for key variables
                st.write("### Key Metrics")
                metrics_cols = st.columns(2)
                for i, col_name in enumerate(NUMERIC_COLS[:6]):  # First 6 numeric columns
                    if col_name in sel.columns:
                        value = sel[col_name].iloc[0]
                        with metrics_cols[i % 2]:
                            st.metric(col_name, f"{value:.2f}")

                # Show a button to quickly enable the map for this MSOA
                if st.button("Show this area on map"):
                    show_map_on()
                    st.rerun()
        else:
            # Show overall stats about the variable
            st.write(f"### Statistics for: {MAP_VAR}")
            stats_cols = st.columns(2)

            with stats_cols[0]:
                st.metric("Minimum", f"{GDF[MAP_VAR].min():.2f}")
                st.metric("Median", f"{GDF[MAP_VAR].median():.2f}")
                st.metric("Maximum", f"{GDF[MAP_VAR].max():.2f}")

            with stats_cols[1]:
                st.metric("Mean", f"{GDF[MAP_VAR].mean():.2f}")
                st.metric("25th Percentile", f"{GDF[MAP_VAR].quantile(0.25):.2f}")
                st.metric("75th Percentile", f"{GDF[MAP_VAR].quantile(0.75):.2f}")

            # Show a button to enable the map
            if st.button("Enable Map View"):
                show_map_on()
                st.rerun()
    else:
        # REGULAR MAP CODE - Only runs when map is enabled
        st.subheader("Choropleth map")

        try:
            # Build layers (highlight optional)
            layer_settings = {
                "get_fill_color": "properties._color",
                "get_line_color": [100, 100, 100, 120],  # Subtle grey border
                "line_width_min_pixels": 0.8,  # Slightly thicker border
                "pickable": True,
                "opacity": 0.75,  # Default opacity for all rendering modes
            }

            # Add extra performance optimizations if high performance mode is enabled
            if perf_mode:
                layer_settings.update({
                    "material": False,  # Disable lighting effects
                    "get_elevation": 0  # Ensure flat rendering
                })

            # Calculate opacity and width based on slider
            border_alpha = int(1.5 * border_emphasis)  # Convert 0-100 to 0-150 opacity
            border_width = 0.4 + (border_emphasis / 100) * 1.0  # Scale width from 0.4 to 1.4

            # Add a second layer with just outlines for better visibility
            layers = [
                # Base colored layer
                pdk.Layer(
                    "GeoJsonLayer",
                    geo_src,
                    **layer_settings
                ),
                # Outline-only layer for better boundary visibility
                pdk.Layer(
                    "GeoJsonLayer",
                    geo_src,
                    get_fill_color=[0, 0, 0, 0],  # Transparent fill
                    get_line_color=[80, 80, 80, border_alpha],  # Medium grey border with user-controlled opacity
                    line_width_min_pixels=border_width,  # User-controlled width
                    pickable=False,  # No need for picking on the outline layer
                    opacity=1.0
                )
            ]

            # Add highlight layer if MSOA is selected
            center_lon, center_lat = CENTRE_LON, CENTRE_LAT
            map_zoom = 5

            if st.session_state.selected_msoa_code:
                sel = GDF[GDF[id_column] == st.session_state.selected_msoa_code]
                if not sel.empty:
                    # Create a more efficient highlight layer using GeoSeries to create valid GeoJSON
                    # Convert the single geometry to a GeoSeries first, then to GeoJSON
                    from shapely.geometry import mapping

                    highlight_data = {
                        "type": "FeatureCollection",
                        "features": [{
                            "type": "Feature",
                            "geometry": mapping(sel.iloc[0]["geometry"]),
                            "properties": {}
                        }]
                    }

                    # Add highlight layer
                    layers.append(
                        pdk.Layer(
                            "GeoJsonLayer",
                            highlight_data,
                            get_fill_color=[255, 255, 255, 0],  # Transparent fill
                            get_line_color=[255, 255, 255, 230],  # Bright white border
                            line_width_min_pixels=3,
                            pickable=False
                        )
                    )

                    # Update map center to focus on the selected area
                    minx, miny, maxx, maxy = sel.total_bounds
                    center_lon, center_lat = (minx + maxx) / 2, (miny + maxy) / 2
                    map_zoom = 9

                    # Add info about selected area
                    area_name = sel[name_column].iloc[0]
                    area_value = sel[MAP_VAR].iloc[0]

                    st.info(f"Selected: {area_name} ({st.session_state.selected_msoa_code})\n\n"
                            f"{MAP_VAR}: {area_value:.2f}")

            # Tooltip with simplified formatting for better display
            tooltip_html = {
                "html": "<b>{" + name_column + "}</b><br/>"
                                               "<b>Code:</b> {" + id_column + "}<br/>"
                                                                              "<b>" + MAP_VAR + ":</b> {" + MAP_VAR + "_formatted}<br/>"
                                                                                                                      "{_outlier_text}",
                "style": {
                    "backgroundColor": "rgba(42, 42, 42, 0.95)",
                    "color": "white",
                    "padding": "10px",
                    "borderRadius": "3px"
                }
            }

            # Ensure initial view state is properly set with direct values
            initial_view_state = pdk.ViewState(
                longitude=center_lon,
                latitude=center_lat,
                zoom=map_zoom,
                pitch=0,
                bearing=0
            )

            # Create deck with optimized settings
            deck = pdk.Deck(
                layers=layers,
                initial_view_state=initial_view_state,
                map_style=map_style,
                tooltip=tooltip_html
            )

            # Display the map
            st.pydeck_chart(deck, use_container_width=True, height=800)

            # Simple caption showing current detail level
            st.caption(f"Using {detail_level} detail level for map rendering ({MAP_VAR}).")

            # ----- Simplified Legend with Data Summary ---------------------------------------------
            st.write("### Color Legend")

            # Calculate evenly-spaced boundaries between p01 and p99 for the 5 colors
            color_thresholds = [P01]  # Start with the lower outlier threshold
            for i in range(1, 5):  # 4 internal boundaries for 5 color bands
                boundary = P01 + (P99 - P01) * (i / 5)
                color_thresholds.append(boundary)
            color_thresholds.append(P99)  # End with the upper outlier threshold

            # Create a streamlined legend with just color bands and their boundary values
            cols = st.columns([1, 3, 3, 3, 3, 3, 1])

            # Color blocks with labels beneath
            colors = [
                "rgb(150,0,0)",  # Outlier - dark red
                "rgb(68,1,84)",  # Viridis color 1 - dark purple
                "rgb(59,82,139)",  # Viridis color 2 - blue-purple
                "rgb(33,144,141)",  # Viridis color 3 - teal
                "rgb(93,201,99)",  # Viridis color 4 - green
                "rgb(253,231,37)",  # Viridis color 5 - yellow
                "rgb(255,165,0)"  # Outlier - orange
            ]

            # Create the color blocks with values beneath
            for i, col in enumerate(cols):
                with col:
                    # Color block
                    st.markdown(
                        f"<div style='background-color:{colors[i]};height:20px;width:100%;'></div>",
                        unsafe_allow_html=True
                    )

                    # Value label - only show the threshold values, not min/max
                    if i == 0:
                        # Bottom outlier threshold only
                        st.caption(f"⚠️ {P01:.1f}")
                    elif i == 6:
                        # Top outlier threshold only
                        st.caption(f"⚠️ {P99:.1f}")
                    elif i < len(color_thresholds):
                        # Regular boundary value
                        st.caption(f"{color_thresholds[i]:.1f}")

            # Brief legend explanation
            st.caption("Colors show data distribution: dark red (<1%), Viridis palette (middle 98%), orange (>99%)")

            # Add the data summary section back
            st.write("### Data Summary")
            summary_cols = st.columns(3)

            with summary_cols[0]:
                st.metric("Minimum", f"{VMIN:.1f}")
                st.metric("1% cutoff", f"{P01:.1f}")
                st.metric("25th percentile", f"{Q1:.1f}")

            with summary_cols[1]:
                st.metric("Median", f"{MEDIAN:.1f}")
                st.metric("Mean", f"{MEAN:.1f}")
                st.metric("75th percentile", f"{Q3:.1f}")

            with summary_cols[2]:
                st.metric("99% cutoff", f"{P99:.1f}")
                st.metric("Maximum", f"{VMAX:.1f}")
                st.metric("Std. deviation", f"{float(GDF[MAP_VAR].std()):.1f}")

        except Exception as e:
            st.error(f"Error rendering map: {e}")
            st.info("Try selecting a different detail level from the sidebar.")

        # Build layers (highlight optional)
        layer_settings = {
            "get_fill_color": "properties._color",
            "get_line_color": [100, 100, 100, 120],  # Subtle grey border
            "line_width_min_pixels": 0.8,  # Slightly thicker border
            "pickable": True,
            "opacity": 0.75,  # Default opacity for all rendering modes
        }

        # Add extra performance optimizations if high performance mode is enabled
        if perf_mode:
            layer_settings.update({
                "material": False,  # Disable lighting effects
                "get_elevation": 0  # Ensure flat rendering
            })

        # Calculate opacity and width based on slider
        border_alpha = int(1.5 * border_emphasis)  # Convert 0-100 to 0-150 opacity
        border_width = 0.4 + (border_emphasis / 100) * 1.0  # Scale width from 0.4 to 1.4