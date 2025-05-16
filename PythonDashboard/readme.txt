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
#   • Special colouring for outliers to better visualize data distribution
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
# Launch from bash:
#   streamlit run streamlit_app_v2.py 
#
# Requirements:
#   pip install streamlit pandas geopandas pydeck shapely pyarrow geoparquet streamlit-aggrid>=1.1.3
# =============================================================================
