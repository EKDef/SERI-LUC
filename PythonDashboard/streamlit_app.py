
# streamlit_app.py
import pathlib
import gzip
import json
import numpy as np
import geopandas as gpd
import streamlit as st
import streamlit.components.v1 as components
import folium

# --- Page config for wide layout ---
st.set_page_config(page_title='MSOA Dashboard', layout='wide')

# --- Cache loading of geojson and GeoDataFrame to avoid repeated IO ---
@st.cache_data
def load_geojson(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def build_gdf(geojson):
    crs = geojson.get('crs', {}).get('properties', {}).get('name', 'EPSG:4326')
    return gpd.GeoDataFrame.from_features(geojson['features'], crs=crs)

# --- Cache map HTML generation to avoid re-building Folium layers on pan/zoom ---
@st.cache_resource
def generate_map_html(geojson_data, df, var, selected_code, height, width):
    m = folium.Map(location=[54.0, -2.0], zoom_start=6)
    # choropleth
    folium.Choropleth(
        geo_data=geojson_data,
        data=df,
        columns=['MSOA21CD', var],
        key_on='feature.properties.MSOA21CD',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=var.replace('_', ' ').title()
    ).add_to(m)
    # highlight
    if selected_code:
        sel = gdf[gdf.MSOA21CD == selected_code]
        folium.GeoJson(
            sel,
            style_function=lambda feat: {'color': 'blue', 'weight': 3, 'fill': False}
        ).add_to(m)
        bounds = sel.total_bounds
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    # render HTML once
    return m.get_root().render()

# 1. Load data (once)
BASE_DIR = pathlib.Path(__file__).parent
GZ_FILE = BASE_DIR / 'msoa_2021.geojson.gz'
geojson_data = load_geojson(GZ_FILE)
gdf = build_gdf(geojson_data)

# 2. Demo numeric columns
gdf['employment_rate'] = np.random.default_rng(42).uniform(40, 90, len(gdf)).round(1)
gdf['years_schooling'] = np.random.default_rng(42).uniform(8, 18, len(gdf)).round(2)
gdf['median_income'] = np.random.default_rng(42).uniform(18_000, 50_000, len(gdf)).round(0)

# 3. UI controls
st.sidebar.title('Controls')
cols = ['employment_rate', 'years_schooling', 'median_income']
var = st.sidebar.selectbox('Variable', cols)
map_h = st.sidebar.slider('Map height', 400, 1000, 800)

df = gdf[['MSOA21CD', 'MSOA21NM'] + cols]
# highlight select
options = ['None'] + df.apply(lambda r: f"{r.MSOA21NM} ({r.MSOA21CD})", axis=1).tolist()
selected = st.sidebar.selectbox('Highlight MSOA', options)
selected_code = None if selected=='None' else selected.split('(')[-1].strip(')')

# 4. Generate & embed static HTML map (no rerun on pan/zoom)
html = generate_map_html(geojson_data, df, var, selected_code, map_h, None)
st.markdown('### Map')
components.html(html, height=map_h, width=0)

# 5. Interactive table below
st.markdown('### MSOA Data')
st.dataframe(df, use_container_width=True, height=st.sidebar.slider('Table height', 300, 1000, 400))
