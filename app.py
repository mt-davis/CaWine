import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from io import BytesIO
import warnings

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="California Wine Industry Environmental Impact Analyzer",
    page_icon="🍷",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #722F37;}
    .sub-header {font-size: 1.8rem; color: #9B6A6C;}
    .region-header {font-size: 1.5rem; color: #2C3333; background-color: #F9F9F9; padding: 10px;}
    .highlight {background-color: #F9F5F0; padding: 20px; border-radius: 5px; border-left: 3px solid #722F37;}
    .metric-container {background-color: #FFFFFF; padding: 15px; border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.1);}
    .footer {text-align: center; color: #666666; font-size: 0.8rem; margin-top: 50px;}
</style>
""", unsafe_allow_html=True)

# ----- Helper Functions -----

@st.cache_data
def load_wine_regions():
    """Load California wine regions GeoJSON data"""
    # For a real app, you would use actual GeoJSON data of wine regions
    # This is a simplified example with fictional data
    regions = {
        "Napa Valley": {"lat": 38.5025, "lon": -122.2654, "acres": 45000, "primary_grapes": ["Cabernet Sauvignon", "Chardonnay"]},
        "Sonoma County": {"lat": 38.5111, "lon": -122.7884, "acres": 60000, "primary_grapes": ["Pinot Noir", "Chardonnay"]},
        "Paso Robles": {"lat": 35.6369, "lon": -120.6545, "acres": 40000, "primary_grapes": ["Cabernet Sauvignon", "Zinfandel"]},
        "Santa Barbara": {"lat": 34.4208, "lon": -119.6982, "acres": 21000, "primary_grapes": ["Pinot Noir", "Chardonnay", "Syrah"]},
        "Mendocino County": {"lat": 39.0307, "lon": -123.0877, "acres": 18000, "primary_grapes": ["Chardonnay", "Pinot Noir"]},
        "Lodi": {"lat": 38.1301, "lon": -121.2744, "acres": 110000, "primary_grapes": ["Zinfandel", "Cabernet Sauvignon"]}
    }
    return regions

@st.cache_data
def load_wildfire_data():
    """Load historical California wildfire data"""
    # In a real app, you would fetch this from a CAL FIRE API or dataset
    # This is sample data for demonstration
    wildfires = pd.DataFrame({
        "name": ["Glass Fire", "LNU Lightning Complex", "August Complex", "Creek Fire", "SCU Lightning Complex", 
                "CZU Lightning Complex", "North Complex", "Kincade Fire", "Walker Fire", "Bobcat Fire"],
        "year": [2020, 2020, 2020, 2020, 2020, 2020, 2020, 2019, 2019, 2020],
        "acres_burned": [67484, 363220, 1032648, 379895, 396624, 86509, 318935, 77758, 54612, 115998],
        "lat": [38.5035, 38.6501, 39.8170, 37.1908, 37.4230, 37.1714, 39.8168, 38.7883, 39.9761, 34.2380],
        "lon": [-122.4753, -122.1179, -122.8210, -119.2659, -121.4408, -122.2229, -121.1204, -122.7884, -120.6679, -118.0870],
        "start_date": ["2020-09-27", "2020-08-17", "2020-08-16", "2020-09-04", "2020-08-18", "2020-08-16", "2020-08-17", "2019-10-23", "2019-09-04", "2020-09-06"],
        "containment_date": ["2020-10-20", "2020-10-02", "2020-11-12", "2020-12-24", "2020-10-01", "2020-09-22", "2020-12-03", "2019-11-06", "2019-09-26", "2020-12-18"],
        "affected_wine_regions": [["Napa Valley", "Sonoma County"], ["Napa Valley", "Sonoma County"], ["Mendocino County"], [], [], [], [], ["Sonoma County"], ["Mendocino County"], []]
    })
    
    # Add more historic data
    historic_fires = pd.DataFrame({
        "name": ["Tubbs Fire", "Atlas Fire", "Nuns Fire", "Thomas Fire", "Mendocino Complex", "Camp Fire"],
        "year": [2017, 2017, 2017, 2017, 2018, 2018],
        "acres_burned": [36807, 51624, 54382, 281893, 459123, 153336],
        "lat": [38.4758, 38.4239, 38.4005, 34.3580, 39.0004, 39.8132],
        "lon": [-122.7247, -122.2600, -122.5082, -119.0823, -122.8016, -121.4364],
        "start_date": ["2017-10-08", "2017-10-08", "2017-10-08", "2017-12-04", "2018-07-27", "2018-11-08"],
        "containment_date": ["2017-10-31", "2017-10-28", "2017-10-30", "2018-01-12", "2018-09-18", "2018-11-25"],
        "affected_wine_regions": [["Sonoma County", "Napa Valley"], ["Napa Valley"], ["Sonoma County"], ["Santa Barbara"], ["Mendocino County"], []]
    })
    
    wildfires = pd.concat([wildfires, historic_fires], ignore_index=True)
    wildfires["start_date"] = pd.to_datetime(wildfires["start_date"])
    wildfires["containment_date"] = pd.to_datetime(wildfires["containment_date"])
    wildfires["duration_days"] = (wildfires["containment_date"] - wildfires["start_date"]).dt.days
    
    return wildfires

@st.cache_data
def load_wine_production_data():
    """Load California wine production data by region and year"""
    # In a real application, this would be actual production data
    # For this demo, creating synthetic data
    years = list(range(2015, 2024))
    regions = ["Napa Valley", "Sonoma County", "Paso Robles", "Santa Barbara", "Mendocino County", "Lodi"]
    
    # Base production levels for each region (in thousands of tons)
    base_production = {
        "Napa Valley": 170,
        "Sonoma County": 220,
        "Paso Robles": 140,
        "Santa Barbara": 85,
        "Mendocino County": 70,
        "Lodi": 550
    }
    
    # Production quality score baseline (0-100)
    base_quality = {
        "Napa Valley": 92,
        "Sonoma County": 90,
        "Paso Robles": 89,
        "Santa Barbara": 91,
        "Mendocino County": 88,
        "Lodi": 86
    }
    
    # Impact events - key: (year, region), value: (production_impact_percent, quality_impact_points)
    # Negative values represent negative impacts
    impact_events = {
        (2017, "Napa Valley"): (-15, -5),  # Tubbs and Atlas fires
        (2017, "Sonoma County"): (-20, -8),  # Tubbs and Nuns fires
        (2018, "Mendocino County"): (-8, -3),  # Mendocino Complex
        (2020, "Napa Valley"): (-25, -12),  # Glass Fire and LNU Complex
        (2020, "Sonoma County"): (-18, -7),  # Multiple fires
        (2020, "Mendocino County"): (-10, -4),  # August Complex
        (2021, "Napa Valley"): (-5, -2),  # Lingering smoke effects
        (2021, "Sonoma County"): (-3, -1),  # Lingering smoke effects
        (2022, "Napa Valley"): (2, 1),     # Recovery year
        (2022, "Sonoma County"): (3, 1)    # Recovery year
    }
    
    # Drought impacts by year (statewide)
    # (production_impact_percent, quality_impact_points)
    drought_impacts = {
        2015: (-5, -1),
        2016: (-8, -2),
        2020: (-10, -3),
        2021: (-15, -4),
        2022: (-12, -3)
    }
    
    # Generate production data
    data = []
    
    for year in years:
        for region in regions:
            # Base production with random variation
            production = base_production[region] * (1 + np.random.normal(0, 0.05))
            quality = base_quality[region] + np.random.normal(0, 1)
            
            # Apply fire impacts
            if (year, region) in impact_events:
                prod_impact, qual_impact = impact_events[(year, region)]
                production = production * (1 + prod_impact/100)
                quality = quality + qual_impact
            
            # Apply drought impacts
            if year in drought_impacts:
                prod_impact, qual_impact = drought_impacts[year]
                production = production * (1 + prod_impact/100)
                quality = quality + qual_impact
            
            # Apply random yearly variations
            production = max(0, production * (1 + np.random.normal(0, 0.03)))
            quality = min(100, max(70, quality + np.random.normal(0, 0.5)))
            
            data.append({
                "year": year,
                "region": region,
                "production_tons": round(production, 1),
                "quality_score": round(quality, 1),
                "revenue_millions": round(production * quality/10 * np.random.uniform(0.9, 1.1), 1)
            })
    
    return pd.DataFrame(data)

@st.cache_data
def load_weather_data():
    """Load historical weather data for wine regions"""
    # In a real app, this would come from NOAA or similar weather APIs
    regions = ["Napa Valley", "Sonoma County", "Paso Robles", "Santa Barbara", "Mendocino County", "Lodi"]
    years = list(range(2015, 2024))
    
    # Base weather profiles for regions
    base_temp = {
        "Napa Valley": 60,      # Average annual temp (°F)
        "Sonoma County": 59,
        "Paso Robles": 65,
        "Santa Barbara": 64,
        "Mendocino County": 57,
        "Lodi": 63
    }
    
    base_rainfall = {
        "Napa Valley": 25,      # Average annual rainfall (inches)
        "Sonoma County": 30,
        "Paso Robles": 15,
        "Santa Barbara": 18,
        "Mendocino County": 40,
        "Lodi": 20
    }
    
    # Yearly variations from normal (statewide)
    yearly_variations = {
        # (temp_offset, rainfall_percent)
        2015: (1.5, -30),       # Drought year
        2016: (0.8, -20),       # Drought year
        2017: (-0.5, 40),       # Wet year
        2018: (1.2, -15),
        2019: (-0.2, 10),
        2020: (2.5, -35),       # Hot, dry year
        2021: (1.8, -40),       # Severe drought
        2022: (1.0, -25),       # Continuing drought
        2023: (-1.0, 50)        # Wet year
    }
    
    # Generate weather data
    data = []
    
    for year in years:
        temp_offset, rainfall_pct = yearly_variations.get(year, (0, 0))
        
        for region in regions:
            avg_temp = base_temp[region] + temp_offset + np.random.normal(0, 0.5)
            rainfall = base_rainfall[region] * (1 + rainfall_pct/100) * np.random.uniform(0.9, 1.1)
            frost_days = max(0, int(np.random.normal(10, 5) - avg_temp/10))
            heat_days = max(0, int(np.random.normal(-30, 20) + avg_temp))
            
            # AQI (Air Quality Index) data - higher during fire years
            base_aqi = 30
            if year == 2017 and region in ["Napa Valley", "Sonoma County"]:
                aqi = base_aqi * np.random.uniform(5, 7)
            elif year == 2018 and region == "Mendocino County":
                aqi = base_aqi * np.random.uniform(4, 6)
            elif year == 2020 and region in ["Napa Valley", "Sonoma County", "Mendocino County"]:
                aqi = base_aqi * np.random.uniform(6, 8)
            else:
                aqi = base_aqi * np.random.uniform(0.8, 1.2)
            
            data.append({
                "year": year,
                "region": region,
                "avg_temperature": round(avg_temp, 1),
                "rainfall_inches": round(rainfall, 1),
                "frost_days": frost_days,
                "days_over_95F": heat_days,
                "avg_aqi": round(aqi, 1),
                "drought_index": round(5 - rainfall/base_rainfall[region] * 5, 1)  # 0-5 scale, 5 is worst
            })
    
    return pd.DataFrame(data)

def calculate_smoke_exposure(wildfires_df, wine_regions):
    """Calculate smoke exposure for each region and year"""
    smoke_exposure = []
    regions = list(wine_regions.keys())
    years = list(range(2015, 2024))
    
    for year in years:
        year_fires = wildfires_df[wildfires_df['year'] == year]
        
        for region in regions:
            region_lat = wine_regions[region]['lat']
            region_lon = wine_regions[region]['lon']
            
            # Calculate exposure based on proximity and fire size
            exposure = 0
            for _, fire in year_fires.iterrows():
                # Distance in degrees (simplified)
                distance = ((fire['lat'] - region_lat) ** 2 + (fire['lon'] - region_lon) ** 2) ** 0.5
                
                # If region is in affected list or very close, higher exposure
                if region in fire['affected_wine_regions']:
                    exposure += fire['acres_burned'] / 10000  # Direct impact
                elif distance < 0.5:  # Within ~35 miles
                    exposure += fire['acres_burned'] / 20000 * (1 - distance)
            
            smoke_exposure.append({
                'year': year,
                'region': region,
                'smoke_exposure_level': round(min(10, exposure), 2)  # Renamed column to avoid conflicts
            })
    
    return pd.DataFrame(smoke_exposure)

def generate_composite_impact_score(wine_df, weather_df, smoke_df):
    """Generate a composite environmental impact score"""
    # Merge datasets with explicit suffixes to avoid column name conflicts
    merged = wine_df.merge(weather_df, on=['year', 'region'])
    merged = merged.merge(smoke_df, on=['year', 'region'])
    
    # Calculate impact score components
    merged['heat_impact'] = merged['days_over_95F'] / 30 * 2  # 0-2 scale
    merged['drought_impact'] = merged['drought_index'] / 5 * 3  # 0-3 scale
    merged['smoke_impact'] = merged['smoke_exposure_level'] / 10 * 5  # 0-5 scale
    
    # Composite score (0-10 scale, higher is worse)
    merged['environmental_impact_score'] = merged[['heat_impact', 'drought_impact', 'smoke_impact']].sum(axis=1)
    merged['environmental_impact_score'] = merged['environmental_impact_score'].apply(lambda x: min(10, x))
    
    # Normalized production (percentage of average for that region)
    region_avg = merged.groupby('region')['production_tons'].transform('mean')
    merged['production_percent'] = merged['production_tons'] / region_avg * 100
    
    return merged

def train_prediction_model(impact_df):
    """Train a model to predict wine quality and yield based on environmental factors"""
    features = ['avg_temperature', 'rainfall_inches', 'frost_days', 'days_over_95F', 
                'avg_aqi', 'drought_index', 'smoke_exposure_level']
    
    # Model for quality
    X = impact_df[features]
    y_quality = impact_df['quality_score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_quality, test_size=0.2, random_state=42)
    
    quality_model = RandomForestRegressor(n_estimators=100, random_state=42)
    quality_model.fit(X_train, y_train)
    
    quality_r2 = r2_score(y_test, quality_model.predict(X_test))
    
    # Model for production
    y_production = impact_df['production_percent']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_production, test_size=0.2, random_state=42)
    
    production_model = RandomForestRegressor(n_estimators=100, random_state=42)
    production_model.fit(X_train, y_train)
    
    production_r2 = r2_score(y_test, production_model.predict(X_test))
    
    return quality_model, production_model, quality_r2, production_r2, features

# ------ Main Application ------

st.markdown('<h1 class="main-header">California Wine Industry Environmental Impact Analyzer</h1>', unsafe_allow_html=True)

st.markdown("""
This application analyzes the impact of wildfires, drought, and other environmental factors on 
California's wine industry. Explore interactive visualizations, predictive models, and insights 
to understand how climate events affect wine production, quality, and revenue across key growing regions.
""")

# Load data
wine_regions = load_wine_regions()
wildfires_df = load_wildfire_data()
wine_production_df = load_wine_production_data()
weather_df = load_weather_data()

# Calculate derived data
smoke_exposure_df = calculate_smoke_exposure(wildfires_df, wine_regions)
impact_df = generate_composite_impact_score(wine_production_df, weather_df, smoke_exposure_df)

# Train models
quality_model, production_model, quality_r2, production_r2, model_features = train_prediction_model(impact_df)

# Create tabs for the app
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dashboard", "Fire Impact Analysis", "Weather Patterns", 
    "Regional Comparison", "Prediction Models"
])

# ------ Tab 1: Dashboard ------
with tab1:
    st.markdown('<h2 class="sub-header">Industry Overview Dashboard</h2>', unsafe_allow_html=True)
    
    # Year filter
    selected_year = st.slider("Select Year", min_value=2015, max_value=2023, value=2023)
    year_data = impact_df[impact_df['year'] == selected_year]
    
    # Top row metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_production = year_data['production_tons'].sum()
        prev_year_production = impact_df[impact_df['year'] == selected_year-1]['production_tons'].sum() if selected_year > 2015 else 0
        pct_change = ((total_production - prev_year_production) / prev_year_production * 100) if prev_year_production > 0 else 0
        
        st.metric("Total Wine Production", f"{total_production:,.0f} tons", f"{pct_change:.1f}%" if pct_change != 0 else None)
    
    with col2:
        avg_quality = year_data['quality_score'].mean()
        prev_year_quality = impact_df[impact_df['year'] == selected_year-1]['quality_score'].mean() if selected_year > 2015 else 0
        quality_change = avg_quality - prev_year_quality if prev_year_quality > 0 else 0
        
        st.metric("Average Quality Score", f"{avg_quality:.1f}/100", f"{quality_change:+.1f}" if quality_change != 0 else None)
    
    with col3:
        total_revenue = year_data['revenue_millions'].sum()
        prev_year_revenue = impact_df[impact_df['year'] == selected_year-1]['revenue_millions'].sum() if selected_year > 2015 else 0
        revenue_pct_change = ((total_revenue - prev_year_revenue) / prev_year_revenue * 100) if prev_year_revenue > 0 else 0
        
        st.metric("Total Revenue", f"${total_revenue:,.0f}M", f"{revenue_pct_change:.1f}%" if revenue_pct_change != 0 else None)
    
    with col4:
        impact_score = year_data['environmental_impact_score'].mean()
        prev_year_impact = impact_df[impact_df['year'] == selected_year-1]['environmental_impact_score'].mean() if selected_year > 2015 else 0
        impact_change = impact_score - prev_year_impact if prev_year_impact > 0 else 0
        
        st.metric("Environmental Impact", f"{impact_score:.1f}/10", f"{impact_change:+.1f}" if impact_change != 0 else None)
    
    # Map and charts row
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### Wine Regions and Wildfire Map")
        
        # Create a Folium map
        m = folium.Map(location=[37.7749, -122.4194], zoom_start=6, tiles="CartoDB positron")
        
        # Add wine regions
        for region, data in wine_regions.items():
            region_data = year_data[year_data['region'] == region].iloc[0]
            production_pct = region_data['production_percent']
            quality = region_data['quality_score']
            
            # Color based on production percent
            if production_pct >= 100:
                color = 'green'
            elif production_pct >= 90:
                color = 'lightgreen'
            elif production_pct >= 80:
                color = 'orange'
            else:
                color = 'red'
            
            popup_text = f"""
            <b>{region}</b><br>
            Production: {region_data['production_tons']:,.0f} tons ({production_pct:.1f}%)<br>
            Quality Score: {quality:.1f}/100<br>
            Revenue: ${region_data['revenue_millions']:,.1f}M<br>
            Acres: {data['acres']:,}<br>
            Primary Grapes: {', '.join(data['primary_grapes'])}
            """
            
            folium.Circle(
                location=[data['lat'], data['lon']],
                radius=data['acres'] * 5,  # Size based on vineyard acres
                color=color,
                fill=True,
                fill_opacity=0.6,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)
        
        # Add fires from selected year
        year_fires = wildfires_df[wildfires_df['year'] == selected_year]
        
        for _, fire in year_fires.iterrows():
            # Size based on acres burned
            radius = (fire['acres_burned']) ** 0.5 * 100
            
            popup_text = f"""
            <b>{fire['name']}</b><br>
            Acres Burned: {fire['acres_burned']:,}<br>
            Started: {fire['start_date'].strftime('%b %d, %Y')}<br>
            Duration: {fire['duration_days']} days<br>
            """
            
            folium.Circle(
                location=[fire['lat'], fire['lon']],
                radius=radius,
                color='red',
                fill=True,
                fill_opacity=0.3,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)
        
        # Display the map
        folium_static(m)
    
    with col2:
        st.markdown("### Production by Region")
        region_production = year_data.sort_values('production_tons', ascending=False)
        
        fig = px.bar(
            region_production,
            x='region',
            y='production_tons',
            color='quality_score',
            color_continuous_scale='RdYlGn',
            labels={
                'region': 'Wine Region',
                'production_tons': 'Production (tons)',
                'quality_score': 'Quality Score'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Environmental factors impact
        st.markdown("### Environmental Factors")
        
        factor_data = year_data.melt(
            id_vars=['region'], 
            value_vars=['drought_index', 'smoke_exposure_level', 'days_over_95F'],
            var_name='factor', 
            value_name='value'
        )
        
        # Rename factors for display
        factor_data['factor'] = factor_data['factor'].replace({
            'drought_index': 'Drought Severity',
            'smoke_exposure_level': 'Smoke Exposure',
            'days_over_95F': 'Heat Days'
        })
        
        fig = px.bar(
            factor_data,
            x='region',
            y='value',
            color='factor',
            barmode='group',
            labels={
                'region': 'Wine Region',
                'value': 'Impact Score',
                'factor': 'Environmental Factor'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Historical trends
    st.markdown("### Historical Trends (2015-2023)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Production trend
        yearly_production = impact_df.groupby('year')['production_tons'].sum().reset_index()
        
        fig = px.line(
            yearly_production,
            x='year',
            y='production_tons',
            labels={
                'year': 'Year',
                'production_tons': 'Total Production (tons)'
            },
            title="California Wine Production Trend"
        )
        
        # Add major fire events as annotations
        major_fires = {
            2017: "Tubbs & Atlas Fires",
            2018: "Mendocino Complex",
            2020: "Glass Fire & LNU Complex"
        }
        
        for year, fire_name in major_fires.items():
            if year in yearly_production['year'].values:
                y_value = yearly_production[yearly_production['year'] == year]['production_tons'].iloc[0]
                fig.add_annotation(
                    x=year,
                    y=y_value,
                    text=fire_name,
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40
                )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Quality trend
        yearly_quality = impact_df.groupby('year')['quality_score'].mean().reset_index()
        
        fig = px.line(
            yearly_quality,
            x='year',
            y='quality_score',
            labels={
                'year': 'Year',
                'quality_score': 'Average Quality Score'
            },
            title="California Wine Quality Trend"
        )
        
        # Add major fire events as annotations
        for year, fire_name in major_fires.items():
            if year in yearly_quality['year'].values:
                y_value = yearly_quality[yearly_quality['year'] == year]['quality_score'].iloc[0]
                fig.add_annotation(
                    x=year,
                    y=y_value,
                    text=fire_name,
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40
                )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Impact correlation
    st.markdown("### Correlation Between Environmental Impact and Wine Production/Quality")
    
    correlation_df = impact_df.copy()
    
    fig = px.scatter(
        correlation_df,
        x='environmental_impact_score',
        y='quality_score',
        size='production_tons',
        color='region',
        facet_col='year',
        facet_col_wrap=3,
        labels={
            'environmental_impact_score': 'Environmental Impact Score',
            'quality_score': 'Wine Quality Score',
            'production_tons': 'Production Volume',
            'region': 'Wine Region'
        },
        title="Environmental Impact vs. Wine Quality",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ------ Tab 2: Fire Impact Analysis ------
with tab2:
    st.markdown('<h2 class="sub-header">Wildfire Impact Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Major California Wildfires")
        
        # Fire year filter
        fire_years = sorted(wildfires_df['year'].unique(), reverse=True)
        selected_fire_year = st.selectbox("Select Year", fire_years)
        
        # Display fire data for selected year
        year_fires = wildfires_df[wildfires_df['year'] == selected_fire_year].sort_values('acres_burned', ascending=False)
        
        for _, fire in year_fires.iterrows():
            with st.expander(f"{fire['name']} ({fire['acres_burned']:,} acres)"):
                st.write(f"**Started:** {fire['start_date'].strftime('%B %d, %Y')}")
                st.write(f"**Contained:** {fire['containment_date'].strftime('%B %d, %Y')}")
                st.write(f"**Duration:** {fire['duration_days']} days")
                
                if fire['affected_wine_regions']:
                    st.write(f"**Affected Wine Regions:** {', '.join(fire['affected_wine_regions'])}")
                else:
                    st.write("**Affected Wine Regions:** None directly affected")
        
        # Fire statistics
        st.markdown("### Fire Statistics")
        st.metric("Total Fires", f"{len(year_fires)}")
        st.metric("Total Acres Burned", f"{year_fires['acres_burned'].sum():,}")
        st.metric("Average Fire Size", f"{year_fires['acres_burned'].mean():,.0f} acres")
        st.metric("Largest Fire", f"{year_fires['acres_burned'].max():,} acres ({year_fires.iloc[0]['name']})")
    
    with col2:
        st.markdown("### Smoke Exposure Analysis")
        
        # Filter smoke data for the selected year
        year_smoke = smoke_exposure_df[smoke_exposure_df['year'] == selected_fire_year]
        
        fig = px.bar(
            year_smoke.sort_values('smoke_exposure_level', ascending=False),
            x='region',
            y='smoke_exposure_level',
            color='smoke_exposure_level',
            color_continuous_scale='Reds',
            labels={
                'region': 'Wine Region',
                'smoke_exposure_level': 'Smoke Exposure Index (0-10)'
            },
            title=f"Smoke Exposure by Wine Region ({selected_fire_year})"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Impact on wine metrics by region
        impact_year = impact_df[impact_df['year'] == selected_fire_year]
        prev_year = impact_df[impact_df['year'] == selected_fire_year - 1] if selected_fire_year > 2015 else None
        
        # Merge with smoke data - using consistent naming
        impact_year = impact_year.merge(year_smoke, on=['year', 'region'], suffixes=('', '_smoke'))
        
        if prev_year is not None:
            # Calculate year-over-year changes
            impact_compare = impact_year.copy()
            for region in impact_compare['region'].unique():
                prev_region = prev_year[prev_year['region'] == region]
                if not prev_region.empty:
                    impact_compare.loc[impact_compare['region'] == region, 'production_change_pct'] = (
                        (impact_compare[impact_compare['region'] == region]['production_tons'].values[0] / 
                         prev_region['production_tons'].values[0] - 1) * 100
                    )
                    
                    impact_compare.loc[impact_compare['region'] == region, 'quality_change'] = (
                        impact_compare[impact_compare['region'] == region]['quality_score'].values[0] - 
                        prev_region['quality_score'].values[0]
                    )
            
            # Create a positive size value for the scatter plot
            impact_compare['quality_change_abs'] = impact_compare['quality_change'].abs() + 5  # Add offset to make all values positive and visible
            
            # Updated to use absolute values for size
            fig = px.scatter(
                impact_compare,
                x='smoke_exposure_level',
                y='production_change_pct',
                size='quality_change_abs',  # Use the corrected column
                color='region',
                labels={
                    'smoke_exposure_level': 'Smoke Exposure Index',
                    'production_change_pct': 'Production Change from Previous Year (%)',
                    'quality_change': 'Quality Score Change',
                    'region': 'Wine Region'
                },
                title=f"Smoke Exposure vs. Production & Quality Changes ({selected_fire_year})"
            )
            
            # Add hover information with original quality change values
            fig.update_traces(
                hovertemplate='<b>%{customdata}</b><br>Smoke Exposure: %{x:.1f}<br>Production Change: %{y:.1f}%<br>Quality Change: %{text:+.1f}<br>',
                text=impact_compare['quality_change'],  # Original quality change for hover
                customdata=impact_compare['region']
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Historical smoke exposure across all years
        st.markdown("### Historical Smoke Exposure")
        
        # Pivot the data for heatmap
        smoke_pivot = smoke_exposure_df.pivot(index='year', columns='region', values='smoke_exposure_level')
        
        fig = px.imshow(
            smoke_pivot,
            labels=dict(x="Wine Region", y="Year", color="Smoke Exposure"),
            x=smoke_pivot.columns,
            y=smoke_pivot.index,
            color_continuous_scale='Reds',
            title="Smoke Exposure by Region and Year (0-10 scale)",
            aspect="auto"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# ------ Tab 3: Weather Patterns ------
with tab3:
    st.markdown('<h2 class="sub-header">Weather Pattern Analysis</h2>', unsafe_allow_html=True)
    
    # Region selector for detailed weather analysis
    weather_region = st.selectbox(
        "Select Wine Region for Weather Analysis",
        options=list(wine_regions.keys())
    )
    
    region_weather = weather_df[weather_df['region'] == weather_region]
    
    # Weather data charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Temperature trend
        fig = px.line(
            region_weather,
            x='year',
            y='avg_temperature',
            labels={
                'year': 'Year',
                'avg_temperature': 'Average Temperature (°F)'
            },
            title=f"Temperature Trend: {weather_region}"
        )
        
        # Add a trendline
        fig.add_traces(
            px.scatter(
                region_weather,
                x='year',
                y='avg_temperature',
                trendline='ols'
            ).data[1]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Heat days
        fig = px.bar(
            region_weather,
            x='year',
            y='days_over_95F',
            labels={
                'year': 'Year',
                'days_over_95F': 'Number of Days over 95°F'
            },
            title=f"Heat Days Trend: {weather_region}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Rainfall trend
        fig = px.line(
            region_weather,
            x='year',
            y='rainfall_inches',
            labels={
                'year': 'Year',
                'rainfall_inches': 'Annual Rainfall (inches)'
            },
            title=f"Rainfall Trend: {weather_region}"
        )
        
        # Ensure region_weather is not empty
        if not region_weather.empty:
            # Calculate the average rainfall
            avg_rainfall = region_weather['rainfall_inches'].mean()

            # Add a horizontal line for the average rainfall
            fig.add_hline(
                y=avg_rainfall,
                line_dash="dash",
                line_color="rgba(0,0,0,0.5)",
                annotation_text="Average Rainfall",
                annotation_position="top left"
            )
        else:
            print("region_weather is empty")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Drought index
        fig = px.bar(
            region_weather,
            x='year',
            y='drought_index',
            color='drought_index',
            color_continuous_scale='YlOrBr',
            labels={
                'year': 'Year',
                'drought_index': 'Drought Severity Index (0-5)'
            },
            title=f"Drought Severity Trend: {weather_region}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Weather impacts on wine production and quality
    st.markdown("### Weather Impacts on Wine Production and Quality")
    
    region_impact = impact_df[impact_df['region'] == weather_region]
    
    # Create metrics to show correlations
    corr_temp_quality = region_impact['avg_temperature'].corr(region_impact['quality_score'])
    corr_rainfall_quality = region_impact['rainfall_inches'].corr(region_impact['quality_score'])
    corr_drought_quality = region_impact['drought_index'].corr(region_impact['quality_score'])
    
    corr_temp_production = region_impact['avg_temperature'].corr(region_impact['production_tons'])
    corr_rainfall_production = region_impact['rainfall_inches'].corr(region_impact['production_tons'])
    corr_drought_production = region_impact['drought_index'].corr(region_impact['production_tons'])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Temperature Impact", f"{corr_temp_quality:.2f}", "on Quality")
        st.metric("", f"{corr_temp_production:.2f}", "on Production")
    
    with col2:
        st.metric("Rainfall Impact", f"{corr_rainfall_quality:.2f}", "on Quality")
        st.metric("", f"{corr_rainfall_production:.2f}", "on Production")
    
    with col3:
        st.metric("Drought Impact", f"{corr_drought_quality:.2f}", "on Quality")
        st.metric("", f"{corr_drought_production:.2f}", "on Production")
    
    # Multivariate analysis with 4 metrics
    st.markdown("### Multivariate Analysis of Weather Effects")
    
    # Scatter plot of rain vs temp, sized by production and colored by quality
    fig = px.scatter(
        region_impact,
        x='rainfall_inches',
        y='avg_temperature',
        size='production_tons',
        color='quality_score',
        hover_name='year',
        labels={
            'rainfall_inches': 'Annual Rainfall (inches)',
            'avg_temperature': 'Average Temperature (°F)',
            'production_tons': 'Production (tons)',
            'quality_score': 'Quality Score'
        },
        title=f"Weather Effects on Wine Production and Quality: {weather_region}"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show correlation matrix for all weather and wine metrics
    st.markdown("### Correlation Matrix")
    
    corr_columns = ['avg_temperature', 'rainfall_inches', 'drought_index', 
                    'days_over_95F', 'frost_days', 'avg_aqi', 'smoke_exposure_level',
                    'quality_score', 'production_tons', 'revenue_millions']
    
    corr_matrix = region_impact[corr_columns].corr().round(2)
    
    fig = px.imshow(
        corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title=f"Correlation Matrix: {weather_region}"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

# ------ Tab 4: Regional Comparison ------
with tab4:
    st.markdown('<h2 class="sub-header">Regional Comparison Analysis</h2>', unsafe_allow_html=True)
    
    # Allow multi-select for regions
    selected_regions = st.multiselect(
        "Select Regions to Compare",
        options=list(wine_regions.keys()),
        default=list(wine_regions.keys())[:3]
    )
    
    if not selected_regions:
        st.warning("Please select at least one region to display data.")
    else:
        # Filter data for selected regions
        filtered_regions = impact_df[impact_df['region'].isin(selected_regions)]
        
        # Standardize production data by region for fair comparison
        region_means = filtered_regions.groupby('region')['production_tons'].mean().reset_index()
        filtered_regions = filtered_regions.merge(region_means, on='region', suffixes=('', '_mean'))
        filtered_regions['production_normalized'] = filtered_regions['production_tons'] / filtered_regions['production_tons_mean'] * 100
        
        # Regional comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Production comparison
            fig = px.line(
                filtered_regions,
                x='year',
                y='production_normalized',
                color='region',
                labels={
                    'year': 'Year',
                    'production_normalized': 'Production (% of Regional Average)',
                    'region': 'Wine Region'
                },
                title="Production Comparison (Normalized)"
            )
            
            # Add a reference line at 100%
            fig.add_shape(
                type="line",
                x0=2015,
                y0=100,
                x1=2023,
                y1=100,
                line=dict(
                    color="black",
                    width=1,
                    dash="dash",
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Quality comparison
            fig = px.line(
                filtered_regions,
                x='year',
                y='quality_score',
                color='region',
                labels={
                    'year': 'Year',
                    'quality_score': 'Quality Score',
                    'region': 'Wine Region'
                },
                title="Quality Score Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Environmental factors comparison
        st.markdown("### Environmental Impacts by Region")
        
        # Create a radar chart for comparing environmental factors
        selected_year = st.slider("Select Year for Comparison", min_value=2015, max_value=2023, value=2020)
        
        year_region_data = filtered_regions[filtered_regions['year'] == selected_year]
        
        # Radar chart features
        radar_features = ['avg_temperature', 'rainfall_inches', 'drought_index', 
                         'days_over_95F', 'smoke_exposure_level', 'avg_aqi']
        
        # Feature names for display
        feature_names = {
            'avg_temperature': 'Temperature (°F)',
            'rainfall_inches': 'Rainfall (in)',
            'drought_index': 'Drought (0-5)',
            'days_over_95F': 'Heat Days',
            'smoke_exposure_level': 'Smoke (0-10)',
            'avg_aqi': 'Air Quality Index'
        }
        
        # Create the radar chart data
        radar_data = []
        
        for region in selected_regions:
            region_data = year_region_data[year_region_data['region'] == region]
            if not region_data.empty:
                values = region_data[radar_features].values[0].tolist()
                radar_data.append({
                    'region': region,
                    'values': values,
                    'feature_names': [feature_names[f] for f in radar_features]
                })
        
        # Create a radar chart for each region
        col1, col2 = st.columns(2)
        
        for i, data in enumerate(radar_data):
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=data['values'],
                theta=data['feature_names'],
                fill='toself',
                name=data['region']
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                    ),
                ),
                showlegend=True,
                title=f"Environmental Factors: {data['region']} ({selected_year})"
            )
            
            if i % 2 == 0:
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                with col2:
                    st.plotly_chart(fig, use_container_width=True)
        
        # Impact summary table
        st.markdown("### Regional Impact Summary")
        
        # Calculate average metrics for each region
        summary_data = []
        
        for region in selected_regions:
            region_data = filtered_regions[filtered_regions['region'] == region]
            
            # Calculate fire years metrics (2017, 2018, 2020)
            fire_years_data = region_data[region_data['year'].isin([2017, 2018, 2020])]
            normal_years_data = region_data[~region_data['year'].isin([2017, 2018, 2020])]
            
            if not fire_years_data.empty and not normal_years_data.empty:
                fire_quality = fire_years_data['quality_score'].mean()
                normal_quality = normal_years_data['quality_score'].mean()
                quality_impact = fire_quality - normal_quality
                
                fire_production = fire_years_data['production_normalized'].mean()
                normal_production = normal_years_data['production_normalized'].mean()
                production_impact = fire_production - normal_production
                
                max_smoke = region_data['smoke_exposure_level'].max()
                max_smoke_year = region_data[region_data['smoke_exposure_level'] == max_smoke]['year'].iloc[0]
                
                most_impacted_year = region_data[region_data['environmental_impact_score'] == 
                                              region_data['environmental_impact_score'].max()]['year'].iloc[0]
                
                summary_data.append({
                    'Region': region,
                    'Quality Impact': quality_impact,
                    'Production Impact': production_impact,
                    'Max Smoke Exposure': max_smoke,
                    'Max Smoke Year': max_smoke_year,
                    'Most Impacted Year': most_impacted_year
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Style the DataFrame
        def highlight_negative(val):
            color = 'red' if val < 0 else 'black'
            return f'color: {color}'
        
        st.dataframe(
            summary_df.style.format({
                'Quality Impact': '{:.1f}',
                'Production Impact': '{:.1f}%',
                'Max Smoke Exposure': '{:.1f}'
            }).applymap(highlight_negative, subset=['Quality Impact', 'Production Impact'])
        )
        
        # Recovery analysis
        st.markdown("### Recovery Analysis")
        
        # For each region, analyze how long it took to recover after major fire years
        col1, col2 = st.columns([1, 2])
        
        with col1:
            recovery_region = st.selectbox(
                "Select Region for Recovery Analysis",
                options=selected_regions
            )
            
            fire_year = st.selectbox(
                "Select Fire Year",
                options=[2017, 2018, 2020]
            )
        
        with col2:
            region_data = filtered_regions[filtered_regions['region'] == recovery_region]
            
            # Check if we have data for fire year and subsequent years
            if fire_year in region_data['year'].values and max(region_data['year']) > fire_year:
                # Get data for the selected fire year and after
                pre_fire = region_data[region_data['year'] == fire_year - 1]['production_normalized'].iloc[0]
                fire_and_after = region_data[region_data['year'] >= fire_year].copy()
                
                # Calculate percentage of pre-fire production
                fire_and_after['pct_of_pre_fire'] = fire_and_after['production_normalized'] / pre_fire * 100
                
                # Quality data
                pre_fire_quality = region_data[region_data['year'] == fire_year - 1]['quality_score'].iloc[0]
                fire_and_after['quality_vs_pre_fire'] = fire_and_after['quality_score'] - pre_fire_quality
                
                fig = go.Figure()
                
                # Add production recovery line
                fig.add_trace(go.Scatter(
                    x=fire_and_after['year'],
                    y=fire_and_after['pct_of_pre_fire'],
                    mode='lines+markers',
                    name='Production Recovery',
                    line=dict(color='blue', width=2)
                ))
                
                # Add quality recovery line (secondary axis)
                fig.add_trace(go.Scatter(
                    x=fire_and_after['year'],
                    y=fire_and_after['quality_vs_pre_fire'],
                    mode='lines+markers',
                    name='Quality Change',
                    line=dict(color='red', width=2),
                    yaxis='y2'
                ))
                
                # Add reference line at 100% recovery
                fig.add_shape(
                    type="line",
                    x0=min(fire_and_after['year']),
                    y0=100,
                    x1=max(fire_and_after['year']),
                    y1=100,
                    line=dict(
                        color="blue",
                        width=1,
                        dash="dash",
                    ),
                    yref="y"  # Use yref for primary y-axis
                )
                
                # Add reference line at 0 quality change
                fig.add_shape(
                    type="line",
                    x0=min(fire_and_after['year']),
                    y0=0,
                    x1=max(fire_and_after['year']),
                    y1=0,
                    line=dict(
                        color="red",
                        width=1,
                        dash="dash",
                    ),
                    yref="y2"  # Use yref instead of yaxis
                )
                
                # Update layout with double y-axis
                fig.update_layout(
                    title=f"Recovery After {fire_year} Fire: {recovery_region}",
                    xaxis=dict(title='Year'),
                    yaxis=dict(
                        title='Production (% of Pre-Fire Year)',
                        side='left',
                        range=[0, max(150, max(fire_and_after['pct_of_pre_fire']) * 1.1)]
                    ),
                    yaxis2=dict(
                        title='Quality Score Change',
                        side='right',
                        overlaying='y',
                        range=[min(-15, min(fire_and_after['quality_vs_pre_fire']) * 1.1), 
                               max(15, max(fire_and_after['quality_vs_pre_fire']) * 1.1)]
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Recovery metrics
                recovery_year = None
                for _, row in fire_and_after.iterrows():
                    if row['pct_of_pre_fire'] >= 100 and row['year'] > fire_year:
                        recovery_year = row['year']
                        break
                
                if recovery_year:
                    recovery_time = recovery_year - fire_year
                    st.success(f"Production recovered to pre-fire levels in {recovery_time} years (by {recovery_year})")
                else:
                    latest_recovery = fire_and_after.iloc[-1]['pct_of_pre_fire']
                    st.warning(f"Production has not yet fully recovered. Currently at {latest_recovery:.1f}% of pre-fire levels")
                
                quality_recovered = any(fire_and_after[fire_and_after['year'] > fire_year]['quality_vs_pre_fire'] >= 0)
                if quality_recovered:
                    quality_recovery_year = fire_and_after[fire_and_after['quality_vs_pre_fire'] >= 0].iloc[0]['year']
                    st.success(f"Quality recovered to pre-fire levels by {quality_recovery_year} ({quality_recovery_year - fire_year} years)")
                else:
                    latest_quality = fire_and_after.iloc[-1]['quality_vs_pre_fire']
                    st.warning(f"Quality has not yet fully recovered. Currently {abs(latest_quality):.1f} points below pre-fire levels")
            
            else:
                st.warning(f"Insufficient data for recovery analysis for {recovery_region} after {fire_year}")

# ------ Tab 5: Prediction Models ------
with tab5:
    st.markdown('<h2 class="sub-header">Predictive Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    This section uses machine learning models to predict wine quality and production based on environmental factors. 
    The models have been trained on historical data from 2015-2023, capturing the relationships between 
    environmental conditions and wine outcomes.
    </div>
    """, unsafe_allow_html=True)
    
    # Display model quality metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Quality Prediction Model", f"R² = {quality_r2:.2f}")
    
    with col2:
        st.metric("Production Prediction Model", f"R² = {production_r2:.2f}")
    
    # Feature importance analysis
    st.markdown("### Feature Importance Analysis")
    st.markdown("Which environmental factors have the biggest impact on wine quality and production?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Quality model feature importance
        quality_importance = pd.DataFrame({
            'Feature': model_features,
            'Importance': quality_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            quality_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance for Wine Quality"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Production model feature importance
        production_importance = pd.DataFrame({
            'Feature': model_features,
            'Importance': production_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            production_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance for Wine Production"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Scenario testing tool
    st.markdown("### Scenario Testing Tool")
    st.markdown("Use this tool to test how different environmental conditions might affect wine quality and production.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Environmental inputs
        st.markdown("#### Environmental Conditions")
        temperature = st.slider("Average Temperature (°F)", 50.0, 70.0, 60.0, 0.1)
        rainfall = st.slider("Annual Rainfall (inches)", 10.0, 50.0, 25.0, 0.5)
        frost_days = st.slider("Frost Days", 0, 30, 10)
        heat_days = st.slider("Days over 95°F", 0, 50, 15)
        aqi = st.slider("Average Air Quality Index", 20.0, 150.0, 40.0, 0.5)
        drought = st.slider("Drought Index (0-5)", 0.0, 5.0, 2.0, 0.1)
        smoke = st.slider("Smoke Exposure Index (0-10)", 0.0, 10.0, 1.0, 0.1)
    
    with col2:
        # Prediction results
        st.markdown("#### Prediction Results")
        
        # Create feature vector
        features = np.array([[
            temperature, rainfall, frost_days, heat_days, 
            aqi, drought, smoke
        ]])
        
        # Make predictions
        quality_pred = quality_model.predict(features)[0]
        production_pred = production_model.predict(features)[0]
        
        # Display predictions with gauge charts
        fig_quality = go.Figure(go.Indicator(
            mode="gauge+number",
            value=quality_pred,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Predicted Quality Score"},
            gauge={
                'axis': {'range': [70, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [70, 80], 'color': "lightgray"},
                    {'range': [80, 90], 'color': "gray"},
                    {'range': [90, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        st.plotly_chart(fig_quality, use_container_width=True)
        
        fig_production = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=production_pred,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Predicted Production (% of Normal)"},
            delta={'reference': 100, 'relative': True},
            gauge={
                'axis': {'range': [0, 120]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 70], 'color': "lightcoral"},
                    {'range': [70, 90], 'color': "lightyellow"},
                    {'range': [90, 120], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 100
                }
            }
        ))
        st.plotly_chart(fig_production, use_container_width=True)
    
    # Climate change projection tool
    st.markdown("### Climate Change Projection Tool")
    
    st.markdown("""
    <div class="highlight">
    This tool projects how climate change might affect California's wine industry over the next decade based on 
    different climate scenarios. The projections build on historical trends and apply climate model adjustments.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        projection_region = st.selectbox(
            "Select Region to Project",
            options=list(wine_regions.keys())
        )
        
        climate_scenario = st.radio(
            "Climate Change Scenario",
            options=["Low Impact", "Moderate Impact", "High Impact"]
        )
        
        projection_years = st.slider("Projection Years", 1, 10, 5)
    
    with col2:
        # Get current conditions for selected region
        last_year_data = impact_df[
            (impact_df['year'] == impact_df['year'].max()) & 
            (impact_df['region'] == projection_region)
        ]
        
        if not last_year_data.empty:
            # Set scenario parameters
            if climate_scenario == "Low Impact":
                temp_increase_per_year = 0.05  # °F per year
                rainfall_decrease_per_year = 0.5  # % per year
                extreme_event_factor = 1.05  # 5% increase in extreme events per year
            elif climate_scenario == "Moderate Impact":
                temp_increase_per_year = 0.1
                rainfall_decrease_per_year = 1.0
                extreme_event_factor = 1.1
            else:  # High Impact
                temp_increase_per_year = 0.2
                rainfall_decrease_per_year = 2.0
                extreme_event_factor = 1.2
            
            # Generate projection data
            projection_data = []
            base_year = impact_df['year'].max()
            base_temp = last_year_data['avg_temperature'].iloc[0]
            base_rainfall = last_year_data['rainfall_inches'].iloc[0]
            base_heat_days = last_year_data['days_over_95F'].iloc[0]
            base_drought = last_year_data['drought_index'].iloc[0]
            base_smoke = 1.0  # Baseline smoke exposure
            
            for year_offset in range(1, projection_years + 1):
                projected_year = base_year + year_offset
                
                # Project environmental conditions
                projected_temp = base_temp + (temp_increase_per_year * year_offset)
                projected_rainfall = base_rainfall * (1 - (rainfall_decrease_per_year/100 * year_offset))
                projected_heat_days = base_heat_days * (extreme_event_factor ** year_offset)
                projected_drought = min(5, base_drought + (0.1 * year_offset))
                projected_smoke = base_smoke * (extreme_event_factor ** (year_offset/2))  # Less linear increase
                
                # Limit to realistic values
                projected_heat_days = min(80, projected_heat_days)
                projected_smoke = min(10, projected_smoke)
                
                # Create feature vector for prediction
                features = np.array([[
                    projected_temp, projected_rainfall, 10, projected_heat_days,  # Using constant for frost days
                    30 + projected_smoke * 10, projected_drought, projected_smoke
                ]])
                
                # Make predictions
                quality_pred = quality_model.predict(features)[0]
                production_pred = production_model.predict(features)[0]
                
                projection_data.append({
                    'Year': projected_year,
                    'Temperature': projected_temp,
                    'Rainfall': projected_rainfall,
                    'Heat Days': projected_heat_days,
                    'Drought Index': projected_drought,
                    'Smoke Risk': projected_smoke,
                    'Predicted Quality': quality_pred,
                    'Predicted Production': production_pred
                })
            
            projection_df = pd.DataFrame(projection_data)
            
            # Create projection charts
            st.markdown(f"#### Projected Impacts for {projection_region} - {climate_scenario} Scenario")
            
            # Create subplot with two y-axes
            fig = go.Figure()
            
            # Add production line
            fig.add_trace(go.Scatter(
                x=projection_df['Year'],
                y=projection_df['Predicted Production'],
                mode='lines+markers',
                name='Production (% of Normal)',
                line=dict(color='green', width=2)
            ))
            
            # Add quality line (secondary axis)
            fig.add_trace(go.Scatter(
                x=projection_df['Year'],
                y=projection_df['Predicted Quality'],
                mode='lines+markers',
                name='Quality Score',
                line=dict(color='purple', width=2),
                yaxis='y2'
            ))
            
            # Add reference line at 100% production
            fig.add_hline(
                y=100,
                line_dash="dash",
                line_color="green",
                annotation_text="Normal Production Level",
                annotation_position="top right"
            )
            
            # Add vertical reference line at projection start
            fig.add_vline(
                x=base_year,
                line_dash="dash",
                line_color="gray",
                annotation_text="Projection Start",
                annotation_position="top"
            )
            
            # Update layout with double y-axis
            fig.update_layout(
                title=f"Projected Wine Production and Quality",
                xaxis=dict(title='Year'),
                yaxis=dict(
                    title='Production (% of Normal)',
                    side='left',
                    range=[0, max(150, max(projection_df['Predicted Production']) * 1.1)]
                ),
                yaxis2=dict(
                    title='Quality Score',
                    side='right',
                    overlaying='y',
                    range=[min(70, min(projection_df['Predicted Quality']) * 0.9), 
                           max(100, max(projection_df['Predicted Quality']) * 1.05)]
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key projection metrics
            st.markdown("#### Projected Metrics")
            
            end_production = projection_df.iloc[-1]['Predicted Production']
            end_quality = projection_df.iloc[-1]['Predicted Quality']
            
            production_change = end_production - 100
            quality_change = end_quality - last_year_data['quality_score'].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    f"Production in {base_year + projection_years}",
                    f"{end_production:.1f}% of normal",
                    f"{production_change:+.1f}%"
                )
            
            with col2:
                st.metric(
                    f"Quality in {base_year + projection_years}",
                    f"{end_quality:.1f}/100",
                    f"{quality_change:+.1f} pts"
                )
            
            with col3:
                # Estimated economic impact
                current_revenue = last_year_data['revenue_millions'].iloc[0]
                projected_revenue = current_revenue * (end_production / 100) * (end_quality / last_year_data['quality_score'].iloc[0])
                revenue_change = projected_revenue - current_revenue
                
                st.metric(
                    f"Revenue Impact in {base_year + projection_years}",
                    f"${projected_revenue:.1f}M",
                    f"{revenue_change:+.1f}M"
                )
            
            # Display environmental changes
            st.markdown("#### Projected Environmental Changes")
            
            env_df = projection_df[['Year', 'Temperature', 'Rainfall', 'Heat Days', 'Drought Index', 'Smoke Risk']]
            st.dataframe(env_df.style.format({
                'Temperature': '{:.1f}°F',
                'Rainfall': '{:.1f} inches',
                'Heat Days': '{:.0f} days',
                'Drought Index': '{:.1f}/5',
                'Smoke Risk': '{:.1f}/10'
            }))
            
            # Add recommendations based on projections
            st.markdown("#### Adaptation Recommendations")
            
            if end_production < 85 or end_quality < 85:
                st.markdown("""
                <div class="highlight" style="border-left-color: #e74c3c;">
                <strong>High Risk Scenario</strong>: Significant negative impacts projected require major adaptation strategies:
                <ul>
                    <li>Explore alternative varietals more resilient to projected conditions</li>
                    <li>Invest in advanced irrigation systems and water storage</li>
                    <li>Implement smoke exposure mitigation systems</li>
                    <li>Consider geographic diversification to reduce risk</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            elif end_production < 95 or end_quality < 90:
                st.markdown("""
                <div class="highlight" style="border-left-color: #f39c12;">
                <strong>Moderate Risk Scenario</strong>: Some negative impacts projected require targeted adaptations:
                <ul>
                    <li>Adjust canopy management to reduce sun exposure</li>
                    <li>Implement water conservation measures</li>
                    <li>Increase monitoring for smoke-related issues</li>
                    <li>Experiment with heat-resistant clones</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="highlight" style="border-left-color: #27ae60;">
                <strong>Low Risk Scenario</strong>: Minimal impacts projected, but preparedness is advised:
                <ul>
                    <li>Monitor changing conditions and establish early warning systems</li>
                    <li>Develop contingency plans for extreme weather events</li>
                    <li>Maintain water reserves and efficient irrigation</li>
                    <li>Participate in industry research on climate adaptation</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.error(f"No current data available for {projection_region}")

# Add a footer with data sources
st.markdown("---")
st.markdown("""
<div class="footer">
<p>Data sources: California Department of Forestry and Fire Protection (CAL FIRE), NOAA Climate Data, California Department of Food and Agriculture</p>
<p>© 2025 California Wine Industry Environmental Impact Analyzer</p>
</div>
""", unsafe_allow_html=True)