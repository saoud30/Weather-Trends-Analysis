import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import numpy as np
from windrose import WindroseAxes
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced Weather Trends Analysis",
    page_icon="🌤️",
    layout="wide"
)

# API Configuration
API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "http://api.openweathermap.org/data/2.5/forecast"
AIR_QUALITY_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
GEOCODING_URL = "http://api.openweathermap.org/geo/1.0/direct"
REVERSE_GEOCODING_URL = "http://api.openweathermap.org/geo/1.0/reverse"

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Define default Asian cities
DEFAULT_CITIES = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "country": "India"},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "country": "India"},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639, "country": "India"},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946, "country": "India"},
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "country": "India"},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462, "country": "India"},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "country": "India"},
    "Shenzhen": {"lat": 22.5431, "lon": 114.0579, "country": "China"},
    "Shanghai": {"lat": 31.2304, "lon": 121.4737, "country": "China"},
    "Beijing": {"lat": 39.9042, "lon": 116.4074, "country": "China"},
    "Tokyo": {"lat": 35.6762, "lon": 139.6503, "country": "Japan"},
    "Seoul": {"lat": 37.5665, "lon": 126.9780, "country": "South Korea"},
    "Singapore": {"lat": 1.3521, "lon": 103.8198, "country": "Singapore"},
    "Bangkok": {"lat": 13.7563, "lon": 100.5018, "country": "Thailand"},
    "Dubai": {"lat": 25.2048, "lon": 55.2708, "country": "UAE"}
}

# Initialize CITIES with default cities
CITIES = DEFAULT_CITIES.copy()

class WeatherData:
    def __init__(self):
        self.api_key = API_KEY
        
    def fetch_weather_data(self, city_name, start_date, end_date):
        """Fetch comprehensive weather data including forecasts and air quality"""
        city_info = CITIES.get(city_name)
        if not city_info:
            return None

        try:
            # Current weather
            weather_params = {
                'lat': city_info['lat'],
                'lon': city_info['lon'],
                'appid': self.api_key,
                'units': 'metric'
            }
            current_weather = requests.get(BASE_URL, params=weather_params).json()

            # 5-day forecast
            forecast_params = weather_params.copy()
            forecast_data = requests.get(FORECAST_URL, params=forecast_params).json()

            # Air quality
            air_params = {
                'lat': city_info['lat'],
                'lon': city_info['lon'],
                'appid': self.api_key
            }
            air_quality = requests.get(AIR_QUALITY_URL, params=air_params).json()

            return {
                'current': current_weather,
                'forecast': forecast_data,
                'air_quality': air_quality
            }
        except requests.RequestException as e:
            st.error(f"Error fetching data for {city_name}: {str(e)}")
            return None

def generate_sample_data(city, start_date, end_date):
    """Generate enhanced sample weather data with more parameters"""
    date_range = pd.date_range(start=start_date, end=end_date, freq='h')
    base_temp = {
        'summer': {'day': 35, 'night': 25},
        'winter': {'day': 20, 'night': 10},
        'monsoon': {'day': 30, 'night': 23}
    }

    # Adjust base temperature based on city's location
    city_info = CITIES.get(city)
    if city_info:
        latitude = city_info['lat']
        # Adjust temperature based on latitude
        temp_adjustment = (abs(latitude) - 20) * 0.5
    else:
        temp_adjustment = 0

    data = {
        'date': date_range,
        'temperature': [
            np.random.normal(
                base_temp['summer']['day'] - temp_adjustment 
                if 6 <= hour <= 18 
                else base_temp['summer']['night'] - temp_adjustment, 
                2
            ) for hour in date_range.hour
        ],
        'feels_like': [],
        'precipitation': np.random.exponential(2, len(date_range)),
        'humidity': np.random.normal(70, 10, len(date_range)),
        'wind_speed': np.random.normal(15, 5, len(date_range)),
        'wind_direction': np.random.uniform(0, 360, len(date_range)),
        'pressure': np.random.normal(1013, 5, len(date_range)),
        'visibility': np.random.normal(10, 2, len(date_range)),
        'uv_index': [],
        'air_quality_index': np.random.normal(50, 20, len(date_range)),
        'cloud_cover': np.random.normal(50, 20, len(date_range))
    }

    # Calculate feels like temperature
    for i in range(len(date_range)):
        temp = data['temperature'][i]
        humidity = data['humidity'][i]
        wind_speed = data['wind_speed'][i]
        
        # Simple heat index calculation
        feels_like = temp + 0.348 * humidity - 0.7 * wind_speed
        data['feels_like'].append(feels_like)
        
        # Calculate UV index based on hour
        hour = date_range[i].hour
        if 6 <= hour <= 18:
            uv = np.random.normal(8, 2)
        else:
            uv = 0
        data['uv_index'].append(max(0, min(11, uv)))

    df = pd.DataFrame(data)
    df['city'] = city
    df['country'] = CITIES.get(city, {}).get('country', 'Unknown')
    return df

def create_line_chart(df, parameter):
    """Create line chart for selected parameter"""
    fig = px.line(df, x='date', y=parameter, color='city',
                  title=f'{parameter.title()} Trends Over Time')
    return fig

def create_bar_chart(df, parameter, comparison_type):
    """Create bar chart for selected parameter"""
    df = df.copy()  # Create a copy to avoid modifying original dataframe
    
    # Convert date to appropriate format
    if isinstance(df['date'].iloc[0], str):
        df['date'] = pd.to_datetime(df['date'])
        
    if comparison_type == "Daily Average":
        df['period'] = df['date'].dt.strftime('%Y-%m-%d')
    elif comparison_type == "Weekly Average":
        df['period'] = df['date'].dt.strftime('%Y-W%U')
    else:  # Monthly Average
        df['period'] = df['date'].dt.strftime('%Y-%m')

    grouped_avg = df.groupby(['city', 'period'])[parameter].mean().reset_index()
    fig = px.bar(grouped_avg, x='period', y=parameter, color='city',
                 title=f'{comparison_type} Average {parameter.title()}')
    return fig

def create_heatmap(df, parameter):
    """Create heatmap for selected parameter"""
    pivot_table = df.pivot_table(
        values=parameter,
        index=df['date'].dt.month,
        columns=df['date'].dt.day,
        aggfunc='mean'
    )
    fig = px.imshow(pivot_table, title=f'{parameter.title()} Heatmap')
    return fig

def create_contour_plot(df, parameter):
    """Create contour plot for selected parameter"""
    pivot_table = df.pivot_table(
        values=parameter,
        index=df['date'].dt.month,
        columns=df['date'].dt.day,
        aggfunc='mean'
    )
    fig = px.imshow(pivot_table, title=f'{parameter.title()} Contour Plot')
    return fig

def create_3d_surface(df, parameter):
    """Create 3D surface plot for selected parameter"""
    pivot_table = df.pivot_table(
        values=parameter,
        index=df['date'].dt.month,
        columns=df['date'].dt.day,
        aggfunc='mean'
    )
    fig = go.Figure(data=[go.Surface(z=pivot_table.values, x=pivot_table.columns, y=pivot_table.index)])
    fig.update_layout(title=f'{parameter.title()} 3D Surface')
    return fig

def create_distribution_plot(df, parameter):
    """Create distribution plot for selected parameter"""
    fig = px.histogram(df, x=parameter, color='city', title=f'{parameter.title()} Distribution')
    return fig

def get_statistics(df, parameter):
    """Calculate statistics for selected parameter"""
    stats = {
        'Mean': df[parameter].mean(),
        'Median': df[parameter].median(),
        'Max': df[parameter].max(),
        'Min': df[parameter].min(),
        'Std Dev': df[parameter].std(),
        'Variance': df[parameter].var()
    }
    return stats

def create_regional_analysis(df, parameters, region_type):
    """Create regional analysis for selected parameters"""
    df = df.copy()  # Create a copy to avoid modifying original dataframe
    
    if region_type == "Country":
        group_by = 'country'
    elif region_type == "Climate Zone":
        # Add climate zone based on latitude
        df['climate_zone'] = df.apply(lambda row: get_climate_zone(row['city'], CITIES), axis=1)
        group_by = 'climate_zone'
    else:  # Latitude Band
        df['latitude_band'] = df.apply(lambda row: get_latitude_band(row['city'], CITIES), axis=1)
        group_by = 'latitude_band'

    for parameter in parameters:
        regional_avg = df.groupby(group_by)[parameter].mean().reset_index()
        fig = px.bar(regional_avg, x=group_by, y=parameter, 
                    title=f'Average {parameter.title()} by {region_type}')
        st.plotly_chart(fig, use_container_width=True)

def get_climate_zone(city, cities_dict):
    """Determine climate zone based on latitude"""
    if city not in cities_dict:
        return "Unknown"
    
    lat = cities_dict[city]['lat']
    
    if lat > 66.5:
        return "Polar"
    elif lat > 50:
        return "Temperate"
    elif lat > 23.5:
        return "Subtropical"
    elif lat > -23.5:
        return "Tropical"
    elif lat > -50:
        return "Subtropical"
    elif lat > -66.5:
        return "Temperate"
    else:
        return "Polar"

def get_latitude_band(city, cities_dict):
    """Determine latitude band"""
    if city not in cities_dict:
        return "Unknown"
    
    lat = cities_dict[city]['lat']
    
    if lat > 60:
        return "60°N to 90°N"
    elif lat > 30:
        return "30°N to 60°N"
    elif lat > 0:
        return "0° to 30°N"
    elif lat > -30:
        return "0° to 30°S"
    elif lat > -60:
        return "30°S to 60°S"
    else:
        return "60°S to 90°S"

def export_data(df, format_type, fig=None, filename=None):
    """Export data in various formats"""
    try:
        if format_type == "CSV":
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="weather_data.csv">📥 Download CSV Report</a>'
            return href
        
        elif format_type == "Excel":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            b64 = base64.b64encode(output.getvalue()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="weather_data.xlsx">📥 Download Excel Report</a>'
            return href
        
        elif format_type == "Summary Stats":
            # Create a summary statistics DataFrame
            summary = pd.DataFrame()
            for param in df.select_dtypes(include=[np.number]).columns:
                if param != 'date':
                    stats = df.groupby('city')[param].agg(['mean', 'min', 'max', 'std']).round(2)
                    summary = pd.concat([summary, stats], axis=1)
            
            # Export summary to Excel with formatting
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                summary.to_excel(writer, sheet_name='Summary Statistics')
                # Add some formatting
                workbook = writer.book
                worksheet = writer.sheets['Summary Statistics']
                header_format = workbook.add_format({'bold': True, 'bg_color': '#D3D3D3'})
                for col_num, value in enumerate(summary.columns.values):
                    worksheet.write(0, col_num + 1, value, header_format)
            
            b64 = base64.b64encode(output.getvalue()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="weather_summary_stats.xlsx">📥 Download Summary Statistics</a>'
            return href
        
        elif format_type in ["PNG", "PDF", "SVG"]:
            if fig is None:
                return "No visualization selected for export"
            
            # Create a temporary file for the visualization
            temp_file = f"temp_viz.{format_type.lower()}"
            if format_type == "PDF":
                fig.write_image(temp_file, format="pdf")
            else:
                fig.write_image(temp_file)
            
            # Read the file and create download link
            with open(temp_file, "rb") as file:
                b64 = base64.b64encode(file.read()).decode()
            
            # Remove temporary file
            os.remove(temp_file)
            
            mime_types = {
                "PNG": "image/png",
                "PDF": "application/pdf",
                "SVG": "image/svg+xml"
            }
            
            href = f'<a href="data:application/{mime_types[format_type]};base64,{b64}" download="weather_visualization.{format_type.lower()}">📥 Download {format_type}</a>'
            return href
            
    except Exception as e:
        return f"Error during export: {str(e)}"

def get_ai_insights(data_description, analysis_type):
    """Get insights from Gemini-1.5-flash model"""
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=generation_config,
        )

        prompts = {
            "Trends": """Analyze this weather data and explain the trends in simple terms:
            {data_description}
            Focus on:
            1. What are the main weather trends?
            2. Are there any concerning patterns?
            3. What do these trends mean for daily life?
            4. Any recommendations based on these trends?""",
            
            "Comparisons": """Compare the weather conditions between different cities:
            {data_description}
            Focus on:
            1. Which cities have better/worse conditions?
            2. What are the main differences between cities?
            3. Why might these differences exist?
            4. What should people in each city be aware of?""",
            
            "Patterns": """Explain these weather patterns in simple terms:
            {data_description}
            Focus on:
            1. What patterns do you notice?
            2. Are these patterns normal?
            3. How do these patterns affect daily life?
            4. What should people prepare for?""",
            
            "Statistics": """Explain these weather statistics in non-technical terms:
            {data_description}
            Focus on:
            1. What do these numbers mean for everyday life?
            2. Are these values normal or concerning?
            3. Which areas need attention?
            4. What actions should people take?""",
            
            "Regional": """Analyze the regional weather differences:
            {data_description}
            Focus on:
            1. How do different regions compare?
            2. What causes these regional differences?
            3. What are the implications for each region?
            4. Region-specific recommendations?"""
        }

        chat = model.start_chat(history=[])
        prompt = prompts[analysis_type].format(data_description=data_description)
        response = chat.send_message(prompt)
        
        return response.text
            
    except Exception as e:
        return f"Error generating insights: {str(e)}"

def prepare_data_description(df, parameters, analysis_type):
    """Prepare data description for AI analysis"""
    description = []
    
    description.append(f"Time period: {df['date'].min()} to {df['date'].max()}")
    description.append(f"Cities analyzed: {', '.join(df['city'].unique())}")
    
    for param in parameters:
        param_stats = df.groupby('city')[param].agg(['mean', 'min', 'max']).round(2)
        description.append(f"\n{param.replace('_', ' ').title()} Analysis:")
        
        for city in param_stats.index:
            stats = param_stats.loc[city]
            description.append(f"- {city}: Average: {stats['mean']}, Range: {stats['min']} to {stats['max']}")
            
            if param == 'air_quality_index':
                if stats['mean'] > 150:
                    description.append(f"  ⚠️ Air quality in {city} is concerning")
            elif param == 'temperature':
                if stats['max'] > 35:
                    description.append(f"  ⚠️ High temperature alerts for {city}")
            elif param == 'humidity':
                if stats['mean'] > 80:
                    description.append(f"  ⚠️ High humidity levels in {city}")
    
    return "\n".join(description)

def search_location(query):
    """Search for a location using OpenWeatherMap Geocoding API"""
    try:
        params = {
            'q': query,
            'limit': 5,  # Get top 5 matches
            'appid': os.getenv('OPENWEATHERMAP_API_KEY')
        }
        
        response = requests.get(GEOCODING_URL, params=params)
        response.raise_for_status()
        results = response.json()
        
        if not results:
            return []
        
        # Format results for display
        formatted_results = []
        for result in results:
            location = {
                'name': result['name'],
                'state': result.get('state', ''),
                'country': result['country'],
                'lat': result['lat'],
                'lon': result['lon'],
                'display_name': f"{result['name']}, {result.get('state', '')}, {result['country']}".replace(', ,', ',')
            }
            formatted_results.append(location)
        
        return formatted_results
    
    except Exception as e:
        st.error(f"Error searching location: {str(e)}")
        return []

def get_current_weather(lat, lon):
    """Get current weather for coordinates"""
    try:
        params = {
            'lat': lat,
            'lon': lon,
            'appid': os.getenv('OPENWEATHERMAP_API_KEY'),
            'units': 'metric'
        }
        
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        return response.json()
    
    except Exception as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return None

def main():
    st.title("🌍 Advanced Weather Trends Analysis")
    
    # Add AI Assistant sidebar toggle
    st.sidebar.header("🤖 AI Assistant")
    enable_ai = st.sidebar.toggle("Enable AI Insights", value=True)
    
    # City Selection Section
    st.sidebar.header("🌆 Select Cities")
    
    # Group cities by country
    cities_by_country = {}
    for city, info in DEFAULT_CITIES.items():
        country = info['country']
        if country not in cities_by_country:
            cities_by_country[country] = []
        cities_by_country[country].append(city)
    
    # Display cities grouped by country with checkboxes
    selected_default_cities = []
    for country in sorted(cities_by_country.keys()):
        with st.sidebar.expander(f"📍 {country}"):
            for city in sorted(cities_by_country[country]):
                if st.checkbox(city, value=True, key=f"default_{city}"):
                    selected_default_cities.append(city)
    
    # Additional City Search
    st.sidebar.header("🔍 Search Additional Cities")
    search_query = st.sidebar.text_input("Search for any city", "")
    
    # Initialize session state for additional cities if not exists
    if 'additional_cities' not in st.session_state:
        st.session_state.additional_cities = {}
    
    if search_query:
        results = search_location(search_query)
        if results:
            st.sidebar.subheader("Search Results")
            for result in results:
                location_key = f"{result['lat']},{result['lon']}"
                if st.sidebar.button(
                    f"📍 {result['display_name']}",
                    key=f"btn_{location_key}",
                    help=f"Lat: {result['lat']}, Lon: {result['lon']}"
                ):
                    # Add to additional cities
                    st.session_state.additional_cities[location_key] = result
                    st.success(f"Added {result['display_name']} to analysis!")
        else:
            st.sidebar.warning("No locations found. Try a different search term.")
    
    # Display additional selected cities
    if st.session_state.additional_cities:
        st.sidebar.subheader("📌 Additional Selected Cities")
        for location_key, location in list(st.session_state.additional_cities.items()):
            col1, col2 = st.sidebar.columns([3, 1])
            col1.write(location['display_name'])
            if col2.button("❌", key=f"remove_{location_key}"):
                del st.session_state.additional_cities[location_key]
                st.rerun()
    
    # Update CITIES dictionary with selected cities
    CITIES.clear()
    
    # Add selected default cities
    for city in selected_default_cities:
        CITIES[city] = DEFAULT_CITIES[city]
    
    # Add additional cities
    for location_key, location in st.session_state.additional_cities.items():
        city_name = f"{location['name']}, {location['country']}"
        CITIES[city_name] = {
            "lat": location['lat'],
            "lon": location['lon'],
            "country": location['country']
        }
    
    # If no cities selected, show message
    if not CITIES:
        st.info("👆 Please select at least one city from the sidebar!")
        return

    # Time range selection with presets
    st.sidebar.header("Time Range")
    preset_ranges = {
        "Last 24 Hours": timedelta(days=1),
        "Last Week": timedelta(days=7),
        "Last Month": timedelta(days=30),
        "Last Year": timedelta(days=365)
    }
    
    time_range = st.sidebar.selectbox("Select Time Range", list(preset_ranges.keys()) + ["Custom"])
    
    if time_range == "Custom":
        date_col1, date_col2 = st.sidebar.columns(2)
        with date_col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
        with date_col2:
            end_date = st.date_input("End Date", datetime.now())
    else:
        end_date = datetime.now()
        start_date = end_date - preset_ranges[time_range]

    # Enhanced parameter selection with categories
    st.sidebar.header("Parameters")
    parameter_categories = {
        "Temperature": ["temperature", "feels_like"],
        "Precipitation": ["precipitation", "humidity"],
        "Wind": ["wind_speed", "wind_direction"],
        "Air Quality": ["air_quality_index", "visibility"],
        "Other": ["pressure", "cloud_cover", "uv_index"]
    }

    selected_parameters = []
    for category, params in parameter_categories.items():
        with st.sidebar.expander(f"📊 {category}"):
            for param in params:
                if st.checkbox(param.replace("_", " ").title(), key=f"param_{param}"):
                    selected_parameters.append(param)

    if not selected_parameters:
        st.warning("Please select at least one parameter to analyze")
        return

    # Create tabs with more analysis options
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Trends", 
        "🔄 Comparisons", 
        "🌡️ Patterns",
        "📊 Statistics",
        "🌍 Regional Analysis"
    ])

    # Collect and process data
    all_data = pd.DataFrame()
    for city in CITIES.keys():
        city_data = generate_sample_data(city, start_date, end_date)
        all_data = pd.concat([all_data, city_data])

    # Add the rest of the visualization and analysis code here...
    with tab1:
        st.header("Weather Trends")
        if enable_ai:
            with st.expander("🤖 AI Weather Insights", expanded=True):
                data_description = prepare_data_description(all_data, selected_parameters, "Trends")
                insights = get_ai_insights(data_description, "Trends")
                st.markdown(insights)
        
        for param in selected_parameters:
            fig = create_line_chart(all_data, param)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("City Comparisons")
        if enable_ai:
            with st.expander("🤖 AI Comparison Insights", expanded=True):
                data_description = prepare_data_description(all_data, selected_parameters, "Comparisons")
                insights = get_ai_insights(data_description, "Comparisons")
                st.markdown(insights)
        
        comparison_type = st.selectbox(
            "Select Comparison Type",
            ["Daily Average", "Weekly Average", "Monthly Average"]
        )
        for param in selected_parameters:
            fig = create_bar_chart(all_data, param, comparison_type)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Weather Patterns")
        if enable_ai:
            with st.expander("🤖 AI Pattern Insights", expanded=True):
                data_description = prepare_data_description(all_data, selected_parameters, "Patterns")
                insights = get_ai_insights(data_description, "Patterns")
                st.markdown(insights)
        
        pattern_view = st.selectbox(
            "Select Pattern View",
            ["Heatmap", "Contour Plot", "3D Surface"]
        )
        for param in selected_parameters:
            if pattern_view == "Heatmap":
                fig = create_heatmap(all_data, param)
            elif pattern_view == "Contour Plot":
                fig = create_contour_plot(all_data, param)
            else:
                fig = create_3d_surface(all_data, param)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("Statistical Analysis")
        if enable_ai:
            with st.expander("🤖 AI Statistical Insights", expanded=True):
                data_description = prepare_data_description(all_data, selected_parameters, "Statistics")
                insights = get_ai_insights(data_description, "Statistics")
                st.markdown(insights)
        
        for param in selected_parameters:
            st.subheader(f"{param.replace('_', ' ').title()} Statistics")
            stats = get_statistics(all_data, param)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean", f"{stats['Mean']:.2f}")
                st.metric("Median", f"{stats['Median']:.2f}")
            with col2:
                st.metric("Max", f"{stats['Max']:.2f}")
                st.metric("Min", f"{stats['Min']:.2f}")
            with col3:
                st.metric("Std Dev", f"{stats['Std Dev']:.2f}")
                st.metric("Variance", f"{stats['Variance']:.2f}")

            fig = create_distribution_plot(all_data, param)
            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.header("Regional Analysis")
        if enable_ai:
            with st.expander("🤖 AI Regional Insights", expanded=True):
                data_description = prepare_data_description(all_data, selected_parameters, "Regional")
                insights = get_ai_insights(data_description, "Regional")
                st.markdown(insights)
        
        region_type = st.selectbox(
            "Select Region Type",
            ["Country", "Climate Zone", "Latitude Band"]
        )
        create_regional_analysis(all_data, selected_parameters, region_type)

    # Export section with enhanced options
    st.sidebar.header("📤 Export Data")
    export_format = st.sidebar.selectbox(
        "Select Export Format",
        ["CSV", "Excel", "Summary Stats", "PNG", "PDF", "SVG"]
    )
    
    if export_format in ["PNG", "PDF", "SVG"]:
        # Allow users to select which visualization to export
        st.sidebar.subheader("Select Visualization to Export")
        viz_type = st.sidebar.selectbox(
            "Visualization Type",
            ["Temperature Trends", "Rainfall Comparison", "Heat Map", "Wind Rose", "Distribution Plot"]
        )
        
        if st.sidebar.button("Generate Export"):
            # Create the selected visualization
            if viz_type == "Temperature Trends":
                fig = create_line_chart(all_data, "temperature")
            elif viz_type == "Rainfall Comparison":
                fig = create_bar_chart(all_data, "precipitation", "Monthly Average")
            elif viz_type == "Heat Map":
                fig = create_heatmap(all_data, "temperature")
            elif viz_type == "Distribution Plot":
                fig = create_distribution_plot(all_data, "temperature")
            
            # Generate download link
            download_link = export_data(all_data, export_format, fig=fig)
            st.sidebar.markdown(download_link, unsafe_allow_html=True)
    
    else:
        if st.sidebar.button("Generate Export"):
            download_link = export_data(all_data, export_format)
            st.sidebar.markdown(download_link, unsafe_allow_html=True)
    
    # Add export instructions
    with st.sidebar.expander("📋 Export Instructions"):
        st.markdown("""
        **Available Export Options:**
        - **CSV**: Raw data in CSV format
        - **Excel**: Formatted data in Excel
        - **Summary Stats**: Statistical summary in Excel
        - **PNG/PDF/SVG**: High-quality visualizations
        
        **Steps to Export:**
        1. Select the desired format
        2. For visualizations, choose the type
        3. Click 'Generate Export'
        4. Click the download link
        """)

if __name__ == "__main__":
    main()
