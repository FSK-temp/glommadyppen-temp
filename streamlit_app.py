"""
Glommadyppen Vanntemperatur Prediksjon
Real-time water temperature prediction for Glommadyppen swimming event

Author: Anton
Date: December 2024
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Glommadyppen Temperatur",
    page_icon="🏊‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# NVE API Configuration
# Try to get from secrets, if not available use placeholder
try:
    NVE_API_KEY = st.secrets["nve_api_key"]
except (KeyError, FileNotFoundError):
    NVE_API_KEY = None
    
NVE_BASE_URL = "https://hydapi.nve.no/api/v1"

# Station IDs
STATION_VORMA = "2.410.0"  # Funnefoss overvann
STATION_BLAKER = "2.17.0"  # Blaker (Glomma)

# Weather location (Mjøsa)
MJOSA_LAT = 60.403489
MJOSA_LON = 11.230855

# Model parameters (from research)
TRAVEL_TIME_HOURS = 25  # Vorma to Fetsund
TEMPERATURE_SURVIVAL = 0.14  # 14% of drop survives dilution
CRITICAL_WIND_SPEED = 1.9  # m/s sustained southerly

# Event information
EVENT_NAME = "Glommadyppen"
EVENT_MONTH = 8  # August
EVENT_DAY_OF_WEEK = 5  # Saturday (0=Monday, 5=Saturday)
EVENT_WEEK = 1  # First Saturday
EVENT_YEAR = 2026

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_nve_data(station_id, parameter, hours_back=72):
    """Fetch data from NVE HydAPI"""
    try:
        url = f"{NVE_BASE_URL}/Observations"
        headers = {
            "X-API-Key": NVE_API_KEY,
            "accept": "application/json"
        }
        
        # Try without ReferenceTime first (gets most recent data)
        params = {
            "StationId": station_id,
            "Parameter": str(parameter),
            "ResolutionTime": "60"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get('data') and len(data['data']) > 0:
            observations = data['data'][0]['observations']
            df = pd.DataFrame(observations)
            df['time'] = pd.to_datetime(df['time'])
            
            end_time = pd.Timestamp.now(tz='UTC')
            cutoff_time = end_time - pd.Timedelta(hours=hours_back)
            df = df[df['time'] >= cutoff_time]
            df = df[df['quality'].isin([1, 2])]  # Quality controlled data only
            df = df.sort_values('time').reset_index(drop=True)
            
            return df[['time', 'value', 'quality']]
        else:
            return pd.DataFrame(columns=['time', 'value', 'quality'])
            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            # Station might not have current data (winter shutdown)
            return pd.DataFrame(columns=['time', 'value', 'quality'])
        else:
            st.warning(f"NVE API error: {e.response.status_code}")
            return pd.DataFrame(columns=['time', 'value', 'quality'])
    except Exception as e:
        st.warning(f"Could not fetch data: {str(e)[:100]}")
        return pd.DataFrame(columns=['time', 'value', 'quality'])

@st.cache_data(ttl=21600)  # Cache for 6 hours
def fetch_weather_forecast(lat, lon, days_ahead=7):
    """Fetch weather forecast from Met.no"""
    try:
        url = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
        headers = {"User-Agent": "GlommadyppenApp/1.0"}
        params = {"lat": lat, "lon": lon}
        
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        forecast_list = []
        max_time = pd.Timestamp.now(tz='UTC') + pd.Timedelta(days=days_ahead)
        
        for ts in data['properties']['timeseries']:
            time = pd.to_datetime(ts['time'])
            if time > max_time:
                break
                
            details = ts['data']['instant']['details']
            forecast_list.append({
                'time': time,
                'air_temperature': details.get('air_temperature'),
                'wind_speed': details.get('wind_speed'),
                'wind_direction': details.get('wind_from_direction'),
                'wind_gust': details.get('wind_speed_of_gust')
            })
        
        return pd.DataFrame(forecast_list)
        
    except Exception as e:
        st.error(f"Error fetching weather forecast: {e}")
        return pd.DataFrame()

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_southerly_wind(df):
    """Calculate southerly wind component (135° to 225°)"""
    if df.empty or 'wind_direction' not in df.columns:
        return df
    
    is_southerly = (df['wind_direction'] >= 135) & (df['wind_direction'] <= 225)
    df['southerly_wind'] = np.where(is_southerly, df['wind_speed'], 0)
    return df

def detect_temperature_drop(df, threshold_C=2.0, window_hours=6):
    """Detect significant temperature drops"""
    if df.empty or len(df) < 2:
        return None
    
    df = df.sort_values('time').copy()
    recent_cutoff = df['time'].max() - pd.Timedelta(hours=window_hours)
    recent = df[df['time'] >= recent_cutoff]
    
    if len(recent) < 2:
        return None
    
    max_temp = recent['value'].max()
    min_temp = recent['value'].min()
    drop = max_temp - min_temp
    
    if drop >= threshold_C:
        max_time = recent[recent['value'] == max_temp]['time'].iloc[0]
        min_time = recent[recent['value'] == min_temp]['time'].iloc[0]
        
        return {
            'magnitude': drop,
            'max_temp': max_temp,
            'min_temp': min_temp,
            'max_time': max_time,
            'min_time': min_time,
            'duration_hours': (min_time - max_time).total_seconds() / 3600
        }
    
    return None

def predict_fetsund_temperature(vorma_temp_df, event_datetime):
    """
    Predict Fetsund temperature for event based on Vorma temperature
    Using 25-hour travel time and 14% survival rate
    """
    if vorma_temp_df.empty:
        return None
    
    # Ensure event_datetime is timezone-aware
    if event_datetime.tzinfo is None:
        event_datetime = event_datetime.replace(tzinfo=pd.Timestamp.now(tz='UTC').tzinfo)
    
    # Get Vorma temperature 25 hours before event
    prediction_time = event_datetime - timedelta(hours=TRAVEL_TIME_HOURS)
    
    # Ensure time column is timezone-aware
    vorma_temp_df = vorma_temp_df.copy()
    vorma_temp_df['time'] = pd.to_datetime(vorma_temp_df['time'])
    if vorma_temp_df['time'].dt.tz is None:
        vorma_temp_df['time'] = vorma_temp_df['time'].dt.tz_localize('UTC')
    
    # Find closest observation
    vorma_temp_df['time_diff'] = abs(vorma_temp_df['time'] - prediction_time)
    closest_idx = vorma_temp_df['time_diff'].idxmin()
    
    if pd.isna(closest_idx):
        return None
    
    vorma_temp = vorma_temp_df.loc[closest_idx, 'value']
    vorma_time = vorma_temp_df.loc[closest_idx, 'time']
    
    # Calculate baseline (average temperature in last 48 hours)
    recent_48h = vorma_temp_df[
        vorma_temp_df['time'] >= (vorma_time - timedelta(hours=48))
    ]
    baseline_temp = recent_48h['value'].mean()
    
    # Calculate temperature anomaly
    anomaly = vorma_temp - baseline_temp
    
    # Apply survival rate (14% of anomaly survives)
    fetsund_anomaly = anomaly * TEMPERATURE_SURVIVAL
    
    # Predicted Fetsund temperature
    fetsund_temp = baseline_temp + fetsund_anomaly
    
    return {
        'predicted_temp': fetsund_temp,
        'vorma_temp': vorma_temp,
        'baseline_temp': baseline_temp,
        'anomaly': anomaly,
        'vorma_time': vorma_time,
        'confidence': calculate_confidence(vorma_temp_df, prediction_time)
    }

def calculate_confidence(df, target_time):
    """Calculate prediction confidence based on data quality and age"""
    if df.empty:
        return 0.0
    
    # Time since last observation
    latest_time = pd.to_datetime(df['time'].max())
    if latest_time.tz is None:
        latest_time = latest_time.tz_localize('UTC')
    if target_time.tz is None:
        target_time = target_time.tz_localize('UTC')
    
    hours_old = (target_time - latest_time).total_seconds() / 3600
    
    # Confidence decreases with data age
    if hours_old < 1:
        time_confidence = 1.0
    elif hours_old < 6:
        time_confidence = 0.9
    elif hours_old < 24:
        time_confidence = 0.7
    else:
        time_confidence = 0.5
    
    # Data completeness
    expected_points = 72  # Last 72 hours
    actual_points = len(df)
    completeness = min(actual_points / expected_points, 1.0)
    
    return time_confidence * completeness

def assess_risk_level(prediction, weather_forecast):
    """Assess overall risk level for the event"""
    if prediction is None:
        return "UNKNOWN", "gray"
    
    predicted_temp = prediction['predicted_temp']
    anomaly = prediction['anomaly']
    
    # Check weather forecast for southerly winds
    southerly_risk = False
    if not weather_forecast.empty:
        next_48h = weather_forecast.head(48)
        if 'southerly_wind' in next_48h.columns:
            avg_southerly = next_48h['southerly_wind'].mean()
            southerly_risk = avg_southerly >= 1.5
    
    # Risk assessment
    if predicted_temp < 14 or anomaly < -3:
        return "HØYRISIKOGROUP", "#dc3545"  # Red
    elif predicted_temp < 16 or anomaly < -2 or southerly_risk:
        return "MODERAT RISIKO", "#ffc107"  # Yellow
    elif predicted_temp < 18:
        return "LAV RISIKO", "#17a2b8"  # Blue
    else:
        return "GODE FORHOLD", "#28a745"  # Green

def calculate_event_date(year):
    """Calculate first Saturday of August"""
    # Start from August 1st
    first_day = datetime(year, EVENT_MONTH, 1)
    
    # Find first Saturday
    days_until_saturday = (EVENT_DAY_OF_WEEK - first_day.weekday()) % 7
    if days_until_saturday == 0 and first_day.weekday() != EVENT_DAY_OF_WEEK:
        days_until_saturday = 7
    
    event_date = first_day + timedelta(days=days_until_saturday)
    
    # Set time to 10:00 (event start) and make timezone-aware
    event_date = event_date.replace(hour=10, minute=0, second=0)
    event_date = pd.Timestamp(event_date).tz_localize('Europe/Oslo').tz_convert('UTC')
    
    return event_date

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_temperature_chart(vorma_df, fetsund_df=None):
    """Create interactive temperature chart"""
    fig = go.Figure()
    
    if not vorma_df.empty:
        # Handle both 'value' and 'temperature' column names
        temp_col = 'temperature' if 'temperature' in vorma_df.columns else 'value'
        
        fig.add_trace(go.Scatter(
            x=vorma_df['time'],
            y=vorma_df[temp_col],
            mode='lines+markers',
            name='Vorma (Funnefoss)',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=4)
        ))
    
    if fetsund_df is not None and not fetsund_df.empty:
        temp_col = 'temperature' if 'temperature' in fetsund_df.columns else 'value'
        
        fig.add_trace(go.Scatter(
            x=fetsund_df['time'],
            y=fetsund_df[temp_col],
            mode='lines+markers',
            name='Fetsund (målt)',
            line=dict(color='#A23B72', width=2),
            marker=dict(size=4)
        ))
    
    fig.update_layout(
        title="Vanntemperatur - Siste 72 timer",
        xaxis_title="Tid",
        yaxis_title="Temperatur (°C)",
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_wind_chart(weather_df):
    """Create wind speed and direction chart"""
    if weather_df.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Vindhastighet', 'Vindretning'),
        vertical_spacing=0.15
    )
    
    # Wind speed
    fig.add_trace(
        go.Scatter(
            x=weather_df['time'],
            y=weather_df['wind_speed'],
            mode='lines',
            name='Vind',
            line=dict(color='#06A77D', width=2),
            fill='tozeroy',
            fillcolor='rgba(6, 167, 125, 0.2)'
        ),
        row=1, col=1
    )
    
    if 'southerly_wind' in weather_df.columns:
        fig.add_trace(
            go.Scatter(
                x=weather_df['time'],
                y=weather_df['southerly_wind'],
                mode='lines',
                name='Sørlig vind',
                line=dict(color='#D62828', width=2, dash='dash')
            ),
            row=1, col=1
        )
    
    # Critical threshold line
    fig.add_hline(
        y=CRITICAL_WIND_SPEED,
        line_dash="dot",
        line_color="red",
        annotation_text=f"Kritisk terskel ({CRITICAL_WIND_SPEED} m/s)",
        row=1, col=1
    )
    
    # Wind direction
    fig.add_trace(
        go.Scatter(
            x=weather_df['time'],
            y=weather_df['wind_direction'],
            mode='markers',
            name='Retning',
            marker=dict(
                size=6,
                color=weather_df['wind_direction'],
                colorscale='HSV',
                showscale=True,
                colorbar=dict(title="Grader")
            )
        ),
        row=2, col=1
    )
    
    # Southerly band
    fig.add_hrect(
        y0=135, y1=225,
        fillcolor="red",
        opacity=0.1,
        line_width=0,
        annotation_text="Sørlig",
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Tid", row=2, col=1)
    fig.update_yaxes(title_text="m/s", row=1, col=1)
    fig.update_yaxes(title_text="Grader", range=[0, 360], row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.title("🏊‍♂️ Glommadyppen Vanntemperatur")
    st.markdown("**Sanntids temperaturprediksjon for Glommadyppen**")
    
    # Calculate next event date
    event_date = calculate_event_date(EVENT_YEAR)
    days_until = (event_date - pd.Timestamp.now(tz='UTC')).days
    
    # Event info banner
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Neste arrangement", event_date.strftime("%d. %B %Y"))
    with col2:
        st.metric("Dager igjen", f"{days_until} dager")
    with col3:
        st.metric("Starttid", "10:00")
    with col4:
        # Station status indicator
        current_month = datetime.now().month
        if 4 <= current_month <= 9:  # April-September
            st.metric("Stasjon", "🟢 Aktiv")
        else:  # October-March
            st.metric("Stasjon", "🔴 Offline (vinter)")
    
    st.divider()
    
    # Fetch data
    with st.spinner("Laster data..."):
        # NVE data
        vorma_temp = fetch_nve_data(STATION_VORMA, 1003, hours_back=72)
        
        # Weather forecast
        weather_forecast = fetch_weather_forecast(MJOSA_LAT, MJOSA_LON, days_ahead=7)
        if not weather_forecast.empty:
            weather_forecast = calculate_southerly_wind(weather_forecast)
    
    # Check data availability
    if vorma_temp.empty:
        st.warning("""
        ⚠️ **Målestasjon offline (vintersesong)**
        
        Vorma temperaturstasjon (Funnefoss) er for øyeblikket offline. Dette er normalt for vintersesongen 
        (november-mars) når målingene er stengt ned for å unngå isskader.
        
        **Stasjonen vil starte opp igjen i april 2026** - i god tid før Glommadyppen!
        
        For nå kan du:
        - Se værvarsel for Mjøsa (oppdateres kontinuerlig)
        - Utforske historiske data og mønstre
        - Teste systemet med demo-data
        """)
        
        # Show weather forecast anyway
        if not weather_forecast.empty:
            st.subheader("💨 Værvarsel (Mjøsa)")
            st.info("Værvarsling er aktiv! Vinddata oppdateres hver 6. time.")
            
            weather_forecast = calculate_southerly_wind(weather_forecast)
            
            # Wind statistics
            next_48h = weather_forecast.head(48)
            avg_wind = next_48h['wind_speed'].mean()
            max_wind = next_48h['wind_speed'].max()
            avg_southerly = next_48h['southerly_wind'].mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Gj.snitt vind (48t)", f"{avg_wind:.1f} m/s")
            with col2:
                st.metric("Maks vind (48t)", f"{max_wind:.1f} m/s")
            with col3:
                if avg_southerly >= 1.5:
                    st.metric(
                        "Sørlig vind (48t)",
                        f"{avg_southerly:.1f} m/s",
                        delta="⚠️ Ville utløst oppdrift!",
                        delta_color="inverse"
                    )
                else:
                    st.metric("Sørlig vind (48t)", f"{avg_southerly:.1f} m/s")
            
            wind_chart = create_wind_chart(weather_forecast.head(168))
            if wind_chart:
                st.plotly_chart(wind_chart, use_container_width=True)
        
        # Historical context
        st.subheader("📚 Historisk kontekst")
        st.markdown("""
        ### Hvordan systemet fungerer
        
        Når stasjonen er aktiv (april-september), vil appen:
        1. **Hente sanntidsdata** fra Vorma hvert time
        2. **Analysere værforhold** over Mjøsa kontinuerlig
        3. **Beregne prediksjon** for Fetsund (25 timer frem i tid)
        4. **Varsle om kalde hendelser** når sørlig vind utløser oppdrift
        
        ### Viktige datoer
        - **April 2026:** Målestasjon starter opp igjen
        - **1. august 2026:** Glommadyppen (første lørdag i august)
        - **Juli 2026:** Full operativ overvåking starter
        
        ### Kom tilbake i april 2026!
        Da vil hele systemet være aktivt med sanntidsmålinger og prognoser.
        """)
        
        st.stop()
    
    
    # Rename value column
    vorma_temp = vorma_temp.rename(columns={'value': 'temperature'})
    
    # Check data recency
    latest_time = pd.to_datetime(vorma_temp.iloc[-1]['time'])
    if latest_time.tz is None:
        latest_time = latest_time.tz_localize('UTC')
    data_age_days = (pd.Timestamp.now(tz='UTC') - latest_time).total_seconds() / 86400
    
    # If data is very old (>7 days), show warning
    if data_age_days > 7:
        st.warning(f"""
        ⚠️ **Utdaterte måledata**
        
        Siste måling fra Vorma er **{data_age_days:.1f} dager gammel** 
        (målt {latest_time.strftime('%d.%m.%Y kl. %H:%M')}).
        
        Dette indikerer at stasjonen kan være offline for vintersesong.
        
        - **Værvarsling fungerer:** Vinddata fra Mjøsa oppdateres fortsatt
        - **Stasjon kommer tilbake:** Forventes aktiv igjen i april 2026
        - **Historiske data:** Vises nedenfor for referanse
        """)
        st.divider()
    
    # Main prediction section
    st.header("📊 Temperaturprediksjon")
    
    # Current status
    latest_vorma = vorma_temp.iloc[-1]
    latest_time = pd.to_datetime(latest_vorma['time'])
    if latest_time.tz is None:
        latest_time = latest_time.tz_localize('UTC')
    data_age_hours = (pd.Timestamp.now(tz='UTC') - latest_time).total_seconds() / 3600
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Calculate 24-hour change only if enough data available
        if len(vorma_temp) >= 24:
            temp_24h_ago = vorma_temp.iloc[-24]['temperature']
            delta_24h = f"{latest_vorma['temperature'] - temp_24h_ago:.1f}°C (24t)"
        else:
            delta_24h = "Ikke nok data"
        
        st.metric(
            "Vorma nå",
            f"{latest_vorma['temperature']:.1f}°C",
            delta=delta_24h
        )
    
    with col2:
        # Check for recent drops
        drop_event = detect_temperature_drop(
            vorma_temp.rename(columns={'temperature': 'value'}),
            threshold_C=2.0,
            window_hours=6
        )
        if drop_event:
            st.metric(
                "Temperaturfall (6t)",
                f"{drop_event['magnitude']:.1f}°C",
                delta=f"⚠️ Detektert!",
                delta_color="inverse"
            )
        else:
            st.metric("Temperaturfall (6t)", "Ingen", delta="✓ Stabilt")
    
    with col3:
        if not weather_forecast.empty:
            current_wind = weather_forecast.iloc[0]
            st.metric(
                "Vind (Mjøsa)",
                f"{current_wind['wind_speed']:.1f} m/s",
                delta=f"{current_wind['wind_direction']:.0f}°"
            )
        else:
            st.metric("Vind (Mjøsa)", "Ikke tilgjengelig")
    
    with col4:
        if data_age_hours < 2:
            freshness = "✓ Fersk"
            color = "normal"
        elif data_age_hours < 6:
            freshness = "⚠️ Noen timer gammel"
            color = "inverse"
        else:
            freshness = "❌ Utdatert"
            color = "inverse"
        
        st.metric(
            "Datastatus",
            f"{data_age_hours:.1f}t siden",
            delta=freshness,
            delta_color=color
        )
    
    st.divider()
    
    # Prediction for event
    prediction = predict_fetsund_temperature(
        vorma_temp.rename(columns={'temperature': 'value'}),
        event_date
    )
    
    # Check if data is too old for meaningful prediction
    days_until_event = (event_date - pd.Timestamp.now(tz='UTC')).days
    
    if data_age_days > 30 and days_until_event > 30:
        st.info(f"""
        📊 **Prediksjon ikke tilgjengelig**
        
        Sanntids-prediksjon krever ferske målinger fra Vorma. Siden siste måling er 
        {data_age_days:.0f} dager gammel, kan vi ikke lage en pålitelig prognose ennå.
        
        **Prediksjonen vil være tilgjengelig når:**
        - Målestasjon starter opp igjen (april 2026)
        - Vi kommer nærmere arrangementsdatoen
        
        **Basert på historiske data:**
        - Gjennomsnittlig temperatur i Fetsund tidlig august: 16-18°C
        - Risiko for kaldt vann hvis vedvarende sørlig vind over Mjøsa
        - 25 timers reisevarsel fra Vorma til Fetsund
        """)
    elif prediction:
        risk_level, risk_color = assess_risk_level(prediction, weather_forecast)
        
        # Big prediction display
        st.markdown(f"""
        <div style='background-color: {risk_color}; padding: 20px; border-radius: 10px; color: white; text-align: center;'>
            <h2 style='margin: 0; color: white;'>Predikert temperatur ved Fetsund</h2>
            <h1 style='margin: 10px 0; font-size: 3em; color: white;'>{prediction['predicted_temp']:.1f}°C</h1>
            <h3 style='margin: 0; color: white;'>{risk_level}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Prediction details
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Vorma temp (25t før)",
                f"{prediction['vorma_temp']:.1f}°C",
                help="Temperatur i Vorma 25 timer før arrangementet"
            )
        
        with col2:
            st.metric(
                "Baseline temp",
                f"{prediction['baseline_temp']:.1f}°C",
                help="Gjennomsnittlig temperatur siste 48 timer"
            )
        
        with col3:
            confidence_pct = prediction['confidence'] * 100
            st.metric(
                "Pålitelighet",
                f"{confidence_pct:.0f}%",
                help="Basert på datakvalitet og aktualitet"
            )
        
        # Confidence interval
        std_error = 2.0  # Standard error from model validation
        margin = std_error * 1.96  # 95% confidence
        
        st.info(f"""
        **95% konfidensintervall:** {prediction['predicted_temp'] - margin:.1f}°C - {prediction['predicted_temp'] + margin:.1f}°C
        
        Denne prediksjonen er basert på:
        - 🕐 25 timers reisetid fra Vorma til Fetsund
        - 📉 14% overlevelsesrate av temperaturendringer (grunnet fortynning)
        - 📊 Historiske data fra {len(vorma_temp)} målinger
        """)
    else:
        st.warning("⚠️ Ikke nok data for å beregne prediksjon for arrangementsdatoen.")
    
    st.divider()
    
    # Temperature chart
    st.subheader("📈 Temperaturhistorikk")
    temp_chart = create_temperature_chart(vorma_temp)
    st.plotly_chart(temp_chart, use_container_width=True)
    
    # Wind analysis
    if not weather_forecast.empty:
        st.subheader("💨 Vindanalyse (Mjøsa)")
        
        # Wind statistics
        next_48h = weather_forecast.head(48)
        avg_wind = next_48h['wind_speed'].mean()
        max_wind = next_48h['wind_speed'].max()
        avg_southerly = next_48h['southerly_wind'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Gj.snitt vind (48t)", f"{avg_wind:.1f} m/s")
        with col2:
            st.metric("Maks vind (48t)", f"{max_wind:.1f} m/s")
        with col3:
            if avg_southerly >= 1.5:
                st.metric(
                    "Sørlig vind (48t)",
                    f"{avg_southerly:.1f} m/s",
                    delta="⚠️ Oppdriftsrisiko!",
                    delta_color="inverse"
                )
            else:
                st.metric("Sørlig vind (48t)", f"{avg_southerly:.1f} m/s")
        
        wind_chart = create_wind_chart(weather_forecast.head(168))  # 7 days
        if wind_chart:
            st.plotly_chart(wind_chart, use_container_width=True)
        
        # Wind warning
        if avg_southerly >= CRITICAL_WIND_SPEED:
            st.error(f"""
            🌊 **OPPDRIFTSVARSEL!**
            
            Vedvarende sørlig vind over {CRITICAL_WIND_SPEED} m/s er varslet.
            Dette kan utløse oppdrift av kaldt dypvann fra Mjøsa.
            
            Forventer temperaturfall i Vorma innen 24-48 timer.
            """)
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ Om systemet")
        
        st.markdown("""
        ### Hvordan det virker
        
        1. **Datainnsamling**
           - Sanntidsdata fra NVE (Vorma)
           - Værvarsling fra Met.no (Mjøsa)
        
        2. **Prediksjonsmodell**
           - 25 timers reisetid fra Vorma til Fetsund
           - 14% overlevelse av temperaturendring
           - Basert på forskning 2015-2025
        
        3. **Risikofaktorer**
           - Sørlig vind over Mjøsa
           - Temperaturfall i Vorma
           - Historiske mønstre
        
        ### Datakilder
        - **NVE HydAPI:** Vanntemperatur og vannføring
        - **Met.no:** Værvarsling
        
        ### Sist oppdatert
        {pd.Timestamp.now(tz='Europe/Oslo').strftime("%d.%m.%Y %H:%M")} (Oslo tid)
        """)
        
        st.markdown("---")
        st.caption("Utviklet av Anton | Glommadyppen 2026")
        
        if st.button("🔄 Oppdater data"):
            st.cache_data.clear()
            st.rerun()

if __name__ == "__main__":
    main()
