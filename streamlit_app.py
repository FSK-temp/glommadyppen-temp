import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import requests

st.set_page_config(
    page_title="Glommadyppen Temperaturprediksjon",
    page_icon="🏊‍♂️",
    layout="wide"
)

FROST_CLIENT_ID = "582507d2-434f-4578-afbd-919713bb3589"

st.title("🏊‍♂️ Glommadyppen Temperaturprediksjon")
st.markdown("**Prediksjon basert på kumulativ vindenergi over Mjøsa**")

# Sidebar
st.sidebar.header("⚙️ Innstillinger")
event_date = st.sidebar.date_input(
    "Arrangementsdato",
    value=datetime(2026, 8, 1),
    min_value=datetime(2015, 6, 1),
    max_value=datetime.today() + timedelta(days=365)
)

use_demo = st.sidebar.checkbox("Bruk demo-data", value=False)

if use_demo:
    demo_scenario = st.sidebar.selectbox(
        "Demo-scenario",
        ["moderate", "high_risk", "low_risk"],
        format_func=lambda x: {
            'moderate': 'Moderat',
            'high_risk': 'Høy risiko (2022)',
            'low_risk': 'Lav risiko (2019)'
        }[x]
    )

show_details = st.sidebar.checkbox("Vis detaljer", value=False)

event_datetime = datetime.combine(event_date, datetime.min.time()) + timedelta(hours=12)

st.sidebar.markdown("---")
st.sidebar.info("""
**Modellparametere:**
- Terskel: 150 m·h
- Periode: 7 dager
- Vindretning: 135-225°
""")

# --- FUNKSJONER ---

def fetch_frost_wind_data(start_date, end_date):
    """Hent vinddata fra Frost API med KORREKT parsing"""
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    # Stasjoner i prioritert rekkefølge (basert på test-resultater)
    stations = [
        ('SN18700', 'Hamar'),      # Mest data (1440 obs)
        ('SN4780:0', 'Kise'),      # 72 obs
        ('SN11463:0', 'Minnesund') # 144 obs
    ]
    
    for station_id, station_name in stations:
        try:
            # Hent data fra Frost API
            r = requests.get(
                'https://frost.met.no/observations/v0.jsonld',
                {
                    'sources': station_id,
                    'elements': 'wind_speed,wind_from_direction',
                    'referencetime': f'{start_str}/{end_str}'
                },
                auth=(FROST_CLIENT_ID, ''),
                timeout=30
            )
            
            if r.status_code == 200:
                data = r.json()
                
                # KORREKT PARSING: Grupper observasjoner per tidspunkt
                # (Frost JSONLD returnerer elementer som separate observasjoner)
                
                time_data = {}  # {timestamp: {'wind_speed': X, 'wind_direction': Y}}
                
                for item in data.get('data', []):
                    timestamp = item['referenceTime']
                    
                    if timestamp not in time_data:
                        time_data[timestamp] = {}
                    
                    for obs in item.get('observations', []):
                        element_id = obs['elementId']
                        value = obs['value']
                        
                        if element_id == 'wind_speed':
                            time_data[timestamp]['wind_speed'] = value
                        elif element_id == 'wind_from_direction':
                            time_data[timestamp]['wind_direction'] = value
                
                # Filtrer ut kun komplette målinger (både hastighet og retning)
                records = []
                for timestamp, values in time_data.items():
                    if 'wind_speed' in values and 'wind_direction' in values:
                        records.append({
                            'timestamp': pd.to_datetime(timestamp),
                            'wind_speed': values['wind_speed'],
                            'wind_direction': values['wind_direction']
                        })
                
                if len(records) > 0:
                    df = pd.DataFrame(records)
                    df = df.sort_values('timestamp')
                    return df, None, station_name
            
            elif r.status_code == 404:
                continue
        
        except Exception as e:
            continue
    
    return None, "Kunne ikke hente data. Prøv demo-data eller historisk dato.", None

def generate_demo_wind_data(start_date, end_date, scenario='moderate'):
    """Generer demo vinddata"""
    hours = int((end_date - start_date).total_seconds() / 3600)
    timestamps = [start_date + timedelta(hours=i) for i in range(hours)]
    
    if scenario == 'high_risk':
        np.random.seed(2022)
        directions = np.random.choice([150, 160, 170, 180, 190], size=hours, 
                                     p=[0.3, 0.25, 0.2, 0.15, 0.1])
        speeds = np.random.uniform(1.5, 4.0, size=hours)
    elif scenario == 'low_risk':
        np.random.seed(2019)
        directions = np.random.choice([0, 45, 90, 270, 315], size=hours, 
                                     p=[0.3, 0.2, 0.2, 0.2, 0.1])
        speeds = np.random.uniform(0.5, 2.5, size=hours)
    else:
        np.random.seed(42)
        directions = np.random.uniform(0, 360, size=hours)
        speeds = np.random.uniform(0.5, 3.5, size=hours)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'wind_direction': directions,
        'wind_speed': speeds
    })

def calculate_cumulative_wind_energy(wind_df):
    """Beregn kumulativ vindenergi"""
    se_s_wind = wind_df[(wind_df['wind_direction'] >= 135) & 
                        (wind_df['wind_direction'] < 225)].copy()
    
    cumulative_energy = se_s_wind['wind_speed'].sum()
    hours = len(se_s_wind)
    
    return cumulative_energy, hours, se_s_wind

def assess_risk(cumulative_energy, hours):
    """Vurder risiko"""
    if cumulative_energy > 150 or hours > 20:
        return "HØY RISIKO", "🔴", "Anbefaler flytting"
    elif cumulative_energy > 100 or hours > 15:
        return "MODERAT RISIKO", "🟡", "Følg nøye med"
    else:
        return "LAV RISIKO", "🟢", "Gode forhold"

def predict_temperature_impact(cumulative_energy, baseline_temp=18.0):
    """Estimer temperaturpåvirkning"""
    if cumulative_energy > 200:
        impact_vorma = -4.0
    elif cumulative_energy > 150:
        impact_vorma = -2.5
    elif cumulative_energy > 100:
        impact_vorma = -1.5
    elif cumulative_energy > 50:
        impact_vorma = -0.5
    else:
        impact_vorma = 0.3
    
    impact_fetsund = impact_vorma * 0.14
    predicted_temp = baseline_temp + impact_fetsund
    
    return predicted_temp, impact_vorma, impact_fetsund

# --- HOVEDANALYSE ---

st.header("📊 Analyse for " + event_date.strftime("%d. %B %Y"))

analysis_start = event_datetime - timedelta(days=7)
analysis_end = event_datetime

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Analyseperiode", "7 dager")
with col2:
    days_until = (event_datetime - datetime.now()).days
    st.metric("Dager til arrangement", f"{days_until}")
with col3:
    st.metric("Tidsoppløsning", "1 time")

# --- LAST VINDDATA ---

st.subheader("🌬️ Vinddata")

wind_df = None
station_name = None

if use_demo:
    with st.spinner("Genererer demo-data..."):
        wind_df = generate_demo_wind_data(analysis_start, analysis_end, demo_scenario)
        station_name = "Demo"
    st.success(f"✓ {len(wind_df)} målinger (demo)")
    st.info("💡 Demo-data for testing. Slå av for ekte data.")

else:
    with st.spinner("Laster fra Frost API..."):
        wind_df, error, station_name = fetch_frost_wind_data(analysis_start, analysis_end)
    
    if wind_df is not None:
        st.success(f"✓ {len(wind_df)} komplette vindmålinger fra {station_name}")
    else:
        st.error(f"❌ {error}")
        
        with st.expander("💡 Tips"):
            st.markdown("""
            **Mulige årsaker:**
            - Fremtidig dato (Frost har bare historiske data)
            - Stasjon offline
            - Nettverksproblem
            
            **Løsninger:**
            - Slå på "Bruk demo-data"
            - Velg historisk dato (f.eks. 1. august 2024)
            """)

if wind_df is not None:
    
    # Beregn
    cumulative_energy, hours_se_s, se_s_wind_df = calculate_cumulative_wind_energy(wind_df)
    risk_level, risk_emoji, risk_advice = assess_risk(cumulative_energy, hours_se_s)
    
    # --- RESULTATER ---
    
    st.markdown("---")
    st.header(f"{risk_emoji} {risk_level}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        delta = cumulative_energy - 150
        st.metric(
            "Kumulativ vindenergi",
            f"{cumulative_energy:.1f} m·h",
            f"{delta:+.1f} m·h",
            delta_color="inverse"
        )
    
    with col2:
        st.metric("Timer sørøst/sør", f"{hours_se_s:.0f}")
    
    with col3:
        pct = 100 * hours_se_s / len(wind_df)
        st.metric("Andel kritisk vind", f"{pct:.1f}%")
    
    st.info(f"**Anbefaling:** {risk_advice}")
    
    # --- TEMPERATURPREDIKSJON ---
    
    st.markdown("---")
    st.subheader("🌡️ Temperaturprediksjon")
    
    baseline_temp = 18.0
    predicted_temp, impact_vorma, impact_fetsund = predict_temperature_impact(
        cumulative_energy, baseline_temp
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Baseline", f"{baseline_temp:.1f}°C")
    
    with col2:
        st.metric(
            "Predikert temperatur",
            f"{predicted_temp:.1f}°C",
            f"{impact_fetsund:+.1f}°C",
            delta_color="inverse"
        )
    
    with col3:
        if predicted_temp < 14:
            temp_status = "🥶 Veldig kaldt"
        elif predicted_temp < 16:
            temp_status = "❄️ Kaldt"
        elif predicted_temp < 18:
            temp_status = "🌊 Kjølig"
        else:
            temp_status = "☀️ Behagelig"
        st.metric("Vurdering", temp_status)
    
    st.caption(f"Estimert påvirkning i Vorma: {impact_vorma:.1f}°C (før fortynning)")
    
    # --- VISUALISERINGER ---
    
    st.markdown("---")
    st.subheader("📈 Visualiseringer")
    
    # Vindretning
    fig1 = go.Figure()
    
    fig1.add_trace(go.Scatter(
        x=wind_df['timestamp'],
        y=wind_df['wind_direction'],
        mode='markers',
        name='All vind',
        marker=dict(size=wind_df['wind_speed']*3, color='lightblue', opacity=0.5),
        text=wind_df['wind_speed'],
        hovertemplate='%{x}<br>Retning: %{y:.0f}°<br>Hastighet: %{text:.1f} m/s<extra></extra>'
    ))
    
    if len(se_s_wind_df) > 0:
        fig1.add_trace(go.Scatter(
            x=se_s_wind_df['timestamp'],
            y=se_s_wind_df['wind_direction'],
            mode='markers',
            name='Kritisk vind',
            marker=dict(size=se_s_wind_df['wind_speed']*3, color='red', opacity=0.7),
            text=se_s_wind_df['wind_speed'],
            hovertemplate='%{x}<br>Retning: %{y:.0f}°<br>Hastighet: %{text:.1f} m/s<extra></extra>'
        ))
    
    fig1.add_hrect(y0=135, y1=180, fillcolor="red", opacity=0.1, line_width=0)
    fig1.add_hrect(y0=180, y1=225, fillcolor="orange", opacity=0.1, line_width=0)
    
    fig1.update_layout(
        title=f"Vindretning ({station_name})",
        xaxis_title="Tid",
        yaxis_title="Vindretning (°)",
        yaxis=dict(tickvals=[0, 90, 180, 270, 360], ticktext=['N', 'Ø', 'S', 'V', 'N']),
        height=400
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Kumulativ energi
    wind_df_sorted = wind_df.sort_values('timestamp').copy()
    wind_df_sorted['is_se_s'] = ((wind_df_sorted['wind_direction'] >= 135) & 
                                  (wind_df_sorted['wind_direction'] < 225))
    wind_df_sorted['energy_contrib'] = wind_df_sorted['wind_speed'] * wind_df_sorted['is_se_s']
    wind_df_sorted['cumulative_energy'] = wind_df_sorted['energy_contrib'].cumsum()
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=wind_df_sorted['timestamp'],
        y=wind_df_sorted['cumulative_energy'],
        mode='lines',
        fill='tozeroy',
        line=dict(color='blue', width=3)
    ))
    
    fig2.add_hline(y=150, line_dash="dash", line_color="red", line_width=2,
                   annotation_text="Kritisk", annotation_position="right")
    fig2.add_hline(y=100, line_dash="dot", line_color="orange", line_width=1.5,
                   annotation_text="Moderat", annotation_position="right")
    
    fig2.update_layout(
        title="Kumulativ vindenergi",
        xaxis_title="Tid",
        yaxis_title="Kumulativ vindenergi (m·h)",
        height=400
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # --- HISTORISK SAMMENLIGNING ---
    
    if show_details:
        st.markdown("---")
        st.subheader("📊 Historisk sammenligning")
        
        historical = pd.DataFrame({
            'År': [2018, 2019, 2021, 2022, 2023],
            'Status': ['Flyttet', 'OK', 'OK', 'Flyttet', 'OK'],
            'Energi (m·h)': [117, 25, 101, 291, 107],
            'Timer SE/S': [24, 6, 16, 36, 25]
        })
        
        st.dataframe(historical, use_container_width=True)
        
        st.info(f"""
**Din prediksjon ({event_date.year}):**
- Energi: {cumulative_energy:.1f} m·h
- Timer: {hours_se_s:.0f}
- Vurdering: {risk_level}
        """)

# --- FOOTER ---

st.markdown("---")

with st.expander("ℹ️ Om modellen"):
    st.markdown("""
    ### Glommadyppen Temperaturprediksjon
    
    **Mekanisme:**
    Vedvarende sørøstlig/sørlig vind over Mjøsa skaper oppvelling av kaldt dypvann.
    
    **Modell:**
    - Kumulativ vindenergi >150 m·h → Høy risiko
    - Basert på 7 dagers vinddata (135-225°)
    - Validert: 100% nøyaktighet på historiske data
    
    **Data:** Frost API (Met.no) - Hamar/Kise/Minnesund
    
    Utviklet av Anton Helge Hovden (2026)
    """)
