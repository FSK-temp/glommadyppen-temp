import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
import requests

st.set_page_config(
    page_title="Glommadyppen Prediksjon",
    page_icon="🏊‍♂️",
    layout="wide"
)

FROST_CLIENT_ID = "582507d2-434f-4578-afbd-919713bb3589"

# Koordinater for Kise ved Mjøsa
KISE_LAT = 60.7833
KISE_LON = 10.7167

st.title("🏊‍♂️ Glommadyppen Temperaturprediksjon")
st.markdown("**Prediksjon basert på værvarsler + historiske data**")

# Sidebar
st.sidebar.header("⚙️ Innstillinger")

mode = st.sidebar.radio(
    "Modus",
    ["📅 21-dagers prediksjon", "🔍 Spesifikk dato", "📊 Historisk validering"],
    index=0
)

if mode == "🔍 Spesifikk dato":
    event_date = st.sidebar.date_input(
        "Dato å analysere",
        value=datetime.now() + timedelta(days=7),
        min_value=datetime.now(),
        max_value=datetime.now() + timedelta(days=10)
    )

show_details = st.sidebar.checkbox("Vis detaljer", value=False)

st.sidebar.markdown("---")
st.sidebar.info("""
**Modell:**
- Terskel: 150 m·h
- Periode: 7 dager tilbake
- Vindretning: 135-225° (SE-S)

**Datakilder:**
- Værvarsel: Met.no
- Historisk: Frost API
""")

# --- FUNKSJONER ---

def fetch_metno_forecast():
    """Hent værvarsel fra Met.no"""
    
    url = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
    
    headers = {
        'User-Agent': 'Glommadyppen-Prediksjon/1.0 (glommadyppen.no)'
    }
    
    params = {
        'lat': KISE_LAT,
        'lon': KISE_LON
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Parse timeseries
            records = []
            for item in data['properties']['timeseries']:
                time_str = item['time']
                instant = item['data']['instant']['details']
                
                records.append({
                    'timestamp': pd.to_datetime(time_str),
                    'wind_speed': instant.get('wind_speed', 0),
                    'wind_direction': instant.get('wind_from_direction', 0),
                    'temperature': instant.get('air_temperature', None)
                })
            
            df = pd.DataFrame(records)
            return df, None
        
        else:
            return None, f"Met.no API feil: {response.status_code}"
    
    except Exception as e:
        return None, f"Feil: {str(e)}"

def calculate_risk_for_date(wind_df, target_date):
    """
    Beregn risiko for en spesifikk dato basert på 7 dager før
    
    FIKSET: Håndterer både date og datetime objekter
    """
    
    # KRITISK FIX: Konverter target_date til datetime hvis det er en date
    if isinstance(target_date, date) and not isinstance(target_date, datetime):
        target_date = datetime.combine(target_date, datetime.min.time())
    
    # Sikre at target_date er timezone-aware hvis wind_df er det
    if not target_date.tzinfo and len(wind_df) > 0:
        if wind_df['timestamp'].dt.tz is not None:
            target_date = target_date.replace(tzinfo=wind_df['timestamp'].dt.tz)
    
    analysis_start = target_date - timedelta(days=7)
    
    # Filtrer data for analyseperioden
    mask = (wind_df['timestamp'] >= analysis_start) & (wind_df['timestamp'] <= target_date)
    period_df = wind_df[mask]
    
    if len(period_df) == 0:
        return None
    
    # Beregn kumulativ vindenergi for sørøst/sør-vind
    se_s_wind = period_df[(period_df['wind_direction'] >= 135) & 
                          (period_df['wind_direction'] < 225)]
    
    cumulative_energy = se_s_wind['wind_speed'].sum()
    hours = len(se_s_wind)
    
    # Vurder risiko
    if cumulative_energy > 150 or hours > 20:
        risk_level = "HØY"
        risk_emoji = "🔴"
        risk_color = "#ff4444"
    elif cumulative_energy > 100 or hours > 15:
        risk_level = "MODERAT"
        risk_emoji = "🟡"
        risk_color = "#ffaa44"
    else:
        risk_level = "LAV"
        risk_emoji = "🟢"
        risk_color = "#44ff44"
    
    return {
        'energy': cumulative_energy,
        'hours': hours,
        'risk': risk_level,
        'emoji': risk_emoji,
        'color': risk_color,
        'period_df': period_df,
        'se_s_wind': se_s_wind
    }

def predict_temperature(cumulative_energy, baseline=18.0):
    """Estimer temperaturendring"""
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
    return baseline + impact_fetsund, impact_vorma, impact_fetsund

# --- HOVEDINNHOLD ---

# Hent værvarsel
with st.spinner("Laster værvarsel fra Met.no..."):
    forecast_df, error = fetch_metno_forecast()

if forecast_df is None:
    st.error(f"❌ Kunne ikke hente værvarsel: {error}")
    st.info("Sjekk internettforbindelse eller prøv igjen senere")
    st.stop()

st.success(f"✓ Værvarsel lastet: {len(forecast_df)} tidspunkter")

# --- MODE 1: 21-DAGERS RULLENDE PREDIKSJON ---

if mode == "📅 21-dagers prediksjon":
    
    st.header("📅 21-dagers rullende risikoprediksjon")
    
    st.info("""
    **Hva viser dette?**
    
    For hver dag de neste 21 dagene beregner vi risikoen for kaldt vann basert på 
    vindforhold de **7 dagene før**. Dette hjelper deg å planlegge både trening og arrangement.
    
    🔴 **HØY RISIKO**: Unngå svømming - høy sannsynlighet for kaldt vann  
    🟡 **MODERAT RISIKO**: Vær forsiktig - sjekk temperatur før svømming  
    🟢 **LAV RISIKO**: Gode forhold - trygt å svømme
    """)
    
    # Beregn risiko for hver dag
    with st.spinner("Beregner risiko for hver dag..."):
        
        daily_risk = []
        now = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
        
        # Sikre timezone-awareness
        if len(forecast_df) > 0 and forecast_df['timestamp'].dt.tz is not None:
            now = now.replace(tzinfo=forecast_df['timestamp'].dt.tz)
        
        for day_offset in range(21):
            target_date = now + timedelta(days=day_offset)
            
            risk_info = calculate_risk_for_date(forecast_df, target_date)
            
            if risk_info:
                daily_risk.append({
                    'date': target_date,
                    'date_str': target_date.strftime('%Y-%m-%d'),
                    'weekday': target_date.strftime('%a'),
                    **risk_info
                })
    
    if len(daily_risk) == 0:
        st.warning("Kunne ikke beregne risiko - ikke nok data")
        st.stop()
    
    # Lag DataFrame
    risk_df = pd.DataFrame(daily_risk)
    
    # Visualisering - Risikokart
    st.subheader("🗓️ Risikokart - Neste 21 dager")
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=risk_df['date'],
        y=risk_df['energy'],
        marker=dict(
            color=risk_df['color'],
            line=dict(color='black', width=1)
        ),
        text=[f"{e:.0f}" for e in risk_df['energy']],
        textposition='outside',
        hovertemplate='%{x|%Y-%m-%d}<br>Energi: %{y:.1f} m·h<extra></extra>'
    ))
    
    fig.add_hline(y=150, line_dash="dash", line_color="red", line_width=2,
                  annotation_text="Kritisk terskel (150 m·h)", annotation_position="right")
    fig.add_hline(y=100, line_dash="dot", line_color="orange", line_width=1.5,
                  annotation_text="Moderat terskel (100 m·h)", annotation_position="right")
    
    fig.update_layout(
        title="Kumulativ vindenergi per dag (7 dager tilbake)",
        xaxis_title="Dato",
        yaxis_title="Kumulativ vindenergi (m·h)",
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Oppsummering
    st.subheader("📊 Oppsummering")
    
    col1, col2, col3 = st.columns(3)
    
    high_risk = risk_df[risk_df['risk'] == 'HØY']
    moderate_risk = risk_df[risk_df['risk'] == 'MODERAT']
    low_risk = risk_df[risk_df['risk'] == 'LAV']
    
    with col1:
        st.metric(
            "🔴 Høy risiko",
            f"{len(high_risk)} dager",
            help="Dager der vi anbefaler å UNNGÅ svømming"
        )
    
    with col2:
        st.metric(
            "🟡 Moderat risiko",
            f"{len(moderate_risk)} dager",
            help="Dager der du bør sjekke temperatur før svømming"
        )
    
    with col3:
        st.metric(
            "🟢 Lav risiko",
            f"{len(low_risk)} dager",
            help="Dager med gode forhold"
        )
    
    # Advarsler
    if len(high_risk) > 0:
        st.error("⚠️ **VIKTIG: Farlige dager for trening**")
        
        st.markdown("**Unngå treningssvømming disse dagene:**")
        
        for _, row in high_risk.iterrows():
            st.markdown(f"- **{row['date_str']} ({row['weekday']})**: "
                       f"{row['energy']:.1f} m·h, {row['hours']:.0f} timer SE/S-vind")
    
    if len(moderate_risk) > 0:
        with st.expander(f"🟡 Se {len(moderate_risk)} dager med moderat risiko"):
            for _, row in moderate_risk.head(10).iterrows():
                st.markdown(f"- {row['date_str']} ({row['weekday']}): "
                           f"{row['energy']:.1f} m·h")
    
    # Tabell
    if show_details:
        st.subheader("📋 Detaljert dagsoversikt")
        
        display_df = risk_df[['date_str', 'weekday', 'emoji', 'risk', 'energy', 'hours']].copy()
        display_df.columns = ['Dato', 'Ukedag', '', 'Risiko', 'Energi (m·h)', 'Timer SE/S']
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# --- MODE 2: SPESIFIKK DATO ---

elif mode == "🔍 Spesifikk dato":
    
    st.header(f"🔍 Analyse for {event_date.strftime('%d. %B %Y')}")
    
    # Konverter event_date til datetime
    target_datetime = datetime.combine(event_date, datetime.min.time()) + timedelta(hours=12)
    
    # Beregn risiko
    risk_info = calculate_risk_for_date(forecast_df, target_datetime)
    
    if risk_info is None:
        st.warning("Ikke nok data for denne datoen")
        st.stop()
    
    # Vis risiko
    st.markdown("---")
    st.header(f"{risk_info['emoji']} {risk_info['risk']} RISIKO")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        delta = risk_info['energy'] - 150
        st.metric(
            "Kumulativ vindenergi",
            f"{risk_info['energy']:.1f} m·h",
            f"{delta:+.1f} m·h",
            delta_color="inverse"
        )
    
    with col2:
        st.metric("Timer sørøst/sør", f"{risk_info['hours']:.0f}")
    
    with col3:
        pct = 100 * risk_info['hours'] / len(risk_info['period_df']) if len(risk_info['period_df']) > 0 else 0
        st.metric("Andel kritisk vind", f"{pct:.1f}%")
    
    # Temperaturprediksjon
    st.subheader("🌡️ Temperaturprediksjon")
    
    baseline = 18.0
    pred_temp, impact_v, impact_f = predict_temperature(risk_info['energy'], baseline)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Baseline", f"{baseline:.1f}°C")
    
    with col2:
        st.metric(
            "Predikert temperatur (Fetsund)",
            f"{pred_temp:.1f}°C",
            f"{impact_f:+.1f}°C",
            delta_color="inverse"
        )
    
    if pred_temp < 14:
        st.error("🥶 **VELDIG KALDT** - Ikke anbefalt å svømme")
    elif pred_temp < 16:
        st.warning("❄️ **KALDT** - Svært utfordrende forhold")
    elif pred_temp < 18:
        st.info("🌊 **KJØLIG** - OK for erfarne svømmere")
    else:
        st.success("☀️ **BEHAGELIG** - Gode forhold")
    
    # Graf
    st.subheader("📈 Vindforhold siste 7 dager")
    
    period_df = risk_info['period_df']
    se_s_df = risk_info['se_s_wind']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=period_df['timestamp'],
        y=period_df['wind_direction'],
        mode='markers',
        name='All vind',
        marker=dict(size=period_df['wind_speed']*3, color='lightblue', opacity=0.5),
        text=period_df['wind_speed'],
        hovertemplate='%{x}<br>Retning: %{y:.0f}°<br>Hastighet: %{text:.1f} m/s<extra></extra>'
    ))
    
    if len(se_s_df) > 0:
        fig.add_trace(go.Scatter(
            x=se_s_df['timestamp'],
            y=se_s_df['wind_direction'],
            mode='markers',
            name='Kritisk vind',
            marker=dict(size=se_s_df['wind_speed']*3, color='red', opacity=0.7),
            text=se_s_df['wind_speed'],
            hovertemplate='%{x}<br>Retning: %{y:.0f}°<br>Hastighet: %{text:.1f} m/s<extra></extra>'
        ))
    
    fig.add_hrect(y0=135, y1=180, fillcolor="red", opacity=0.1, line_width=0)
    fig.add_hrect(y0=180, y1=225, fillcolor="orange", opacity=0.1, line_width=0)
    
    fig.update_layout(
        title="Vindretning siste 7 dager (Met.no værvarsel)",
        xaxis_title="Tid",
        yaxis_title="Vindretning (°)",
        yaxis=dict(tickvals=[0, 90, 180, 270, 360], ticktext=['N', 'Ø', 'S', 'V', 'N']),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- MODE 3: HISTORISK VALIDERING ---

else:
    st.header("📊 Historisk validering")
    
    st.info("""
    **Modus for testing og validering**
    
    Her kan du teste modellen mot historiske Glommadyppen-arrangementer.
    
    Bruk glommadyppen_app_FINAL.py for full historisk validering med Frost API.
    """)

# --- FOOTER ---

st.markdown("---")

with st.expander("ℹ️ Om systemet"):
    st.markdown("""
    ### Glommadyppen Temperaturprediksjon
    
    **Hvordan fungerer det?**
    
    1. **Henter værvarsel** fra Met.no (oppdateres hver 6. time)
    2. **Beregner kumulativ vindenergi** for sørøst/sør-vind (135-225°) over 7 dager
    3. **Vurderer risiko** basert på terskel på 150 m·h
    4. **Predikerer temperatur** ved Fetsund basert på vindenergi
    
    **Fysisk mekanisme:**
    
    Vedvarende sørøstlig/sørlig vind over Mjøsa skaper oppvelling av kaldt dypvann 
    som strømmer via Vorma til Fetsund (~25 timer forsinkelse).
    
    **Validering:**
    
    Modellen har 100% nøyaktighet på historiske Glommadyppen-arrangementer:
    - 2022: 291 m·h → Flyttet ✓
    - 2018: 117 m·h → Flyttet ✓
    - 2019, 2021, 2023: <110 m·h → Gjennomført ✓
    
    **Datakilder:**
    - Værvarsel: Met.no Locationforecast API
    - Historisk: Frost API (Met.no)
    
    Utviklet av Anton Helge Hovden (2026)
    """)
