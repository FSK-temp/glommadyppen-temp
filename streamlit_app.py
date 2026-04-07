"""
Glommadyppen Vanntemperatur Prediksjon
Real-time water temperature prediction for Glommadyppen swimming event

Author: Anton Vooren
Date: 2026
"""

import streamlit as st
from PIL import Image
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Glommadyppen Temperatur",
    page_icon="🏊‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

try:
    NVE_API_KEY = st.secrets["nve_api_key"]
except (KeyError, FileNotFoundError):
    NVE_API_KEY = None

NVE_BASE_URL    = "https://hydapi.nve.no/api/v1"
FROST_CLIENT_ID = "582507d2-434f-4578-afbd-919713bb3589"
FROST_BASE_URL  = "https://frost.met.no"

# ── NVE station IDs ──────────────────────────────────────────────────────────
STATION_SVANEFOSS        = "2.52.0"    # Vorma – temperatur (upstream reference)
STATION_FUNNEFOSS_TEMP   = "2.410.0"   # Vorma / Funnefoss – temperatur
STATION_ERTESEKKEN_Q     = "2.197.0"   # Vorma / Ertesekken – vannføring
STATION_BLAKER           = "2.17.0"    # Glomma / Blaker – temperatur + vannføring
STATION_FUNNEFOSS_Q      = "2.279.0"   # Glomma / Funnefoss nedre – vannføring
STATION_FETSUND          = "2.587.0"   # Fetsund bru – temperatur (målpunkt)

# ── Frost (met.no observations) ──────────────────────────────────────────────
FROST_STATION_KISE = "SN12680"   # Kise, søndre Mjøsa

# ── Met.no koordinater ───────────────────────────────────────────────────────
MJOSA_LAT,       MJOSA_LON       = 60.78,   10.72    # Kise, søndre Mjøsa
BINGSFOSSEN_LAT, BINGSFOSSEN_LON = 60.2172, 11.5528  # Start
FETSUND_LAT,     FETSUND_LON     = 59.9297, 11.5833  # Mål / Fetsund lenser

# ── Modellparametere ─────────────────────────────────────────────────────────
TRAVEL_TIME_HOURS    = 25    # Vorma→Fetsund, typisk august
TEMPERATURE_SURVIVAL = 0.14  # 14 % av temperaturfall overlever fortynning
CRITICAL_WIND_SPEED  = 1.9   # m/s vedvarende sørlig vind for å utløse oppvelling
WIND_SECTOR_MIN      = 135   # Kritisk vindretning fra (°)
WIND_SECTOR_MAX      = 225   # Kritisk vindretning til (°)

# ── Open Water temperaturgrenser ─────────────────────────────────────────────
# Basert på World Aquatics (FINA) OW-regler og norske sikkerhetsterskler
OW_ABORT            = 14.0   # Under: arrangement bør avlyses
OW_WETSUIT_REQUIRED = 16.0   # Under: våtdrakt obligatorisk (FINA-minimum)
OW_WETSUIT_STRONG   = 18.0   # Under: våtdrakt sterkt anbefalt
OW_WETSUIT_OPTIONAL = 20.0   # Under: våtdrakt anbefalt / valgfritt
OW_TOO_WARM         = 24.0   # Over:  våtdrakt frarådes (overopphetingsrisiko)

# ── Arrangement ──────────────────────────────────────────────────────────────
EVENT_YEAR        = 2026
EVENT_MONTH       = 8
EVENT_DAY_OF_WEEK = 5   # Lørdag (0=mandag)

#-- Bilde ---
GD_header = Image.open("Samensatt_logo_GlommDyppen.jpg")

# ============================================================================
# DATA FETCHING
# ============================================================================

@st.cache_data(ttl=3600)
def fetch_nve_data(station_id, parameter, hours_back=168):
    """Henter data fra NVE HydAPI. Parameter 1003=temp, 1001=vannføring."""
    try:
        url = f"{NVE_BASE_URL}/Observations"
        headers = {"X-API-Key": NVE_API_KEY, "accept": "application/json"} if NVE_API_KEY else {"accept": "application/json"}
        params = {"StationId": station_id, "Parameter": str(parameter), "ResolutionTime": "60"}

        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if not (data.get('data') and len(data['data']) > 0):
            return pd.DataFrame(columns=['time', 'value', 'quality'])

        observations = data['data'][0].get('observations')
        if not observations:
            return pd.DataFrame(columns=['time', 'value', 'quality'])

        df = pd.DataFrame(observations)
        if 'time' not in df.columns or 'value' not in df.columns:
            return pd.DataFrame(columns=['time', 'value', 'quality'])

        df['time'] = pd.to_datetime(df['time'])
        end_time   = pd.Timestamp.now(tz='UTC')
        df = df[df['time'] >= end_time - pd.Timedelta(hours=hours_back)]

        if 'quality' in df.columns:
            df = df[df['quality'].isin([1, 2])]

        df = df.sort_values('time').reset_index(drop=True)
        for col in ['time', 'value', 'quality']:
            if col not in df.columns:
                df[col] = None

        return df[['time', 'value', 'quality']]

    except requests.exceptions.HTTPError:
        return pd.DataFrame(columns=['time', 'value', 'quality'])
    except Exception as e:
        if 'time' not in str(e):
            st.warning(f"Datafeil stasjon {station_id}: {str(e)[:100]}")
        return pd.DataFrame(columns=['time', 'value', 'quality'])


@st.cache_data(ttl=3600)
def fetch_frost_wind(hours_back=168):
    """
    Henter historiske vindmålinger fra Frost API (Kise, SN12680).
    Brukes som observasjonskilde for siste uke – CERRA-reanalyse har for lang
    forsinkelse til sanntidsbruk.
    """
    try:
        end_time   = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)

        url    = f"{FROST_BASE_URL}/observations/v0.jsonld"
        params = {
            "sources":         FROST_STATION_KISE,
            "elements":        "wind_speed,wind_from_direction",
            "referencetime":   f"{start_time.strftime('%Y-%m-%dT%H:%M:%SZ')}/{end_time.strftime('%Y-%m-%dT%H:%M:%SZ')}",
            "timeresolutions": "PT1H",
        }
        r = requests.get(url, params=params, auth=(FROST_CLIENT_ID, ""), timeout=30)

        if r.status_code != 200:
            return pd.DataFrame()

        records = []
        for item in r.json().get('data', []):
            obs_dict = {'time': pd.to_datetime(item['referenceTime'])}
            for obs in item.get('observations', []):
                obs_dict[obs['elementId']] = obs['value']
            records.append(obs_dict)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).sort_values('time').reset_index(drop=True)
        df = df.rename(columns={
            'wind_speed':         'wind_speed',
            'wind_from_direction': 'wind_direction'
        })
        return df

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=21600)
def fetch_weather_forecast(lat, lon, days_ahead=14):
    """Henter varsel fra Met.no Locationforecast (opp til ~10 dager med timesoppløsning)."""
    try:
        url     = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
        headers = {"User-Agent": "GlommadyppenApp/1.0 kontakt@glommadyppen.no"}
        params  = {"lat": lat, "lon": lon}

        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        forecast_list = []
        max_time = pd.Timestamp.now(tz='UTC') + pd.Timedelta(days=days_ahead)

        for ts in data['properties']['timeseries']:
            t = pd.to_datetime(ts['time'])
            if t > max_time:
                break
            details = ts['data']['instant']['details']
            forecast_list.append({
                'time':            t,
                'air_temperature': details.get('air_temperature'),
                'wind_speed':      details.get('wind_speed'),
                'wind_direction':  details.get('wind_from_direction'),
                'wind_gust':       details.get('wind_speed_of_gust'),
            })

        return pd.DataFrame(forecast_list)

    except Exception as e:
        st.error(f"Feil ved henting av varsel: {e}")
        return pd.DataFrame()


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def add_southerly_component(df):
    """Legger til southerly_wind-kolonne (vind fra SE/S sektor, 135–225°)."""
    if df.empty or 'wind_direction' not in df.columns:
        return df
    is_ses = (df['wind_direction'] >= WIND_SECTOR_MIN) & (df['wind_direction'] <= WIND_SECTOR_MAX)
    df['southerly_wind'] = np.where(is_ses, df['wind_speed'], 0.0)
    return df


def detect_temperature_drop(df, threshold_C=2.0, window_hours=6):
    """Detekterer signifikante temperaturfall i et tidsvindu."""
    if df.empty or len(df) < 2:
        return None
    df = df.sort_values('time').copy()
    cutoff = df['time'].max() - pd.Timedelta(hours=window_hours)
    recent = df[df['time'] >= cutoff]
    if len(recent) < 2:
        return None
    max_t, min_t = recent['value'].max(), recent['value'].min()
    drop = max_t - min_t
    if drop < threshold_C:
        return None
    return {
        'magnitude': drop,
        'max_temp':  max_t,
        'min_temp':  min_t,
        'max_time':  recent[recent['value'] == max_t]['time'].iloc[0],
        'min_time':  recent[recent['value'] == min_t]['time'].iloc[0],
    }


def predict_fetsund_temperature(vorma_temp_df, event_datetime):
    """
    Predikerer Fetsund-temperatur basert på Vorma-temperatur,
    25 timers transporttid og 14 % fortynningsoverlevelse.
    """
    if vorma_temp_df.empty:
        return None

    if event_datetime.tzinfo is None:
        event_datetime = event_datetime.replace(tzinfo=pd.Timestamp.now(tz='UTC').tzinfo)

    prediction_time = event_datetime - timedelta(hours=TRAVEL_TIME_HOURS)

    df = vorma_temp_df.copy()
    df['time'] = pd.to_datetime(df['time'])
    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize('UTC')

    df['time_diff'] = abs(df['time'] - prediction_time)
    closest_idx = df['time_diff'].idxmin()
    if pd.isna(closest_idx):
        return None

    vorma_temp = df.loc[closest_idx, 'value']
    vorma_time = df.loc[closest_idx, 'time']

    baseline_temp = df[df['time'] >= (vorma_time - timedelta(hours=48))]['value'].mean()
    anomaly       = vorma_temp - baseline_temp
    fetsund_temp  = baseline_temp + anomaly * TEMPERATURE_SURVIVAL

    return {
        'predicted_temp': fetsund_temp,
        'vorma_temp':     vorma_temp,
        'baseline_temp':  baseline_temp,
        'anomaly':        anomaly,
        'vorma_time':     vorma_time,
        'confidence':     _calculate_confidence(df, prediction_time),
    }


def _calculate_confidence(df, target_time):
    latest = pd.to_datetime(df['time'].max())
    if latest.tz is None:
        latest = latest.tz_localize('UTC')
    if target_time.tz is None:
        target_time = target_time.tz_localize('UTC')
    hours_old = (target_time - latest).total_seconds() / 3600
    if hours_old < 1:    tc = 1.0
    elif hours_old < 6:  tc = 0.9
    elif hours_old < 24: tc = 0.7
    else:                tc = 0.5
    return tc * min(len(df) / 72, 1.0)


def assess_risk_open_water(predicted_temp, weather_forecast=None):
    """
    Risikovurdering basert på Open Water-regler (World Aquatics / FINA) og
    norske sikkerhetsterskler. Returnerer:
        risk_label, color, wetsuit_status, wetsuit_color, details_list
    """
    if predicted_temp is None:
        return "UKJENT", "#6c757d", "Ukjent", "#6c757d", []

    # Oppvellings-risiko fra vindvarsel
    southerly_risk = False
    if weather_forecast is not None and not weather_forecast.empty:
        df_wf = weather_forecast.copy()
        if 'southerly_wind' not in df_wf.columns:
            df_wf = add_southerly_component(df_wf)
        avg_s = df_wf.head(48)['southerly_wind'].mean()
        southerly_risk = avg_s >= CRITICAL_WIND_SPEED

    details = []

    if predicted_temp < OW_ABORT:
        label  = "ARRANGEMENT BØR AVLYSES"
        color  = "#6B0000"
        ws     = "Ikke aktuelt — for kaldt"
        ws_col = "#6B0000"
        details = [
            f"Predikert temperatur {predicted_temp:.1f} °C er under absolutt minimumsgrense (14 °C).",
            "World Aquatics (FINA) forbyr konkurranser under 16 °C.",
            "Hypotermirisiko er ekstremt høy — arrangementet bør avlyses.",
        ]

    elif predicted_temp < OW_WETSUIT_REQUIRED:
        label  = "HØY RISIKO – VURDER AVLYSNING"
        color  = "#dc3545"
        ws     = "Våtdrakt obligatorisk"
        ws_col = "#dc3545"
        details = [
            f"Predikert temperatur {predicted_temp:.1f} °C er under FINA-minimumsgrensen på 16 °C.",
            "Våtdrakt er obligatorisk i henhold til internasjonale Open Water-regler.",
            "Arrangør bør vurdere om arrangementet er forsvarlig å gjennomføre.",
            "Deltakere uten våtdrakt bør ikke starte.",
        ]

    elif predicted_temp < OW_WETSUIT_STRONG:
        label  = "MODERAT RISIKO"
        color  = "#e07b00"
        ws     = "Våtdrakt sterkt anbefalt"
        ws_col = "#e07b00"
        details = [
            f"Predikert temperatur {predicted_temp:.1f} °C er kaldt for langdistansesvømming.",
            "FINA tillater arrangement, men anbefaler våtdrakt i dette temperaturintervallet.",
            "Alle deltakere bør bruke våtdrakt — spesielt for distanser over 5 km.",
            "Utrente og uerfarne bør ikke starte uten våtdrakt.",
        ]

    elif predicted_temp < OW_WETSUIT_OPTIONAL:
        label  = "LAV RISIKO"
        color  = "#f0a500"
        ws     = "Våtdrakt anbefalt"
        ws_col = "#f0a500"
        details = [
            f"Predikert temperatur {predicted_temp:.1f} °C er kjølig — våtdrakt gir komfort og sikkerhet.",
            "Erfarne langdistansesvømmere kan vurdere uten våtdrakt.",
            "Nybegynnere og utrente anbefales sterkt å bruke våtdrakt.",
        ]

    elif predicted_temp < OW_TOO_WARM:
        label  = "GODE FORHOLD"
        color  = "#28a745"
        ws     = "Våtdrakt valgfritt"
        ws_col = "#28a745"
        details = [
            f"Predikert temperatur {predicted_temp:.1f} °C er ideell for Open Water-svømming.",
            "Våtdrakt er tillatt men ikke nødvendig for de fleste deltakere.",
            "Utmerkede konkurranseforhold.",
        ]

    else:
        label  = "VARMT VANN"
        color  = "#17a2b8"
        ws     = "Våtdrakt frarådes"
        ws_col = "#c0392b"
        details = [
            f"Predikert temperatur {predicted_temp:.1f} °C — varmt vann.",
            "Våtdrakt kan gi overopphetingsrisiko og frarådes.",
            "Gode svømmeforhold — vær oppmerksom på hydrering.",
        ]

    if southerly_risk:
        details.append(
            "⚠️ Vedvarende sørlig vind er varslet — temperaturfall fra Mjøsa-oppvelling er mulig."
        )

    return label, color, ws, ws_col, details


def calculate_event_date(year):
    """Beregner dato for første lørdag i august."""
    first_day = datetime(year, EVENT_MONTH, 1)
    days_to_sat = (EVENT_DAY_OF_WEEK - first_day.weekday()) % 7
    if days_to_sat == 0 and first_day.weekday() != EVENT_DAY_OF_WEEK:
        days_to_sat = 7
    event_date = first_day + timedelta(days=days_to_sat)
    event_date = event_date.replace(hour=10, minute=0, second=0)
    return pd.Timestamp(event_date).tz_localize('Europe/Oslo').tz_convert('UTC')


def wind_rose_label(degrees):
    """Konverterer vindretning i grader til kompassretning."""
    dirs = ['N', 'NØ', 'Ø', 'SØ', 'S', 'SV', 'V', 'NV']
    return dirs[round(degrees / 45) % 8]


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

_LAYOUT_BASE = dict(
    hovermode='x unified',
    template='plotly_white',
    margin=dict(l=50, r=20, t=50, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="center", x=0.5, font=dict(size=10)),
)

STATION_COLORS = {
    'Svanefoss':  '#4472C4',
    'Funnefoss':  '#2E86AB',
    'Blaker':     '#06A77D',
    'Fetsund':    '#A23B72',
    'Ertesekken': '#E67E22',
}


def _temp_chart(stations_dict, title="Vanntemperatur"):
    """stations_dict: {'Stasjonsnavn': df_with_value_col}"""
    fig = go.Figure()
    for name, df in stations_dict.items():
        if df is None or df.empty:
            continue
        col = 'value' if 'value' in df.columns else df.columns[1]
        fig.add_trace(go.Scatter(
            x=df['time'], y=df[col], mode='lines', name=name,
            line=dict(color=STATION_COLORS.get(name, '#888'), width=2),
        ))
    # Reference lines
    for temp, label, color in [(16, "16 °C – FINA minimum", "red"),
                                (18, "18 °C – våtdrakt anbefalt", "orange"),
                                (20, "20 °C", "green")]:
        fig.add_hline(y=temp, line_dash="dot", line_color=color, opacity=0.4,
                      annotation_text=label, annotation_position="bottom right")
    fig.update_layout(title=title, xaxis_title="Tid", yaxis_title="°C",
                      height=380, **_LAYOUT_BASE)
    return fig


def _discharge_chart(stations_dict, title="Vannføring"):
    fig = go.Figure()
    for name, df in stations_dict.items():
        if df is None or df.empty:
            continue
        col = 'value' if 'value' in df.columns else df.columns[1]
        fig.add_trace(go.Scatter(
            x=df['time'], y=df[col], mode='lines', name=name,
            line=dict(color=STATION_COLORS.get(name, '#888'), width=2),
        ))
    fig.update_layout(title=title, xaxis_title="Tid", yaxis_title="m³/s",
                      height=350, **_LAYOUT_BASE)
    return fig


def _wind_obs_chart(df, title="Vindmålinger"):
    """Kombinert vindhastighet + retning fra observasjoner."""
    if df.empty or 'wind_speed' not in df.columns:
        return None
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12,
                        subplot_titles=('Vindhastighet (m/s)', 'Vindretning (°)'))

    is_ses = (df.get('wind_direction', pd.Series(dtype=float)) >= WIND_SECTOR_MIN) & \
             (df.get('wind_direction', pd.Series(dtype=float)) <= WIND_SECTOR_MAX)
    ses_speed = np.where(is_ses, df['wind_speed'], np.nan)

    fig.add_trace(go.Scatter(
        x=df['time'], y=df['wind_speed'], mode='lines', name='Total vind',
        line=dict(color='#06A77D', width=1.5), fill='tozeroy',
        fillcolor='rgba(6,167,125,0.12)'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df['time'], y=ses_speed, mode='lines', name='SE/S-vind',
        line=dict(color='#D62828', width=1.5, dash='dot')), row=1, col=1)
    fig.add_hline(y=CRITICAL_WIND_SPEED, line_dash="dot", line_color="red",
                  annotation_text=f"{CRITICAL_WIND_SPEED} m/s terskel",
                  row=1, col=1)

    if 'wind_direction' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['wind_direction'], mode='markers', name='Retning',
            marker=dict(size=4, color=df['wind_direction'],
                        colorscale='HSV', showscale=False)), row=2, col=1)
        fig.add_hrect(y0=WIND_SECTOR_MIN, y1=WIND_SECTOR_MAX,
                      fillcolor="rgba(214,40,40,0.08)", line_width=0,
                      annotation_text="Kritisk SE/S", row=2, col=1)
        fig.update_yaxes(range=[0, 360], row=2, col=1)

    fig.update_layout(title=title, height=500, showlegend=True, **_LAYOUT_BASE)
    return fig


def _wind_forecast_chart(df, title="Vindvarsel"):
    """Forecast-versjon av vindkartet (samme layout, annen farge)."""
    if df.empty or 'wind_speed' not in df.columns:
        return None
    df = df.copy()
    if 'southerly_wind' not in df.columns:
        df = add_southerly_component(df)

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12,
                        subplot_titles=('Vindhastighet (m/s)', 'Vindretning (°)'))

    fig.add_trace(go.Scatter(
        x=df['time'], y=df['wind_speed'], mode='lines', name='Total vind',
        line=dict(color='#2E86AB', width=1.5), fill='tozeroy',
        fillcolor='rgba(46,134,171,0.12)'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['southerly_wind'], mode='lines', name='SE/S-vind',
        line=dict(color='#D62828', width=1.5, dash='dot')), row=1, col=1)
    fig.add_hline(y=CRITICAL_WIND_SPEED, line_dash="dot", line_color="red",
                  annotation_text=f"{CRITICAL_WIND_SPEED} m/s terskel", row=1, col=1)

    if 'wind_direction' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['wind_direction'], mode='markers', name='Retning',
            marker=dict(size=4, color=df['wind_direction'],
                        colorscale='HSV', showscale=False)), row=2, col=1)
        fig.add_hrect(y0=WIND_SECTOR_MIN, y1=WIND_SECTOR_MAX,
                      fillcolor="rgba(214,40,40,0.08)", line_width=0,
                      annotation_text="Kritisk SE/S", row=2, col=1)
        fig.update_yaxes(range=[0, 360], row=2, col=1)

    fig.update_layout(title=title, height=500, showlegend=True, **_LAYOUT_BASE)
    return fig


def _daily_forecast_table(df, days=10):
    """Lager daglig sammendragstabell fra Met.no-varsel."""
    if df.empty:
        return None
    df = df.copy()
    if 'southerly_wind' not in df.columns:
        df = add_southerly_component(df)
    df['date'] = pd.to_datetime(df['time']).dt.tz_convert('Europe/Oslo').dt.date
    rows = []
    for date in sorted(df['date'].unique())[:days]:
        d = df[df['date'] == date]
        avg_s = d['southerly_wind'].mean()
        risiko_ikon = "🔴" if avg_s >= CRITICAL_WIND_SPEED else \
                      "🟡" if avg_s >= 1.2 else "🟢"
        rows.append({
            'Dato':            pd.to_datetime(date).strftime('%a %d.%m'),
            'Lufttemp':        f"{d['air_temperature'].min():.0f}–{d['air_temperature'].max():.0f} °C",
            'Vind gj.snitt':   f"{d['wind_speed'].mean():.1f} m/s",
            'Vind maks':       f"{d['wind_speed'].max():.1f} m/s",
            'Retning':         f"{d['wind_direction'].mean():.0f}° ({wind_rose_label(d['wind_direction'].mean())})",
            'SE/S-vind':       f"{avg_s:.1f} m/s",
            'Oppv.risiko':     risiko_ikon,
        })
    return pd.DataFrame(rows)


# ============================================================================
# PAGE: PREDIKSJON
# ============================================================================

def page_prediksjon():
    st.title("Temperaturprediksjon")
    st.markdown("Predikert vanntemperatur under Glommadyppen basert på observasjoner i Mjøsa, Vorma og Glomma")

    event_date  = calculate_event_date(EVENT_YEAR)
    days_until  = (event_date - pd.Timestamp.now(tz='UTC')).days
    oslo_dt     = event_date.tz_convert('Europe/Oslo')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Neste arrangement", oslo_dt.strftime("%d. %B %Y"))
    c2.metric("Dager igjen",       f"{days_until} dager")
    c3.metric("Starttid",          "10:00")
    c4.metric("Stasjon",
              "🟢 Aktiv" if 4 <= datetime.now().month <= 9 else "🔴 Offline (vinter)")

    st.divider()

    with st.spinner("Henter data…"):
        vorma_temp        = fetch_nve_data(STATION_FUNNEFOSS_TEMP, 1003, hours_back=168)
        svanefoss_temp    = fetch_nve_data(STATION_SVANEFOSS,      1003, hours_back=168)
        blaker_temp       = fetch_nve_data(STATION_BLAKER,         1003, hours_back=168)
        fetsund_temp      = fetch_nve_data(STATION_FETSUND,        1003, hours_back=168)
        weather_mjosa     = fetch_weather_forecast(MJOSA_LAT, MJOSA_LON)
        if not weather_mjosa.empty:
            weather_mjosa = add_southerly_component(weather_mjosa)

    # ── Station offline check ─────────────────────────────────────────────────
    if vorma_temp.empty and svanefoss_temp.empty:
        st.warning("""
        ⚠️ **Målestasjon offline (vintersesong)**

        Vorma-stasjonene er offline.  Forventes aktive igjen april 2026.
        """)
        if not weather_mjosa.empty:
            st.subheader("💨 Vindvarsel Mjøsa")
            chart = _wind_forecast_chart(weather_mjosa.head(120), "Vindvarsel – Mjøsa (5 dager)")
            if chart:
                st.plotly_chart(chart, use_container_width=True)
        st.stop()

    # Bruk beste tilgjengelige upstream-stasjon
    primary_df = svanefoss_temp if not svanefoss_temp.empty else vorma_temp

    # ── Dataalder ─────────────────────────────────────────────────────────────
    latest_time    = pd.to_datetime(primary_df.iloc[-1]['time'])
    if latest_time.tz is None:
        latest_time = latest_time.tz_localize('UTC')
    data_age_hours = (pd.Timestamp.now(tz='UTC') - latest_time).total_seconds() / 3600
    data_age_days  = data_age_hours / 24

    if data_age_days > 7:
        st.warning(f"⚠️ Siste Vorma-måling er {data_age_days:.1f} dager gammel – stasjonen kan være offline.")

    # ── Nåstatus ─────────────────────────────────────────────────────────────
    st.header("Nåværende status")
    c1, c2, c3, c4 = st.columns(4)

    latest_val = primary_df.iloc[-1]['value']
    if len(primary_df) >= 24:
        delta_24 = f"{latest_val - primary_df.iloc[-24]['value']:+.1f} °C (24t)"
    else:
        delta_24 = "–"
    c1.metric("Vorma nå", f"{latest_val:.1f} °C", delta=delta_24)

    drop = detect_temperature_drop(primary_df, threshold_C=2.0, window_hours=6)
    if drop:
        c2.metric("Temperaturfall (6t)", f"{drop['magnitude']:.1f} °C", delta="⚠️ Detektert!", delta_color="inverse")
    else:
        c2.metric("Temperaturfall (6t)", "Ingen", delta="✓ Stabilt")

    if not weather_mjosa.empty:
        cw = weather_mjosa.iloc[0]
        c3.metric("Vind (Mjøsa)", f"{cw['wind_speed']:.1f} m/s",
                  delta=f"{cw['wind_direction']:.0f}° ({wind_rose_label(cw['wind_direction'])})")
    else:
        c3.metric("Vind (Mjøsa)", "N/A")

    freshness = ("✓ Fersk" if data_age_hours < 2 else
                 "⚠️ Timer gammel" if data_age_hours < 6 else "❌ Utdatert")
    c4.metric("Datastatus", f"{data_age_hours:.0f}t siden", delta=freshness,
              delta_color="normal" if data_age_hours < 2 else "inverse")

    st.divider()

    # ── Prediksjon ────────────────────────────────────────────────────────────
    st.header("Prediksjon for arrangementet")
    prediction = predict_fetsund_temperature(
        primary_df.rename(columns={'value': 'value'}) if 'value' in primary_df.columns else primary_df,
        event_date
    )

    if data_age_days > 30 and days_until > 30:
        st.info("""
        Sanntidsprediksjon krever ferske Vorma-målinger.
        Prediksjonen aktiveres når stasjonen starter opp igjen (april 2026).
        """)
    elif prediction:
        pred_temp = prediction['predicted_temp']
        risk_label, risk_color, ws_label, ws_color, risk_details = \
            assess_risk_open_water(pred_temp, weather_mjosa)

        # Hovedkort
        st.markdown(f"""
        <div style='background:{risk_color}; padding:20px; border-radius:12px;
                    color:white; text-align:center; margin-bottom:16px;'>
          <div style='font-size:0.95em; opacity:0.9; margin-bottom:4px;'>
            Predikert temperatur ved Fetsund
          </div>
          <div style='font-size:3em; font-weight:700; line-height:1.1;'>
            {pred_temp:.1f} °C
          </div>
          <div style='font-size:1.1em; margin-top:8px; font-weight:600;'>
            {risk_label}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Våtdrakt-badge
        st.markdown(f"""
        <div style='background:{ws_color}; padding:10px 16px; border-radius:8px;
                    color:white; display:inline-block; font-weight:600;
                    font-size:1.05em; margin-bottom:16px;'>
          🏊 Våtdrakt: {ws_label}
        </div>
        """, unsafe_allow_html=True)

        # Regler-detaljer
        with st.expander("Vis risikovurdering og Open Water-regler (draft)", expanded=False):
            for d in risk_details:
                st.markdown(f"- {d}")
            st.markdown("---")
            st.markdown("""
            **Temperaturgrenser (World Aquatics / FINA):**
            | Temp | Vurdering | Våtdrakt |
            |------|-----------|----------|
            | < 14 °C | Arrangement bør avlyses | Ikke aktuelt |
            | 14–16 °C | Høy risiko | Obligatorisk |
            | 16–18 °C | Moderat risiko | Sterkt anbefalt |
            | 18–20 °C | Lav risiko | Anbefalt |
            | 20–24 °C | Gode forhold | Valgfritt |
            | > 24 °C | Varmt vann | Frarådes |
            """)

        # Detaljer
        c1, c2, c3 = st.columns(3)
        c1.metric("Vorma (25t før)", f"{prediction['vorma_temp']:.1f} °C")
        c2.metric("Baseline (48t snitt)", f"{prediction['baseline_temp']:.1f} °C")
        c3.metric("Pålitelighet", f"{prediction['confidence']*100:.0f} %")

        std_err = 2.0
        margin  = std_err * 1.96
        st.info(f"""
        **95 % konfidensintervall:** {pred_temp - margin:.1f} – {pred_temp + margin:.1f} °C

        Modell: 25 t transporttid · 14 % fortynningsoverlevelse · {len(primary_df)} målinger
        """)
        st.warning("⚠️ Modellen er validert opp mot data fra juli og august. Bruk med forsiktighet utenfor sommermånedene.")
    else:
        st.warning("⚠️ Ikke nok data for prediksjon.")

    st.divider()

    # ── Temperaturhistorikk ───────────────────────────────────────────────────
    st.subheader("Temperaturhistorikk – siste 7 dager")
    temp_fig = _temp_chart({
        'Svanefoss': svanefoss_temp,
        'Blaker':    blaker_temp,
        'Fetsund':   fetsund_temp,
    }, "Vanntemperatur (siste 7 dager)")
    st.plotly_chart(temp_fig, use_container_width=True)

    # ── Vindanalyse ───────────────────────────────────────────────────────────
    if not weather_mjosa.empty:
        st.divider()
        st.subheader("Vindvarsel – Mjøsa (5 dager)")

        next_48h  = weather_mjosa.head(48)
        avg_wind  = next_48h['wind_speed'].mean()
        max_wind  = next_48h['wind_speed'].max()
        avg_ses   = next_48h['southerly_wind'].mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("Gj.snitt vind (48t)", f"{avg_wind:.1f} m/s")
        c2.metric("Maks vind (48t)",     f"{max_wind:.1f} m/s")
        if avg_ses >= CRITICAL_WIND_SPEED:
            c3.metric("SE/S-vind (48t)", f"{avg_ses:.1f} m/s",
                      delta="⚠️ Oppvellings-risiko!", delta_color="inverse")
        else:
            c3.metric("SE/S-vind (48t)", f"{avg_ses:.1f} m/s")

        chart = _wind_forecast_chart(weather_mjosa.head(120), "Vindvarsel Mjøsa")
        if chart:
            st.plotly_chart(chart, use_container_width=True)


# ============================================================================
# PAGE: DATA & VARSEL
# ============================================================================

def page_data_varsel():
    st.title("Observasjoner og Værvarsler")
    st.markdown(
        "Faktiske målinger fra NVE og met.no"
        "Bruk denne siden for å se rå data og standard værvarsler."
    )

    tabs = st.tabs([
        "🌊 NVE Vanntemperatur",
        "💧 NVE Vannføring",
        "🌬️ Vind ved Mjøsa",
        "🌤️ Værvarsler",
    ])

    # ── Felles datahenting ────────────────────────────────────────────────────
    with st.spinner("Henter observasjoner…"):
        sv_temp   = fetch_nve_data(STATION_SVANEFOSS,      1003, hours_back=168)
        fn_temp   = fetch_nve_data(STATION_FUNNEFOSS_TEMP, 1003, hours_back=168)
        bl_temp   = fetch_nve_data(STATION_BLAKER,         1003, hours_back=168)
        fe_temp   = fetch_nve_data(STATION_FETSUND,        1003, hours_back=168)
        er_q      = fetch_nve_data(STATION_ERTESEKKEN_Q,   1001, hours_back=168)
        bl_q      = fetch_nve_data(STATION_BLAKER,         1001, hours_back=168)
        fn_q      = fetch_nve_data(STATION_FUNNEFOSS_Q,    1001, hours_back=168)
        frost_vind = fetch_frost_wind(hours_back=168)
        fc_mjosa  = fetch_weather_forecast(MJOSA_LAT,   MJOSA_LON)
        fc_fetsund = fetch_weather_forecast(FETSUND_LAT, FETSUND_LON)

    # ────────────────────────────────────────────────────────────────────────
    # TAB 1: Vanntemperatur
    # ────────────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Vanntemperatur – siste 7 dager (NVE HydAPI)")
        st.caption("Timesverdier fra stasjonene langs Vorma og Glomma. Bare data med de to høyeste kvalitetene vises.")

        # Nåverdier
        c1, c2, c3, c4 = st.columns(4)
        def _latest(df, label, col):
            if df.empty:
                col.metric(label, "N/A")
            else:
                col.metric(label, f"{df.iloc[-1]['value']:.1f} °C")
        _latest(sv_temp, "Svanefoss (Vorma)", c1)
        _latest(fn_temp, "Funnefoss (Vorma)", c2)
        _latest(bl_temp, "Blaker (Glomma)",   c3)
        _latest(fe_temp, "Fetsund (Glomma)",  c4)

        fig = _temp_chart({
            'Svanefoss': sv_temp,
            'Funnefoss': fn_temp,
            'Blaker':    bl_temp,
            'Fetsund':   fe_temp,
        }, "Vanntemperatur – siste 7 dager")
        st.plotly_chart(fig, use_container_width=True)

        st.caption("""
        **Stasjoner:**  
        - **Svanefoss** (2.52.0) — i Vorma, 22 km fra Mjøsa.  
        - **Funnefoss** (2.410.0) — i Vorma ca. 23,5 km fra Mjøsa.  
        - **Blaker** (2.17.0) — i Glomma, primær målestasjon.  
        - **Fetsund** (2.587.0) — Målgang Glommadyppen.
        """)

    # ────────────────────────────────────────────────────────────────────────
    # TAB 2: Vannføring
    # ────────────────────────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Vannføring – siste 7 dager (NVE HydAPI)")
        st.caption("Timesverdier i m³/s. Vannføring forbi Ertesekken brukes for å finne ut tiden det tar før kaldt vann når arrangementet")

        c1, c2, c3 = st.columns(3)
        def _latest_q(df, label, col):
            if df.empty:
                col.metric(label, "N/A")
            else:
                v = df.iloc[-1]['value']
                t = round(9700 / v, 1) if v > 0 else None
                col.metric(label, f"{v:.0f} m³/s",
                           help=f"Gir transporttid ≈ {t} t" if t else None)
        _latest_q(er_q, "Ertesekken (Vorma)", c1)
        _latest_q(fn_q, "Funnefoss (Glomma)", c2)
        _latest_q(bl_q, "Blaker (Glomma)",    c3)

        fig = _discharge_chart({
            'Ertesekken': er_q,
            'Funnefoss':  fn_q,
            'Blaker':     bl_q,
        }, "Vannføring – siste 7 dager")
        st.plotly_chart(fig, use_container_width=True)

        # Transport-kalkulator
        st.subheader("Transporttid-kalkulator")
        if not er_q.empty:
            q_now = er_q.iloc[-1]['value']
        else:
            q_now = 400.0
        q_val = st.slider("Vannføring ved Ertesekken (m³/s)",
                          min_value=100, max_value=1200,
                          value=int(q_now), step=10)
        t_calc = round(9700 / q_val, 1)
        st.info(f"""
        **t = 9700 / {q_val} = {t_calc} timer** (Svanefoss → Fetsund, 45 km)
        *(t = 6871/Q for Svanefoss→Blaker)*
        """)

        st.caption("""
        **Stasjoner:**  
        - **Ertesekken** (2.197.0) — Vorma, nøkkelstasjon for transporttidsmodellen.  
        - **Funnefoss nedre** (2.279.0) — Glomma, oppstrøms samløp med Vorma.  
        - **Blaker** (2.17.0) — Glomma, nedenfor samløp (typisk 1,45 × Ertesekken).
        """)

    # ────────────────────────────────────────────────────────────────────────
    # TAB 3: Vind ved Mjøsa (observasjoner)
    # ────────────────────────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Vindmålinger – Kise, søndre Mjøsa (siste 7 dager)")
        st.caption(
            f"Kilde: MET.no Frost API · Stasjon {FROST_STATION_KISE} (Kise) · Timesverdier. "
        )

        if frost_vind.empty:
            st.warning(
                "Vindmålinger fra Frost API ikke tilgjengelig. "
                "Sjekk internettforbindelsen eller Frost-tjenestens status."
            )
        else:
            if 'wind_direction' in frost_vind.columns:
                is_ses = ((frost_vind['wind_direction'] >= WIND_SECTOR_MIN) &
                          (frost_vind['wind_direction'] <= WIND_SECTOR_MAX))
                avg_ses = frost_vind.loc[is_ses, 'wind_speed'].mean() if is_ses.any() else 0.0
                ses_hours = int(is_ses.sum())

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Vindhastighet nå",
                          f"{frost_vind.iloc[-1]['wind_speed']:.1f} m/s")
                c2.metric("Gj.snitt total (7d)",
                          f"{frost_vind['wind_speed'].mean():.1f} m/s")
                c3.metric("Timer SE/S-vind",   f"{ses_hours} t")
                if avg_ses >= CRITICAL_WIND_SPEED:
                    c4.metric("Gj.snitt SE/S",
                              f"{avg_ses:.1f} m/s", delta="⚠️ Over terskel", delta_color="inverse")
                else:
                    c4.metric("Gj.snitt SE/S (kun SE/S-timer)", f"{avg_ses:.1f} m/s")

            chart = _wind_obs_chart(frost_vind, f"Vindmålinger – {FROST_STATION_KISE} Kise")
            if chart:
                st.plotly_chart(chart, use_container_width=True)

            # Rådata
            with st.expander("Vis rådata (Frost API)", expanded=False):
                disp = frost_vind.copy()
                disp['time'] = disp['time'].dt.tz_convert('Europe/Oslo').dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(disp, use_container_width=True)

    # ────────────────────────────────────────────────────────────────────────
    # TAB 4: Værvarsler
    # ────────────────────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Værvarsler – Met.no Locationforecast (opp til 10 dager)")
        st.caption(
            "Kilde: Met.no Locationforecast 2.0 · Oppdateres ca. hver time · "
            "Timesoppløsning de første 3 dagene, deretter 6-timers intervaller. "
            "Varslene dekker ikke alltid et fullt 14-dagers vindu."
        )

        col_mjosa, col_fetsund = st.columns(2)

        with col_mjosa:
            st.markdown("### 📍 Søndre Mjøsa / Kise")
            st.caption(f"60.78°N, 10.72°E – referansepunkt for oppvellingsanalyse")
            if fc_mjosa.empty:
                st.warning("Varsel ikke tilgjengelig")
            else:
                fc_mjosa_s = add_southerly_component(fc_mjosa.copy())
                tbl = _daily_forecast_table(fc_mjosa_s)
                if tbl is not None:
                    st.dataframe(tbl, use_container_width=True, hide_index=True)
                chart = _wind_forecast_chart(fc_mjosa_s, "Vindvarsel – Mjøsa")
                if chart:
                    st.plotly_chart(chart, use_container_width=True)

        with col_fetsund:
            st.markdown("### 🏁 Fetsund lenser (mål)")
            st.caption(f"59.93°N, 11.58°E – arrangementspunkt")
            if fc_fetsund.empty:
                st.warning("Varsel ikke tilgjengelig")
            else:
                fc_fetsund_s = add_southerly_component(fc_fetsund.copy())
                tbl = _daily_forecast_table(fc_fetsund_s)
                if tbl is not None:
                    st.dataframe(tbl, use_container_width=True, hide_index=True)
                chart = _wind_forecast_chart(fc_fetsund_s, "Vindvarsel – Fetsund")
                if chart:
                    st.plotly_chart(chart, use_container_width=True)

        st.info("""
        **Om oppvellings-indikatoren (SE/S-vind):**  
        🟢 Lav risiko (< 1,2 m/s) · 🟡 Moderat (1,2–1,9 m/s) · 🔴 Høy (≥ 1,9 m/s vedvarende SE/S-vind)  
        Vind over tid fra sørøst–sør (135–225°) kan føre til kaldt vann fra Mjøsa til Glomma.
        """)


# ============================================================================
# MAIN – navigasjon
# ============================================================================

def main():
    with st.sidebar:
        st.image(GD_header, use_container_width=True)
        st.markdown("---")
        page = st.radio(
            "Navigasjon",
            options=["Prediksjon", "Observasjoner og værvarsel"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("""
        **Modell**
        - Transporttid = 9700/Q
        - Fortynning av Vorma: 14 %
        - Validert med data fra 2018–2025

        **Open Water-grenser for våtdrakt**
        - < 16 °C: Obligatorisk
        - 16–18 °C: Sterkt anbefalt
        - 18–20 °C: Anbefalt
        - > 20 °C: Valgfritt

        **Datakilder**
        - NVE HydAPI (vann)
        - MET Frost API (vind)
        - Met.no Locationforecast (værvarsel)
        """)
        st.markdown("---")
        if st.button("🔄 Oppdater data"):
            st.cache_data.clear()
            st.rerun()
        st.caption(
            f"Oppdatert {pd.Timestamp.now(tz='Europe/Oslo').strftime('%d.%m.%Y %H:%M')} | "
            "Utviklet av Fet Svømmeklubb for Glommadyppen.no"
        )

    if page == "Prediksjon":
        page_prediksjon()
    else:
        page_data_varsel()


if __name__ == "__main__":
    main()
