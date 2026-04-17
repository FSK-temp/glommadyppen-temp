"""
GlommaDyppen Vanntemperatur Prediksjon
Real-time water temperature prediction for GlommaDyppen swimming event

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
    page_title="GlommaDyppen Temperatur",
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
STATION_FUNNEFOSS_Q      = "2.412.0"   # Glomma / Funnefoss kraftverk
STATION_FETSUND          = "2.587.0"   # Fetsund bru – temperatur (målpunkt)

# ── Frost (met.no observations) ──────────────────────────────────────────────
FROST_STATION_KISE = "SN12680"   # Kise, søndre Mjøsa

# ── Met.no koordinater ───────────────────────────────────────────────────────
MJOSA_LAT,       MJOSA_LON       = 60.78,   10.72
BINGSFOSSEN_LAT, BINGSFOSSEN_LON = 60.2172, 11.5528
FETSUND_LAT,     FETSUND_LON     = 59.9297, 11.5833

# ── Modellparametere ─────────────────────────────────────────────────────────
TRANSPORT_COEFF      = 9700
TRANSPORT_COEFF_BLA  = 6871
FALLBACK_DISCHARGE   = 437.0
TEMPERATURE_SURVIVAL = 0.14
CRITICAL_WIND_SPEED  = 1.9
WIND_SECTOR_MIN      = 135
WIND_SECTOR_MAX      = 225
# Kumulativ vindenergi-terskel (timesdata, dt=1 per obs = 3× CERRA-konvensjon)
ENERGY_THRESHOLD     = 210.0   # m·h – HØY RISIKO (tilsvarer CERRA-kalibrert 70 m·h)
ENERGY_WARN          = 140.0   # m·h – MODERAT RISIKO (~67 % av terskel)

# ── Open Water temperaturgrenser ─────────────────────────────────────────────
OW_ABORT            = 14.0
OW_WETSUIT_REQUIRED = 16.0
OW_WETSUIT_STRONG   = 18.0
OW_WETSUIT_OPTIONAL = 20.0
OW_TOO_WARM         = 24.0

# ── Arrangement ──────────────────────────────────────────────────────────────
EVENT_YEAR        = 2026
EVENT_MONTH       = 8
EVENT_DAY_OF_WEEK = 5

GD_header = Image.open("Samensatt_logo_GlommDyppen.jpg")

# ============================================================================
# DATA FETCHING
# ============================================================================

@st.cache_data(ttl=3600)
def fetch_nve_data(station_id, parameter, hours_back=168):
    """
    Henter data fra NVE HydAPI.
    Parameter-koder (NVE exdat-format):
        1001 = vassføring (m³/s)
        1003 = vanntemperatur (°C)
    """
    try:
        url = f"{NVE_BASE_URL}/Observations"
        headers = ({"X-API-Key": NVE_API_KEY, "accept": "application/json"}
                   if NVE_API_KEY else {"accept": "application/json"})
        end_dt   = datetime.utcnow()
        start_dt = end_dt - timedelta(hours=hours_back)
        params = {
            "StationId":      station_id,
            "Parameter":      str(parameter),
            "ResolutionTime": "60",
            "ReferenceTime":  (
                f"{start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}/"
                f"{end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            ),
        }
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
            df = df[df['quality'].isin([0, 1, 2])]

        # Fysisk sanity-sjekk (Svanefoss har kjente sensorfeil ~−20 °C)
        if parameter == 1003:
            df = df[(df['value'] > 0.0) & (df['value'] < 35.0)]

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
    """Henter historiske vindmålinger fra Frost API (Kise, SN12680)."""
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
        df = df.rename(columns={'wind_from_direction': 'wind_direction'})
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=21600)
def fetch_weather_forecast(lat, lon, days_ahead=14):
    """Henter varsel fra Met.no Locationforecast."""
    try:
        url     = "https://api.met.no/weatherapi/locationforecast/2.0/complete"
        headers = {"User-Agent": "GlommaDyppenApp/1.0 stevne@fetsk.no"}
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
            precip = None
            for window in ('next_1_hours', 'next_6_hours'):
                if window in ts['data']:
                    precip = ts['data'][window].get('details', {}).get('precipitation_amount')
                    break
            forecast_list.append({
                'time':            t,
                'air_temperature': details.get('air_temperature'),
                'wind_speed':      details.get('wind_speed'),
                'wind_direction':  details.get('wind_from_direction'),
                'wind_gust':       details.get('wind_speed_of_gust'),
                'precipitation':   precip,
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
    is_ses = ((df['wind_direction'] >= WIND_SECTOR_MIN) &
              (df['wind_direction'] <= WIND_SECTOR_MAX))
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


def calculate_travel_time(discharge_df):
    """
    Beregner transporttid Svanefoss → Fetsund som t = 9700 / Q (timer).
    Returnerer (travel_time_hours, q_used, source_label).
    """
    if discharge_df is not None and not discharge_df.empty:
        recent = discharge_df.copy()
        recent['time'] = pd.to_datetime(recent['time'])
        cutoff = recent['time'].max() - pd.Timedelta(hours=24)
        last24 = recent[recent['time'] >= cutoff]['value']
        if len(last24) > 0:
            q = last24.median()
            return round(TRANSPORT_COEFF / q, 1), round(q, 0), "siste 24t (Ertesekken)"
    q = FALLBACK_DISCHARGE
    return round(TRANSPORT_COEFF / q, 1), q, f"august-median ({FALLBACK_DISCHARGE:.0f} m³/s)"


def predict_fetsund_temperature(vorma_temp_df, discharge_df, event_datetime):
    """
    Predikerer Fetsund-temperatur for arrangementet.
    Transporttid = 9700 / Q. 14 % av avviket overlever til Fetsund.
    """
    if vorma_temp_df.empty:
        return None
    if event_datetime.tzinfo is None:
        event_datetime = event_datetime.replace(tzinfo=pd.Timestamp.now(tz='UTC').tzinfo)

    travel_hours, q_used, q_source = calculate_travel_time(discharge_df)
    prediction_time = event_datetime - timedelta(hours=travel_hours)

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
        'travel_hours':   travel_hours,
        'q_used':         q_used,
        'q_source':       q_source,
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
    Risikovurdering basert på World Aquatics / FINA OW-regler.
    Returnerer: risk_label, color, wetsuit_status, wetsuit_color, details_list
    """
    if predicted_temp is None:
        return "UKJENT", "#6c757d", "Ukjent", "#6c757d", []

    southerly_risk = False
    if weather_forecast is not None and not weather_forecast.empty:
        df_wf = weather_forecast.copy()
        if 'southerly_wind' not in df_wf.columns:
            df_wf = add_southerly_component(df_wf)
        avg_s = df_wf.head(48)['southerly_wind'].mean()
        southerly_risk = avg_s >= CRITICAL_WIND_SPEED

    if predicted_temp < OW_ABORT:
        label, color = "Svømming bør ikke gjennomføres", "#6B0000"
        ws,  ws_col  = "Ikke aktuelt — for kaldt",       "#6B0000"
        details = [
            f"Predikert temperatur {predicted_temp:.1f} °C er under absolutt minimumsgrense (14 °C).",
            "World Aquatics (FINA) forbyr konkurranser under 16 °C.",
            "Hypotermirisiko er ekstremt høy — svømming bør ikke gjennomføres.",
        ]
    elif predicted_temp < OW_WETSUIT_REQUIRED:
        label, color = "Høy risiko – vurder svømming nøye", "#dc3545"
        ws,  ws_col  = "Våtdrakt obligatorisk",             "#dc3545"
        details = [
            f"Predikert temperatur {predicted_temp:.1f} °C er under FINA-minimumsgrensen på 16 °C.",
            "Våtdrakt er obligatorisk i henhold til internasjonale Open Water-regler.",
            "Arrangør bør vurdere om arrangementet er forsvarlig å gjennomføre.",
        ]
    elif predicted_temp < OW_WETSUIT_STRONG:
        label, color = "Moderat risiko", "#e07b00"
        ws,  ws_col  = "Våtdrakt sterkt anbefalt", "#e07b00"
        details = [
            f"Predikert temperatur {predicted_temp:.1f} °C er kaldt for langdistansesvømming.",
            "FINA tillater arrangement, men anbefaler våtdrakt i dette intervallet.",
            "Alle deltakere bør bruke våtdrakt — spesielt for distanser over 5 km.",
        ]
    elif predicted_temp < OW_WETSUIT_OPTIONAL:
        label, color = "Lav risiko", "#f0a500"
        ws,  ws_col  = "Våtdrakt anbefalt", "#f0a500"
        details = [
            f"Predikert temperatur {predicted_temp:.1f} °C — kjølig, våtdrakt gir komfort og sikkerhet.",
            "Erfarne langdistansesvømmere kan vurdere uten våtdrakt.",
        ]
    elif predicted_temp < OW_TOO_WARM:
        label, color = "Gode forhold", "#28a745"
        ws,  ws_col  = "Våtdrakt valgfritt", "#28a745"
        details = [
            f"Predikert temperatur {predicted_temp:.1f} °C er ideell for Open Water-svømming.",
            "Våtdrakt er tillatt men ikke nødvendig for de fleste deltakere.",
        ]
    else:
        label, color = "Varmt vann", "#17a2b8"
        ws,  ws_col  = "Våtdrakt frarådes", "#c0392b"
        details = [
            f"Predikert temperatur {predicted_temp:.1f} °C — varmt vann.",
            "Våtdrakt kan gi overopphetingsrisiko og frarådes.",
        ]

    if southerly_risk:
        details.append(
            "⚠️ Vedvarende sørlig vind er varslet — temperaturfall fra Mjøsa-oppvelling er mulig."
        )
    return label, color, ws, ws_col, details


def calculate_event_date(year):
    """Beregner dato for første lørdag i august."""
    first_day   = datetime(year, EVENT_MONTH, 1)
    days_to_sat = (EVENT_DAY_OF_WEEK - first_day.weekday()) % 7
    if days_to_sat == 0 and first_day.weekday() != EVENT_DAY_OF_WEEK:
        days_to_sat = 7
    event_date = first_day + timedelta(days=days_to_sat)
    event_date = event_date.replace(hour=10, minute=0, second=0)
    return pd.Timestamp(event_date).tz_localize('Europe/Oslo').tz_convert('UTC')


def wind_rose_label(degrees):
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
    fig = go.Figure()
    for name, df in stations_dict.items():
        if df is None or df.empty:
            continue
        col = 'value' if 'value' in df.columns else df.columns[1]
        fig.add_trace(go.Scatter(
            x=df['time'], y=df[col], mode='lines', name=name,
            line=dict(color=STATION_COLORS.get(name, '#888'), width=2),
        ))
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
    if df.empty or 'wind_speed' not in df.columns:
        return None
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12,
                        subplot_titles=('Vindhastighet (m/s)', 'Vindretning (°)'))
    is_ses = ((df.get('wind_direction', pd.Series(dtype=float)) >= WIND_SECTOR_MIN) &
              (df.get('wind_direction', pd.Series(dtype=float)) <= WIND_SECTOR_MAX))
    ses_speed = np.where(is_ses, df['wind_speed'], np.nan)
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['wind_speed'], mode='lines', name='Total vind',
        line=dict(color='#06A77D', width=1.5), fill='tozeroy',
        fillcolor='rgba(6,167,125,0.12)'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df['time'], y=ses_speed, mode='lines', name='SE/S-vind',
        line=dict(color='#D62828', width=1.5, dash='dot')), row=1, col=1)
    fig.add_hline(y=CRITICAL_WIND_SPEED, line_dash="dot", line_color="red",
                  annotation_text=f"{CRITICAL_WIND_SPEED} m/s terskel", row=1, col=1)
    if 'wind_direction' in df.columns:
        is_ses_bool = is_ses.values if hasattr(is_ses, 'values') else is_ses
        marker_colors = ['#D62828' if s else '#AAAAAA' for s in is_ses_bool]
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['wind_direction'], mode='markers', name='Retning',
            marker=dict(size=5, color=marker_colors),
            hovertemplate='%{y:.0f}°<extra></extra>'), row=2, col=1)
        fig.add_hrect(y0=WIND_SECTOR_MIN, y1=WIND_SECTOR_MAX,
                      fillcolor="rgba(214,40,40,0.08)", line_width=0,
                      annotation_text="Kritisk SE/S (135–225°)", row=2, col=1)
        fig.update_yaxes(range=[0, 360], tickvals=[0, 90, 180, 270, 360],
                         ticktext=['N', 'Ø', 'S', 'V', 'N'], row=2, col=1)
    fig.update_layout(title=title, height=500, showlegend=True, **_LAYOUT_BASE)
    return fig


def _wind_forecast_chart(df, title="Vindvarsel"):
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
        is_ses = ((df['wind_direction'] >= WIND_SECTOR_MIN) &
                  (df['wind_direction'] <= WIND_SECTOR_MAX))
        marker_colors = ['#D62828' if s else '#AAAAAA' for s in is_ses]
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['wind_direction'], mode='markers', name='Retning',
            marker=dict(size=5, color=marker_colors),
            hovertemplate='%{y:.0f}°<extra></extra>'), row=2, col=1)
        fig.add_hrect(y0=WIND_SECTOR_MIN, y1=WIND_SECTOR_MAX,
                      fillcolor="rgba(214,40,40,0.08)", line_width=0,
                      annotation_text="Kritisk SE/S (135–225°)", row=2, col=1)
        fig.update_yaxes(range=[0, 360], tickvals=[0, 90, 180, 270, 360],
                         ticktext=['N', 'Ø', 'S', 'V', 'N'], row=2, col=1)
    fig.update_layout(title=title, height=500, showlegend=True, **_LAYOUT_BASE)
    return fig


def _daily_forecast_table(df, days=10):
    if df.empty:
        return None
    df = df.copy()
    if 'southerly_wind' not in df.columns:
        df = add_southerly_component(df)
    df['date'] = pd.to_datetime(df['time']).dt.tz_convert('Europe/Oslo').dt.date
    rows = []
    for date in sorted(df['date'].unique())[:days]:
        d    = df[df['date'] == date]
        avg_s = d['southerly_wind'].mean()
        risiko_ikon = ("🔴" if avg_s >= CRITICAL_WIND_SPEED else
                       "🟡" if avg_s >= 1.2 else "🟢")
        rows.append({
            'Dato':          pd.to_datetime(date).strftime('%a %d.%m'),
            'Lufttemp':      f"{d['air_temperature'].min():.0f}–{d['air_temperature'].max():.0f} °C",
            'Vind gj.snitt': f"{d['wind_speed'].mean():.1f} m/s",
            'Vind maks':     f"{d['wind_speed'].max():.1f} m/s",
            'Retning':       f"{d['wind_direction'].mean():.0f}° ({wind_rose_label(d['wind_direction'].mean())})",
            'SE/S-vind':     f"{avg_s:.1f} m/s",
            'Oppv.risiko':   risiko_ikon,
        })
    return pd.DataFrame(rows)


def _daily_forecast_table_fetsund(df, days=10):
    if df.empty:
        return None
    df = df.copy()
    df['date'] = pd.to_datetime(df['time']).dt.tz_convert('Europe/Oslo').dt.date
    rows = []
    for date in sorted(df['date'].unique())[:days]:
        d = df[df['date'] == date]
        precip_sum = d['precipitation'].sum() if 'precipitation' in d.columns else None
        precip_str = f"{precip_sum:.1f} mm" if precip_sum is not None else "–"
        rows.append({
            'Dato':          pd.to_datetime(date).strftime('%a %d.%m'),
            'Lufttemp':      f"{d['air_temperature'].min():.0f}–{d['air_temperature'].max():.0f} °C",
            'Vind gj.snitt': f"{d['wind_speed'].mean():.1f} m/s",
            'Vind maks':     f"{d['wind_speed'].max():.1f} m/s",
            'Nedbør':        precip_str,
        })
    return pd.DataFrame(rows)


def _weather_fetsund_chart(df, title="Værvarsler – Fetsund"):
    if df.empty or 'wind_speed' not in df.columns:
        return None
    fig = make_subplots(
        rows=3, cols=1, vertical_spacing=0.10,
        subplot_titles=('Lufttemperatur (°C)', 'Vindhastighet (m/s)', 'Nedbør (mm/t)')
    )
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['air_temperature'], mode='lines', name='Lufttemp',
        line=dict(color='#E67E22', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['wind_speed'], mode='lines', name='Vind',
        line=dict(color='#2E86AB', width=1.5), fill='tozeroy',
        fillcolor='rgba(46,134,171,0.12)'), row=2, col=1)
    if 'precipitation' in df.columns:
        fig.add_trace(go.Bar(
            x=df['time'], y=df['precipitation'], name='Nedbør',
            marker_color='rgba(70,130,180,0.6)'), row=3, col=1)
    fig.update_layout(title=title, height=520, showlegend=False, **_LAYOUT_BASE)
    return fig


# ============================================================================
# FORECAST FUNCTIONS
# ============================================================================

def build_fetsund_forecast(vorma_df, fetsund_df, discharge_df,
                           hours_ahead=120, step_h=3):
    """
    Beregner tidsserie for predikert Fetsund-temperatur med usikkerhetsintervaller.

    Usikkerhetsmodell
    -----------------
    Båndenebegynner med bredde 0 ved siste observerte Fetsund-temperatur og
    vokser i to faser:
      1. Lineær oppramp   (t = 0 → travel_h):   σ = MODEL_SIGMA × (h / travel_h)
      2. Kvadratrot-vekst (t > travel_h):        σ = MODEL_SIGMA × √(1 + (h − travel_h) / 24)
    Dette gir σ = 0 ved t = 0 og σ ≈ 2 °C ved transporttidshorisonten.

    Predikert temperatur
    --------------------
    For hvert fremtidig tidspunkt t brukes faktisk Vorma-obs der data finnes
    (≤ 2 timers toleranse), ellers ekstrapoleres siste anomali eksponentielt
    (τ = 36 t). Lineær blending over transporttidsvinduet forankrer starten
    i siste observerte Fetsund-temperatur for å unngå hopp.
    """
    MODEL_SIGMA = 2.0

    travel_h, _, _ = calculate_travel_time(discharge_df)

    if vorma_df is None or vorma_df.empty:
        return pd.DataFrame()
    if fetsund_df is None or fetsund_df.empty:
        return pd.DataFrame()

    fe = fetsund_df.copy()
    fe['time'] = pd.to_datetime(fe['time'])
    if fe['time'].dt.tz is None:
        fe['time'] = fe['time'].dt.tz_localize('UTC')
    fe = fe.sort_values('time')

    vo = vorma_df.copy()
    vo['time'] = pd.to_datetime(vo['time'])
    if vo['time'].dt.tz is None:
        vo['time'] = vo['time'].dt.tz_localize('UTC')
    vo = vo.sort_values('time')

    last_fetsund_obs  = fe.iloc[-1]['value']
    last_fetsund_time = fe.iloc[-1]['time']
    last_vorma_time   = vo.iloc[-1]['time']

    # Vorma-baseline: median av siste 48 timer
    cutoff_48h = last_vorma_time - timedelta(hours=48)
    baseline   = vo.loc[vo['time'] >= cutoff_48h, 'value'].median()
    if np.isnan(baseline):
        baseline = vo['value'].median()

    last_vorma_anomaly = vo.iloc[-1]['value'] - baseline

    future_times = pd.date_range(
        start=last_fetsund_time,
        periods=hours_ahead // step_h + 1,
        freq=f'{step_h}h',
        tz='UTC',
    )

    rows = []
    for t_fut in future_times:
        h_elapsed = (t_fut - last_fetsund_time).total_seconds() / 3600

        # ── Vorma-oppslag ─────────────────────────────────────────────────
        vorma_lookup = t_fut - timedelta(hours=travel_h)
        time_diffs   = (vo['time'] - vorma_lookup).abs()
        nearest_idx  = time_diffs.idxmin()
        gap_h        = time_diffs[nearest_idx].total_seconds() / 3600

        if gap_h <= 2.0:
            anomaly = vo.loc[nearest_idx, 'value'] - baseline
        else:
            # Eksponentiell demping av anomali der Vorma-data mangler
            extrap_h = max(0.0, (vorma_lookup - last_vorma_time).total_seconds() / 3600)
            anomaly  = last_vorma_anomaly * np.exp(-extrap_h / 36.0)

        raw_pred = baseline + anomaly * TEMPERATURE_SURVIVAL

        # ── Blending: siste Fetsund-obs → modellprediksjon ───────────────
        alpha = min(1.0, h_elapsed / travel_h)
        pred  = last_fetsund_obs * (1.0 - alpha) + raw_pred * alpha

        # ── Usikkerhet ────────────────────────────────────────────────────
        ramp   = min(1.0, h_elapsed / travel_h)
        extrap = max(0.0, h_elapsed - travel_h)
        sigma  = MODEL_SIGMA * ramp * np.sqrt(1.0 + extrap / 24.0)

        rows.append({
            'time':      t_fut,
            'predicted': round(pred,          2),
            'lower_68':  round(pred - sigma,        2),
            'upper_68':  round(pred + sigma,        2),
            'lower_95':  round(pred - 1.96 * sigma, 2),
            'upper_95':  round(pred + 1.96 * sigma, 2),
        })

    return pd.DataFrame(rows)


def _forecast_chart(fetsund_obs_df, forecast_df, travel_hours,
                    title="Temperaturprognose – Fetsund"):
    """
    Kombinert Plotly-graf: historiske Fetsund-målinger + prediksjon med
    usikkerhetsbånd (68 % og 95 % KI). Båndenebegynner i null-bredde ved
    siste observasjon og vokser med tid.
    """
    fig = go.Figure()

    # ── Fargebakgrunn per risikonivå ──────────────────────────────────────────
    risk_zones = [
        (24, 28, "rgba(23,162,184,0.07)"),
        (20, 24, "rgba(40,167,69,0.09)"),
        (18, 20, "rgba(240,165,0,0.09)"),
        (16, 18, "rgba(200,100,0,0.12)"),
        (14, 16, "rgba(220,53,69,0.12)"),
        ( 8, 14, "rgba(107,0,0,0.13)"),
    ]
    for y0, y1, color in risk_zones:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0, layer="below")

    threshold_labels = {
        14: "14 °C – farlig",
        16: "16 °C – FINA min.",
        18: "18 °C",
        20: "20 °C",
        24: "24 °C – varmt",
    }
    for temp, label in threshold_labels.items():
        fig.add_hline(
            y=temp,
            line_dash="dot",
            line_color="rgba(110,110,110,0.28)",
            line_width=0.8,
            annotation_text=label,
            annotation_position="right",
            annotation_font_size=10,
            annotation_font_color="rgba(110,110,110,0.65)",
        )

    # ── Prediksjonsintervaller ────────────────────────────────────────────────
    if forecast_df is not None and not forecast_df.empty:
        t_fwd = list(forecast_df['time'])
        t_rev = list(forecast_df['time'])[::-1]

        # 95 % KI (lyseblå fyll)
        fig.add_trace(go.Scatter(
            x         = t_fwd + t_rev,
            y         = list(forecast_df['upper_95']) + list(forecast_df['lower_95'])[::-1],
            fill      = 'toself',
            fillcolor = 'rgba(56,141,228,0.10)',
            line      = dict(color='rgba(0,0,0,0)', width=0),
            name      = '95 % KI',
            hoverinfo = 'skip',
        ))

        # 68 % KI (middelblå fyll)
        fig.add_trace(go.Scatter(
            x         = t_fwd + t_rev,
            y         = list(forecast_df['upper_68']) + list(forecast_df['lower_68'])[::-1],
            fill      = 'toself',
            fillcolor = 'rgba(56,141,228,0.22)',
            line      = dict(color='rgba(0,0,0,0)', width=0),
            name      = '68 % KI',
            hoverinfo = 'skip',
        ))

        # Predikert midtlinje
        fig.add_trace(go.Scatter(
            x          = forecast_df['time'],
            y          = forecast_df['predicted'],
            mode       = 'lines',
            name       = 'Prediksjon',
            line       = dict(color='#185FA5', width=2, dash='dash'),
            customdata = forecast_df[['lower_68', 'upper_68',
                                      'lower_95', 'upper_95']].values,
            hovertemplate=(
                '<b>Prediksjon</b>: %{y:.1f} °C<br>'
                '68 % KI: %{customdata[0]:.1f}–%{customdata[1]:.1f} °C<br>'
                '95 % KI: %{customdata[2]:.1f}–%{customdata[3]:.1f} °C'
                '<extra></extra>'
            ),
        ))

        # Vertikale markørlinjer
        now_ms     = pd.Timestamp.now(tz='UTC').timestamp() * 1000
        horizon_ms = (pd.Timestamp.now(tz='UTC') +
                      timedelta(hours=travel_hours)).timestamp() * 1000

        fig.add_vline(
            x=now_ms,
            line_dash='dot', line_color='rgba(100,100,100,0.50)', line_width=1,
            annotation_text='Nå',
            annotation_position='top left',
            annotation_font_size=11,
            annotation_font_color='rgba(100,100,100,0.80)',
        )
        fig.add_vline(
            x=horizon_ms,
            line_dash='dot', line_color='rgba(56,141,228,0.45)', line_width=1,
            annotation_text=f'Datahorisont (+{travel_hours:.0f} t)',
            annotation_position='top right',
            annotation_font_size=10,
            annotation_font_color='rgba(56,141,228,0.75)',
        )

    # ── Historiske Fetsund-målinger (lagt til sist → øverst i stack) ──────────
    if fetsund_obs_df is not None and not fetsund_obs_df.empty:
        fig.add_trace(go.Scatter(
            x             = fetsund_obs_df['time'],
            y             = fetsund_obs_df['value'],
            mode          = 'lines',
            name          = 'Observert (Fetsund)',
            line          = dict(color='#185FA5', width=2),
            hovertemplate = '<b>Observert</b>: %{y:.1f} °C<extra></extra>',
        ))

    fig.update_layout(
        title      = title,
        xaxis_title = '',
        yaxis      = dict(title='°C', range=[10, 28], fixedrange=True),
        height     = 430,
        hovermode  = 'x unified',
        template   = 'plotly_white',
        margin     = dict(l=50, r=90, t=50, b=40),
        legend     = dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='center', x=0.5, font=dict(size=10),
        ),
    )
    return fig


# ============================================================================
# WIND ENERGY FUNCTIONS
# ============================================================================

def build_wind_energy_series(frost_df, forecast_df, window_hours=168):
    """
    Beregner rullende 7-dagers kumulativ SE/S-vindenergi E(t).

    Formel:  E(t) = Σ  v_i × Δtᵢ   for alle obs i [t − window_hours, t]
                        der vindretning ∈ [WIND_SECTOR_MIN, WIND_SECTOR_MAX]

    Δtᵢ (timer mellom observasjoner) håndterer automatisk at Frost API er
    timesbasert (Δt≈1) mens Met.no-varselet er 6-timers etter dag 3 (Δt≈6).
    Terskel er 210 m·h uavhengig av dataoppløsning (= CERRA 70 m·h med dt=3).

    Broen mellom historikk og prognose er sømløs: begge kildene leverer
    wind_speed og wind_direction, og formelen brukes identisk.

    Returnerer DataFrame:
        time, wind_speed, wind_direction, v_ses, dt,
        E, E_upper, E_lower, is_forecast
    """
    rows = []

    if frost_df is not None and not frost_df.empty:
        df_f = frost_df.copy()
        df_f['time'] = pd.to_datetime(df_f['time'])
        if df_f['time'].dt.tz is None:
            df_f['time'] = df_f['time'].dt.tz_localize('UTC')
        for _, r in df_f.sort_values('time').iterrows():
            rows.append({
                'time':           r['time'],
                'wind_speed':     float(r.get('wind_speed', 0) or 0),
                'wind_direction': float(r.get('wind_direction', 0) or 0),
                'is_forecast':    False,
            })

    last_obs_time = (rows[-1]['time'] if rows
                     else pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=1))

    if forecast_df is not None and not forecast_df.empty:
        df_fc = forecast_df.copy()
        df_fc['time'] = pd.to_datetime(df_fc['time'])
        if df_fc['time'].dt.tz is None:
            df_fc['time'] = df_fc['time'].dt.tz_localize('UTC')
        for _, r in df_fc[df_fc['time'] > last_obs_time].sort_values('time').iterrows():
            rows.append({
                'time':           r['time'],
                'wind_speed':     float(r.get('wind_speed', 0) or 0),
                'wind_direction': float(r.get('wind_direction', 0) or 0),
                'is_forecast':    True,
            })

    if len(rows) < 2:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values('time').reset_index(drop=True)

    # Δt per observasjon (timer) – håndterer variabel oppløsning
    df['dt'] = (df['time'].diff().dt.total_seconds().fillna(3600) / 3600).clip(0.5, 12)

    # SE/S-komponent og energibidrag
    in_sector   = ((df['wind_direction'] >= WIND_SECTOR_MIN) &
                   (df['wind_direction'] <= WIND_SECTOR_MAX))
    df['v_ses']     = np.where(in_sector, df['wind_speed'], 0.0)
    df['e_contrib'] = df['v_ses'] * df['dt']

    # Rullende sum over tidsvindusperioden
    df_idx     = df.set_index('time')
    df_idx['E'] = (df_idx['e_contrib']
                   .rolling(f'{window_hours}h', min_periods=1)
                   .sum()
                   .round(2))
    df = df_idx.reset_index()

    # Usikkerhetsbånd på prognosen – vokser som √(tid fremover)
    now_utc  = pd.Timestamp.now(tz='UTC')
    max_fc_h = 120.0
    df['E_upper'] = df['E']
    df['E_lower'] = df['E']

    fc_mask = df['is_forecast'].values
    if fc_mask.any():
        h_ahead = ((df.loc[fc_mask, 'time'] - now_utc)
                   .dt.total_seconds().div(3600).clip(lower=0).values)
        unc = 25.0 * np.sqrt(h_ahead / max_fc_h)
        df.loc[fc_mask, 'E_upper'] = np.round(df.loc[fc_mask, 'E'].values + unc, 2)
        df.loc[fc_mask, 'E_lower'] = np.round(
            np.maximum(0, df.loc[fc_mask, 'E'].values - unc), 2)

    return df


def _wind_energy_chart(energy_df,
                       title="Kumulativ SE/S-vindenergi – oppvellingsrisiko"):
    """
    To-panel Plotly-graf:
      Panel 1 – Rullende 7-dagers E (observert + prognose + usikkerhetsbånd)
                med risikosoner, terskel (210 m·h) og advarselsnivå (140 m·h).
      Panel 2 – SE/S vindhastighetsstolper per tidssteg (obs + prognose).

    Spesielle markører:
      • «Nå»-linje der historikk slutter og prognose begynner.
      • «Gammel hendelse faller ut»-linje: tidspunktet der den eldste store
        SE/S-hendelsen forlater det 7-dagers rullende vinduet.
    """
    if energy_df is None or energy_df.empty:
        return None

    obs = energy_df[~energy_df['is_forecast']].copy()
    fc  = energy_df[ energy_df['is_forecast']].copy()
    now_utc = pd.Timestamp.now(tz='UTC')

    # Koble observert og prognose slik at linjen er kontinuerlig
    if not obs.empty and not fc.empty:
        bridge = obs.iloc[-1:].copy()
        bridge['is_forecast'] = True
        fc = pd.concat([bridge, fc]).reset_index(drop=True)

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.08,
        shared_xaxes=True,
        subplot_titles=(
            'Akkumulert SE/S-vindenergi – rullende 7-dagers vindu',
            'SE/S vindstyrke per tidssteg',
        ),
    )

    # ── Risikosoner (panel 1) ─────────────────────────────────────────────────
    fig.add_hrect(y0=ENERGY_THRESHOLD, y1=310,
                  fillcolor='rgba(220,53,69,0.09)',  line_width=0, row=1, col=1)
    fig.add_hrect(y0=ENERGY_WARN, y1=ENERGY_THRESHOLD,
                  fillcolor='rgba(239,159,39,0.09)', line_width=0, row=1, col=1)
    fig.add_hrect(y0=0, y1=ENERGY_WARN,
                  fillcolor='rgba(40,167,69,0.07)',  line_width=0, row=1, col=1)

    fig.add_hline(y=ENERGY_THRESHOLD,
                  line_dash='dot', line_color='rgba(163,45,45,0.55)', line_width=1.2,
                  annotation_text='210 m·h – terskel',
                  annotation_position='right',
                  annotation_font_size=10,
                  annotation_font_color='rgba(163,45,45,0.75)',
                  row=1, col=1)
    fig.add_hline(y=ENERGY_WARN,
                  line_dash='dot', line_color='rgba(186,117,23,0.45)', line_width=1.0,
                  annotation_text='140 m·h – advarsel',
                  annotation_position='right',
                  annotation_font_size=10,
                  annotation_font_color='rgba(186,117,23,0.70)',
                  row=1, col=1)

    # ── Usikkerhetsbånd (polygon-fill, panel 1) ───────────────────────────────
    if not fc.empty:
        t_fwd = list(fc['time'])
        t_rev = list(fc['time'])[::-1]
        fig.add_trace(go.Scatter(
            x=t_fwd + t_rev,
            y=list(fc['E_upper']) + list(fc['E_lower'])[::-1],
            fill='toself',
            fillcolor='rgba(56,141,228,0.13)',
            line=dict(color='rgba(0,0,0,0)', width=0),
            name='Usikkerhet (±1σ)',
            hoverinfo='skip',
        ), row=1, col=1)

    # ── E-kurve: observert (panel 1) ──────────────────────────────────────────
    if not obs.empty:
        fig.add_trace(go.Scatter(
            x=obs['time'], y=obs['E'],
            mode='lines', name='Observert E (Frost)',
            line=dict(color='#185FA5', width=2),
            hovertemplate='<b>E (obs)</b>: %{y:.1f} m·h<extra></extra>',
        ), row=1, col=1)

    # ── E-kurve: prognose (panel 1) ───────────────────────────────────────────
    if not fc.empty:
        fig.add_trace(go.Scatter(
            x=fc['time'], y=fc['E'],
            mode='lines', name='Prognosert E (Met.no)',
            line=dict(color='#185FA5', width=2, dash='dash'),
            hovertemplate='<b>E (varsel)</b>: %{y:.1f} m·h<extra></extra>',
        ), row=1, col=1)

    # ── «Nå»-markør ───────────────────────────────────────────────────────────
    now_ms = now_utc.timestamp() * 1000
    for row in [1, 2]:
        fig.add_vline(x=now_ms,
                      line_dash='dot',
                      line_color='rgba(100,100,100,0.45)',
                      line_width=1,
                      annotation_text='Nå' if row == 1 else '',
                      annotation_position='top left',
                      annotation_font_size=11,
                      annotation_font_color='rgba(100,100,100,0.75)',
                      row=row, col=1)

    # ── «Gammel hendelse faller ut»-markør ────────────────────────────────────
    # Finn eldste tidspunkt med signifikant SE/S-bidrag og legg til 7 dager
    if not obs.empty:
        big_ses = obs[obs['e_contrib'] > 1.5]
        if not big_ses.empty:
            rolloff_time = big_ses.iloc[0]['time'] + pd.Timedelta(hours=168)
            if rolloff_time > now_utc:
                ro_ms = rolloff_time.timestamp() * 1000
                fig.add_vline(x=ro_ms,
                              line_dash='dot',
                              line_color='rgba(186,117,23,0.40)',
                              line_width=1,
                              annotation_text='gammel hendelse faller ut',
                              annotation_position='top right',
                              annotation_font_size=10,
                              annotation_font_color='rgba(186,117,23,0.70)',
                              row=1, col=1)

    # ── SE/S-vindstolper (panel 2) ────────────────────────────────────────────
    if not obs.empty:
        fig.add_trace(go.Bar(
            x=obs['time'], y=obs['v_ses'],
            name='SE/S vind (obs)',
            marker_color='rgba(239,159,39,0.55)',
            hovertemplate='%{y:.1f} m/s<extra></extra>',
        ), row=2, col=1)

    if not fc.empty:
        fig.add_trace(go.Bar(
            x=fc['time'], y=fc['v_ses'],
            name='SE/S vind (varsel)',
            marker_color='rgba(239,159,39,0.25)',
            hovertemplate='%{y:.1f} m/s<extra></extra>',
        ), row=2, col=1)

    fig.add_hline(y=CRITICAL_WIND_SPEED,
                  line_dash='dot',
                  line_color='rgba(163,45,45,0.40)',
                  line_width=1,
                  annotation_text=f'{CRITICAL_WIND_SPEED} m/s',
                  annotation_position='right',
                  annotation_font_size=10,
                  annotation_font_color='rgba(163,45,45,0.65)',
                  row=2, col=1)

    fig.update_layout(
        title     = title,
        hovermode = 'x unified',
        template  = 'plotly_white',
        height    = 540,
        margin    = dict(l=50, r=120, t=50, b=40),
        barmode   = 'overlay',
        legend    = dict(orientation='h', yanchor='bottom', y=1.02,
                         xanchor='center', x=0.5, font=dict(size=10)),
    )
    fig.update_yaxes(title_text='E (m·h)', range=[0, 290], row=1, col=1)
    fig.update_yaxes(title_text='m/s',                     row=2, col=1)
    fig.update_xaxes(showticklabels=True,                  row=2, col=1)

    return fig


# ============================================================================
# PAGE: INFORMASJON
# ============================================================================

def page_informasjon():
    st.title("Om temperaturvarsel for svømming i Glomma og GlommaDyppen")

    st.markdown("""
    GlommaDyppen er et Open Water arrangement fra Bingsfossen til Fetsund lenser
    langs Glomma, arrangert av Fetsund Lenser, Fet Svømmeklubb og Sørumsand IF Svømmegruppe den første lørdagen i august hvert år.
    Distansen på den lengste øvelsen, Fløter'n, er 11 km og gjennomføres uansett vær – men temperaturen i vannet
    kan variere mye fra år til år hvilket påvirker sikkerheten.

    Denne siden er laget for å gi arrangører og deltakere bedre grunnlag for å
    planlegge arrangement og treningsturer.
    """)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌊 Hvorfor varierer temperaturen?")
        st.markdown("""
        Kaldt vann fra Mjøsas dyplag kan nå Glomma ved Fetsund gjennom en kjede av hendelser:

        1. **Sørøst/sørlig vind** over en viss hastighet og tid ved sørenden av Mjøsa skaper Ekman-transport som presser overflatevann
           mot nordenden av innsjøen og drar kaldt bunnvann (hypolimnion) opp i sør.
        2. Det kalde vannet strømmer ut i **Vorma ved Minnesund** og transporteres
           sørover i elven.
        3. Etter **ca. 25 timer** (ved typisk augustvannføring) når det kalde vannet
           **samløpet med Glomma** nedenfor Funnefoss.
        4. Her blandes det med Glomma-vannet: bare **~14 %** av temperaturavviket
           overlever fortynningen/dispersjon og når Fetsund.

        Effekten kan gi temperaturfall på 3–5 °C ved arrangementet ved
        kraftig og vedvarende sørøst/sørlig vind.
        """)

    with col2:
        st.subheader("📡 Datakilder")
        st.markdown("""
        **NVE HydAPI** – timesverdier for vanntemperatur og vannføring:
        - Svanefoss (2.52.0) — Vorma, vanntemperatur, ca. 22 km nedenfor Mjøsa
        - Ertesekken (2.197.0) — Vorma, vannføring, ca. 21 km nedenfor Mjøsa
        - Funnefoss (2.410.0) — Glomma, vannføring og vanntemperatur, ca. 7 km ovenfor samløpet med Vorma
        - Blaker (2.17.0) — Glomma, vanntemperatur og vannføring, ca, 21 km nedenfor samløpet
        - Fetsund (2.587.0) — Glomma, vanntemperatur, Målgang / arrangementspunkt

        **MET Frost API** – historiske vindmålinger fra Kise (SN12680) ved søndre Mjøsa.

        **Met.no Locationforecast 2.0** – værvarsler for Mjøsa og Fetsund lenser,
        oppdateres ca. hver time.
        """)

    st.divider()

    st.subheader("🔮 Hva betyr prediksjonen – og når er den pålitelig?")
    st.markdown("""
    Prediksjonsmodellen beregner forventet vanntemperatur ved Fetsund basert på
    **aktuelle målinger i Vorma** og transporttidsformelen **t = 9700 / Q** (timer).
    """)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        **Prediksjonen er pålitelig når:**
        - Det er aktive målinger fra Svanefoss eller Funnefoss (april–september)
        - Du ønsker å vite omtrent hva temperaturen er ved Fetsund **i dag eller i morgen**
        - Det er innen **1–2 uker** før GlommaDyppen eller i sommermånedene for treningssvømming
        """)
    with col4:
        st.markdown("""
        **Prediksjonen er *ikke* en langtidsprognose:**
        - Mange måneder før arrangementet reflekterer prediksjonen kun **nåværende
          forhold**, ikke hva som vil skje i august
        - Usikkerheten i prediksjonen er ±2–3 °C (95 % KI)
        """)

    st.divider()

    st.subheader("🏊 Våtdrakt-regler (World Aquatics / FINA)")
    st.markdown("""
    | Temperatur | Vurdering | Våtdrakt |
    |---|---|---|
    | < 14 °C | Svømming bør ikke gjennomføres | Ikke aktuelt |
    | 14–16 °C | Høy risiko – vurder avlysning | Obligatorisk |
    | 16–18 °C | Moderat risiko | Sterkt anbefalt |
    | 18–20 °C | Lav risiko | Anbefalt |
    | 20–24 °C | Gode forhold | Valgfritt |
    | > 24 °C | Varmt vann | Frarådes |

    Disse grensene gjelder internasjonale konkurranser i åpent vann. Arrangøren
    kan sette strengere krav, og individuelle deltakere bør vurdere egen erfaring
    og toleranse for kaldt vann uavhengig av regelverket.
    """)


# ============================================================================
# PAGE: PREDIKSJON
# ============================================================================

def page_prediksjon():
    st.title("Temperaturprediksjon")
    st.markdown(
        "Predikert vanntemperatur i Glomma basert på observasjoner i Mjøsa, Vorma og Glomma. "
        "Prediksjonen er laget for GlommaDyppen, men kan benyttes for andre aktiviteter "
        "i Glomma i sommermånedene."
    )

    event_date = calculate_event_date(EVENT_YEAR)
    days_until = (event_date - pd.Timestamp.now(tz='UTC')).days
    oslo_dt    = event_date.tz_convert('Europe/Oslo')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Neste arrangement", oslo_dt.strftime("%d. %B %Y"))
    c2.metric("Dager igjen",       f"{days_until} dager")
    c3.metric("Starttid",          "10:00")
    c4.metric("Stasjon",
              "🟢 Aktiv" if 4 <= datetime.now().month <= 9 else "🔴 Offline (vinter)")

    st.divider()

    with st.spinner("Henter data…"):
        vorma_temp     = fetch_nve_data(STATION_FUNNEFOSS_TEMP, 1003, hours_back=168)
        svanefoss_temp = fetch_nve_data(STATION_SVANEFOSS,      1003, hours_back=168)
        blaker_temp    = fetch_nve_data(STATION_BLAKER,         1003, hours_back=168)
        fetsund_temp   = fetch_nve_data(STATION_FETSUND,        1003, hours_back=168)
        ertesekken_q   = fetch_nve_data(STATION_ERTESEKKEN_Q,   1001, hours_back=168)
        frost_vind     = fetch_frost_wind(hours_back=168)
        weather_mjosa  = fetch_weather_forecast(MJOSA_LAT, MJOSA_LON)
        if not weather_mjosa.empty:
            weather_mjosa = add_southerly_component(weather_mjosa)

    # ── Station offline check ─────────────────────────────────────────────────
    if vorma_temp.empty and svanefoss_temp.empty:
        st.warning("""
        ⚠️ **Målestasjon offline (vintersesong)**

        Vorma-stasjonene er offline. Forventes aktive igjen april 2026.
        """)
        if not weather_mjosa.empty:
            st.subheader("Vindvarsel Mjøsa")
            chart = _wind_forecast_chart(weather_mjosa.head(120), "Vindvarsel – Mjøsa (5 dager)")
            if chart:
                st.plotly_chart(chart, use_container_width=True)
        st.stop()

    primary_df = svanefoss_temp if not svanefoss_temp.empty else vorma_temp

    latest_time    = pd.to_datetime(primary_df.iloc[-1]['time'])
    if latest_time.tz is None:
        latest_time = latest_time.tz_localize('UTC')
    data_age_hours = (pd.Timestamp.now(tz='UTC') - latest_time).total_seconds() / 3600
    data_age_days  = data_age_hours / 24

    if data_age_days > 7:
        st.warning(
            f"⚠️ Siste Vorma-måling er {data_age_days:.1f} dager gammel – "
            "stasjonen kan være offline."
        )

    # ── Nåstatus ─────────────────────────────────────────────────────────────
    st.header("Nåværende status")
    c1, c2, c3, c4 = st.columns(4)

    latest_val = primary_df.iloc[-1]['value']
    delta_24   = (f"{latest_val - primary_df.iloc[-24]['value']:+.1f} °C (24t)"
                  if len(primary_df) >= 24 else "–")
    c1.metric("Vorma nå", f"{latest_val:.1f} °C", delta=delta_24)

    drop = detect_temperature_drop(primary_df, threshold_C=2.0, window_hours=6)
    if drop:
        c2.metric("Temperaturfall (6t)", f"{drop['magnitude']:.1f} °C",
                  delta="⚠️ Detektert!", delta_color="inverse")
    else:
        c2.metric("Temperaturfall (6t)", "Ingen", delta="✓ Stabilt")

    if not weather_mjosa.empty:
        cw = weather_mjosa.iloc[0]
        c3.metric("Vind (Mjøsa)", f"{cw['wind_speed']:.1f} m/s",
                  delta=f"{cw['wind_direction']:.0f}° ({wind_rose_label(cw['wind_direction'])})")
    else:
        c3.metric("Vind (Mjøsa)", "N/A")

    t_hours, q_val, q_src = calculate_travel_time(ertesekken_q)
    c4.metric("Transporttid nå", f"{t_hours} t",
              help=f"t = 9700 / {q_val:.0f} m³/s ({q_src})")

    st.divider()

    # ── Prediksjon for arrangementet ──────────────────────────────────────────
    st.header("Prediksjon for arrangementet")

    travel_hours_now, _, _ = calculate_travel_time(ertesekken_q)
    pred_valid_to  = (pd.Timestamp.now(tz='UTC') +
                      pd.Timedelta(hours=travel_hours_now)).tz_convert('Europe/Oslo')

    if days_until > 14:
        st.info(
            f"ℹ️ Prediksjonen viser nåværende forhold, ikke en prognose for august. "
            f"Det er {days_until} dager til GlommaDyppen "
            f"({oslo_dt.strftime('%-d. %B %Y')}). "
            f"Prediksjonen er gyldig frem til ca. "
            f"{pred_valid_to.strftime('%-d. %b kl %H:%M')} "
            f"(neste {travel_hours_now:.0f} timer)."
        )
    else:
        st.success(
            f"✅ Prediksjon for arrangementet er aktiv ({days_until} dager igjen). "
            f"Basert på Vorma-temperatur {travel_hours_now:.0f} timer før start."
        )

    prediction = predict_fetsund_temperature(primary_df, ertesekken_q, event_date)

    if data_age_days > 30 and days_until > 30:
        st.info(
            "Sanntidsprediksjon krever ferske Vorma-målinger. "
            "Aktiveres når stasjonen starter opp igjen (april 2026)."
        )
    elif prediction:
        pred_temp  = prediction['predicted_temp']
        sigma      = 2.0
        lb         = pred_temp - 1.96 * sigma
        ub         = pred_temp + 1.96 * sigma
        risk_label, risk_color, ws_label, ws_color, risk_details = \
            assess_risk_open_water(pred_temp, weather_mjosa)

        # ── Kompakt prediksjonsbrikke ─────────────────────────────────────
        st.markdown(
            f"""
            <div style="
                display: flex;
                align-items: center;
                gap: 20px;
                border-left: 5px solid {risk_color};
                border-radius: 8px;
                padding: 12px 18px;
                background: {risk_color}14;
                margin-bottom: 12px;
            ">
                <div style="text-align:center; min-width:72px;">
                    <div style="font-size:2.0em; font-weight:700;
                                color:{risk_color}; line-height:1.1;">
                        {pred_temp:.1f}°C
                    </div>
                    <div style="font-size:11px; color:#888; margin-top:2px;">
                        Fetsund
                    </div>
                </div>
                <div style="flex:1; min-width:0;">
                    <div style="font-weight:600; font-size:0.95em;">
                        {risk_label}
                    </div>
                    <div style="font-size:0.82em; color:#666; margin-top:3px;">
                        🏊 {ws_label}
                        &nbsp;·&nbsp;
                        95&nbsp;%&nbsp;KI:&nbsp;{lb:.1f}–{ub:.1f}&nbsp;°C
                        &nbsp;·&nbsp;
                        Transporttid:&nbsp;{prediction['travel_hours']:.0f}&nbsp;t
                        &nbsp;({prediction['q_source']})
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Vis detaljer og risikovurdering", expanded=False):
            for d in risk_details:
                st.markdown(f"- {d}")
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Vorma (oppstrøms)",  f"{prediction['vorma_temp']:.1f} °C")
            c2.metric("Baseline (48t)",      f"{prediction['baseline_temp']:.1f} °C")
            c3.metric("Transporttid brukt",  f"{prediction['travel_hours']} t",
                      help=f"t = 9700 / {prediction['q_used']:.0f} m³/s")
            c4.metric("Pålitelighet",        f"{prediction['confidence']*100:.0f} %")
            st.caption(
                "⚠️ Modellen er validert mot data fra juli og august. "
                "Bruk med forsiktighet utenfor sommermånedene."
            )
    else:
        st.warning("⚠️ Ikke nok data for prediksjon.")

    st.divider()

    # ── Temperaturprognose (historikk + fremtid) ──────────────────────────────
    st.subheader("Temperaturprognose – Fetsund")

    forecast_df = build_fetsund_forecast(primary_df, fetsund_temp, ertesekken_q)
    travel_h_now, _, _ = calculate_travel_time(ertesekken_q)

    if not forecast_df.empty:
        fig_fc = _forecast_chart(fetsund_temp, forecast_df, travel_h_now)
        st.plotly_chart(fig_fc, use_container_width=True)
        st.caption(
            "Solid linje: observert · Stiplet linje: prediksjon · "
            "68 % og 95 % KI starter med bredde 0 ved siste observasjon. "
            "Datahorisonten markerer der Vorma-observasjoner gir direkte grunnlag "
            f"(σ ≈ 2 °C). Etter dette ekstrapoleres Vorma-anomalien med "
            "eksponentiell demping og usikkerheten vokser tilsvarende."
        )
    else:
        st.warning("Ikke nok data for prognosevisning.")

    # ── Vind og oppvellingsrisiko ─────────────────────────────────────────────
    if not weather_mjosa.empty or not frost_vind.empty:
        st.divider()
        st.subheader("Vind og oppvellingsrisiko – Mjøsa")

        energy_df = build_wind_energy_series(frost_vind, weather_mjosa)

        # ── Metrics ──────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)

        if not energy_df.empty:
            obs_e  = energy_df[~energy_df['is_forecast']]
            fc_e   = energy_df[ energy_df['is_forecast']]
            cur_E  = float(obs_e['E'].iloc[-1]) if not obs_e.empty else 0.0
            pct    = round(cur_E / ENERGY_THRESHOLD * 100)
            fc_E   = float(fc_e['E'].iloc[-1])       if not fc_e.empty else cur_E
            fc_Ehi = float(fc_e['E_upper'].max())    if not fc_e.empty else cur_E

            c1.metric("Kumulativ E nå",   f"{cur_E:.1f} m·h",
                      help="Rullende 7-dagers sum av SE/S-vindenergi (Frost API)")
            c2.metric("Andel av terskel", f"{pct} %",
                      help="210 m·h = 100 % (timesdata-konvensjon = CERRA 70 m·h × 3)")
            c3.metric("Prognosert E (dag +5)", f"{fc_E:.1f} m·h",
                      delta="⚠️ Kan overskride terskel!" if fc_Ehi >= ENERGY_THRESHOLD else None,
                      delta_color="inverse" if fc_Ehi >= ENERGY_THRESHOLD else "normal")
        else:
            c1.metric("Kumulativ E nå", "N/A")
            c2.metric("Andel av terskel", "N/A")
            c3.metric("Prognosert E (dag +5)", "N/A")

        if not weather_mjosa.empty:
            avg_ses = weather_mjosa.head(48)['southerly_wind'].mean()
            if avg_ses >= CRITICAL_WIND_SPEED:
                c4.metric("SE/S-vind (48t)", f"{avg_ses:.1f} m/s",
                          delta="⚠️ Oppvellings-risiko!", delta_color="inverse")
            else:
                c4.metric("SE/S-vind (48t)", f"{avg_ses:.1f} m/s")

        # ── Faner: energikurve | vindretning ─────────────────────────────────
        wind_tabs = st.tabs(["Kumulativ oppvellingsrisiko", "Vindretning og -hastighet"])

        with wind_tabs[0]:
            if not energy_df.empty:
                fig_e = _wind_energy_chart(energy_df)
                if fig_e:
                    st.plotly_chart(fig_e, use_container_width=True)
                st.caption(
                    "E = Σ v_i × Δtᵢ for alle obs der vindretning ∈ 135–225° (SE/S), "
                    "rullende 7-dagers vindu · "
                    "Terskel 210 m·h (timesdata) = CERRA-kalibrert 70 m·h · "
                    "Usikkerheten vokser som √(tid fremover). "
                    "Markøren «gammel hendelse faller ut» viser når siste store "
                    "SE/S-episode forlater vinduet og E synker naturlig."
                )
            else:
                st.warning("Vindenergi-beregning krever Frost API-data.")

        with wind_tabs[1]:
            chart = _wind_forecast_chart(weather_mjosa.head(120), "Vindvarsel – Mjøsa")
            if chart:
                st.plotly_chart(chart, use_container_width=True)


# ============================================================================
# PAGE: DATA & VARSEL
# ============================================================================

def page_data_varsel():
    st.title("Observasjoner og Værvarsler")
    st.markdown(
        "Faktiske målinger fra NVE og met.no. "
        "Bruk denne siden for å se rå data og standard værvarsler."
    )

    tabs = st.tabs([
        "🌊 NVE Vanntemperatur",
        "💧 NVE Vannføring",
        "🌬️ Vind ved Mjøsa",
        "🌤️ Værvarsler",
    ])

    with st.spinner("Henter observasjoner…"):
        sv_temp    = fetch_nve_data(STATION_SVANEFOSS,      1003, hours_back=168)
        fn_temp    = fetch_nve_data(STATION_FUNNEFOSS_TEMP, 1003, hours_back=168)
        bl_temp    = fetch_nve_data(STATION_BLAKER,         1003, hours_back=168)
        fe_temp    = fetch_nve_data(STATION_FETSUND,        1003, hours_back=168)
        er_q       = fetch_nve_data(STATION_ERTESEKKEN_Q,   1001, hours_back=168)
        bl_q       = fetch_nve_data(STATION_BLAKER,         1001, hours_back=168)
        fn_q       = fetch_nve_data(STATION_FUNNEFOSS_Q,    1001, hours_back=168)
        frost_vind = fetch_frost_wind(hours_back=168)
        fc_mjosa   = fetch_weather_forecast(MJOSA_LAT,   MJOSA_LON)
        fc_fetsund = fetch_weather_forecast(FETSUND_LAT, FETSUND_LON)

    # ── TAB 1: Vanntemperatur ─────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Vanntemperatur – siste 7 dager (NVE HydAPI)")
        st.caption("Timesverdier fra stasjonene langs Vorma og Glomma.")

        c1, c2, c3, c4 = st.columns(4)
        def _latest(df, label, col):
            if df.empty: col.metric(label, "N/A")
            else:        col.metric(label, f"{df.iloc[-1]['value']:.1f} °C")
        _latest(sv_temp, "Svanefoss (Vorma)", c1)
        _latest(fn_temp, "Funnefoss (Vorma)", c2)
        _latest(bl_temp, "Blaker (Glomma)",   c3)
        _latest(fe_temp, "Fetsund (Glomma)",  c4)

        fig = _temp_chart({
            'Svanefoss': sv_temp, 'Funnefoss': fn_temp,
            'Blaker':    bl_temp, 'Fetsund':   fe_temp,
        }, "Vanntemperatur – siste 7 dager")
        st.plotly_chart(fig, use_container_width=True)

        st.caption("""
        **Stasjoner:**
        Svanefoss (2.52.0) i Vorma 22 km fra Mjøsa ·
        Funnefoss (2.410.0) i Vorma 23,5 km fra Mjøsa ·
        Blaker (2.17.0) i Glomma nedenfor samløp ·
        Fetsund (2.587.0) målgang GlommaDyppen.
        """)

    # ── TAB 2: Vannføring ─────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Vannføring – siste 7 dager (NVE HydAPI)")
        st.caption("Timesverdier i m³/s. Ertesekken brukes for transporttid t = 9700/Q.")

        c1, c2, c3 = st.columns(3)
        def _latest_q(df, label, col):
            if df.empty:
                col.metric(label, "N/A")
            else:
                v = df.iloc[-1]['value']
                t = round(TRANSPORT_COEFF / v, 1) if v > 0 else None
                col.metric(label, f"{v:.0f} m³/s",
                           help=f"Transporttid ≈ {t} t" if t else None)
        _latest_q(er_q, "Ertesekken (Vorma)", c1)
        _latest_q(fn_q, "Funnefoss (Glomma)", c2)
        _latest_q(bl_q, "Blaker (Glomma)",    c3)

        fig = _discharge_chart({
            'Ertesekken': er_q, 'Funnefoss': fn_q, 'Blaker': bl_q,
        }, "Vannføring – siste 7 dager")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Transporttid-kalkulator")
        q_now = er_q.iloc[-1]['value'] if not er_q.empty else FALLBACK_DISCHARGE
        q_val = st.slider("Vannføring ved Ertesekken (m³/s)",
                          min_value=100, max_value=1200,
                          value=int(q_now), step=10)
        t_calc = round(TRANSPORT_COEFF / q_val, 1)
        st.info(
            f"**t = 9700 / {q_val} = {t_calc} timer** (Svanefoss → Fetsund, 45 km)  \n"
            "*(t = 6871 / Q for Svanefoss → Blaker)*"
        )

    # ── TAB 3: Vind ved Mjøsa ─────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Vindmålinger – Kise, søndre Mjøsa (siste 7 dager)")
        st.caption(f"Kilde: MET.no Frost API · Stasjon {FROST_STATION_KISE} · Timesverdier.")

        if frost_vind.empty:
            st.warning("Vindmålinger fra Frost API ikke tilgjengelig.")
        else:
            if 'wind_direction' in frost_vind.columns:
                is_ses    = ((frost_vind['wind_direction'] >= WIND_SECTOR_MIN) &
                             (frost_vind['wind_direction'] <= WIND_SECTOR_MAX))
                avg_ses   = frost_vind.loc[is_ses, 'wind_speed'].mean() if is_ses.any() else 0.0
                ses_hours = int(is_ses.sum())

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Vindhastighet nå",    f"{frost_vind.iloc[-1]['wind_speed']:.1f} m/s")
                c2.metric("Gj.snitt total (7d)", f"{frost_vind['wind_speed'].mean():.1f} m/s")
                c3.metric("Timer SE/S-vind",      f"{ses_hours} t")
                if avg_ses >= CRITICAL_WIND_SPEED:
                    c4.metric("Gj.snitt SE/S", f"{avg_ses:.1f} m/s",
                              delta="⚠️ Over terskel", delta_color="inverse")
                else:
                    c4.metric("Gj.snitt SE/S (kun SE/S-timer)", f"{avg_ses:.1f} m/s")

            chart = _wind_obs_chart(frost_vind, f"Vindmålinger – {FROST_STATION_KISE} Kise")
            if chart:
                st.plotly_chart(chart, use_container_width=True)

            with st.expander("Vis rådata (Frost API)", expanded=False):
                disp = frost_vind.copy()
                disp['time'] = (disp['time'].dt.tz_convert('Europe/Oslo')
                                            .dt.strftime('%Y-%m-%d %H:%M'))
                st.dataframe(disp, use_container_width=True)

    # ── TAB 4: Værvarsler ─────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Værvarsler – Met.no Locationforecast (opp til 10 dager)")
        st.caption(
            "Kilde: Met.no Locationforecast 2.0 · Oppdateres ca. hver time · "
            "Timesoppløsning de første 3 dagene, deretter 6-timers intervaller."
        )

        col_mjosa, col_fetsund = st.columns(2)

        with col_mjosa:
            st.markdown("### 📍 Søndre Mjøsa / Kise")
            st.caption("60.78°N, 10.72°E – referansepunkt for oppvellingsanalyse")
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
            st.caption("59.93°N, 11.58°E – arrangementspunkt")
            if fc_fetsund.empty:
                st.warning("Varsel ikke tilgjengelig")
            else:
                tbl = _daily_forecast_table_fetsund(fc_fetsund)
                if tbl is not None:
                    st.dataframe(tbl, use_container_width=True, hide_index=True)
                chart = _weather_fetsund_chart(fc_fetsund, "Værvarsler – Fetsund lenser")
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
        st.markdown(
            '<a href="https://glommadyppen.no" target="_blank">'
            + '<img src="data:image/jpeg;base64,{}" style="width:100%;cursor:pointer;">'
            .format(__import__('base64').b64encode(
                open('Samensatt_logo_GlommDyppen.jpg', 'rb').read()).decode())
            + '</a>',
            unsafe_allow_html=True
        )
        st.markdown("---")
        page = st.radio(
            "Navigasjon",
            options=["Om siden", "Observasjoner og værvarsel", "Prediksjon"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("""
        **Modell**
        - Transporttid = 9700 / Q (dynamisk)
        - Innvirkning av temp. i Vorma: 14 %
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
            "Utviklet av Fet Svømmeklubb for GlommaDyppen.no"
        )

    if page == "Om siden":
        page_informasjon()
    elif page == "Observasjoner og værvarsel":
        page_data_varsel()
    else:
        page_prediksjon()


if __name__ == "__main__":
    main()
