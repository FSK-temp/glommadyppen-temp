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
STATION_FUNNEFOSS_Q      = "2.412.0"   # Glomma / Funnefoss kraftverk
STATION_FETSUND          = "2.587.0"   # Fetsund bru – temperatur (målpunkt)

# ── Frost (met.no observations) ──────────────────────────────────────────────
FROST_STATION_KISE = "SN12680"   # Kise, søndre Mjøsa

# ── Met.no koordinater ───────────────────────────────────────────────────────
MJOSA_LAT,       MJOSA_LON       = 60.78,   10.72    # Kise, søndre Mjøsa
BINGSFOSSEN_LAT, BINGSFOSSEN_LON = 60.2172, 11.5528  # Start
FETSUND_LAT,     FETSUND_LON     = 59.9297, 11.5833  # Mål / Fetsund lenser

# ── Modellparametere ─────────────────────────────────────────────────────────
# Transporttid beregnes dynamisk som t = TRANSPORT_COEFF / Q  (timer)
# 9700 er empirisk bestemt som Svanefoss → Fetsund (45 km), R²=0.73, n=19
TRANSPORT_COEFF      = 9700     # m / (m³/s) → timer; Svanefoss→Fetsund
TRANSPORT_COEFF_BLA  = 6871     # m / (m³/s) → timer; Svanefoss→Blaker
FALLBACK_DISCHARGE   = 437.0    # m³/s – median august; brukes kun hvis Q-data mangler
TEMPERATURE_SURVIVAL = 0.14     # 14 % av temperaturfall overlever fortynning og dispersjon
CRITICAL_WIND_SPEED  = 1.9      # m/s vedvarende sørlig vind for å utløse oppvelling
WIND_SECTOR_MIN      = 135      # Kritisk vindretning fra (°)
WIND_SECTOR_MAX      = 225      # Kritisk vindretning til (°)

# ── Open Water temperaturgrenser ─────────────────────────────────────────────
# Basert på World Aquatics (FINA) OW-regler og norske sikkerhetsterskler
OW_ABORT            = 14.0   # Under: svømming i Glomma bør ikke gjennomføres
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
    """
    Henter data fra NVE HydAPI.
    Parameter-koder (NVE exdat-format):
        1001 = vassføring (m³/s)
        1003 = vanntemperatur (°C)
    """
    try:
        url = f"{NVE_BASE_URL}/Observations"
        headers = {"X-API-Key": NVE_API_KEY, "accept": "application/json"} if NVE_API_KEY else {"accept": "application/json"}
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

        # Inkluder ukontrollerte data (quality=0) i tillegg til godkjente (1) og korrigerte (2).
        # NVE-data blir sjelden etterkontrollert i sanntid, så quality=0 er normalt
        # for ferske målinger. Fysisk sanity-sjekk nedenfor er tilstrekkelig filter.
        if 'quality' in df.columns:
            df = df[df['quality'].isin([0, 1, 2])]

        # Fysisk sanity-sjekk: forkast umulige verdier
        # (Svanefoss har kjente sensorfeil med verdier rundt -20 °C med quality=1)
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
        df = df.rename(columns={'wind_from_direction': 'wind_direction'})
        return df

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=21600)
def fetch_weather_forecast(lat, lon, days_ahead=14):
    """Henter varsel fra Met.no Locationforecast (opp til ~10 dager med timesoppløsning)."""
    try:
        url     = "https://api.met.no/weatherapi/locationforecast/2.0/complete"
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
            # Nedbør hentes fra next_1_hours eller next_6_hours
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


def calculate_travel_time(discharge_df):
    """
    Beregner transporttid Svanefoss → Fetsund som t = 9700 / Q (timer).

    Bruker medianen av siste 24 timers observasjoner for å dempe korttidssvingninger.
    Faller tilbake til august-medianen (437 m³/s → 22 t) hvis data mangler.

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

    # Fallback
    q = FALLBACK_DISCHARGE
    return round(TRANSPORT_COEFF / q, 1), q, f"august-median ({FALLBACK_DISCHARGE:.0f} m³/s)"


def predict_fetsund_temperature(vorma_temp_df, discharge_df, event_datetime):
    """
    Predikerer Fetsund-temperatur for arrangementet.

    Transporttid beregnes dynamisk som t = 9700 / Q (Svanefoss → Fetsund),
    der Q er medianen av siste 24 timers målinger ved Ertesekken.
    14 % av temperaturavviket i Vorma overlever transport og dispersjon frem til Fetsund.
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
        'predicted_temp':  fetsund_temp,
        'vorma_temp':      vorma_temp,
        'baseline_temp':   baseline_temp,
        'anomaly':         anomaly,
        'vorma_time':      vorma_time,
        'travel_hours':    travel_hours,
        'q_used':          q_used,
        'q_source':        q_source,
        'confidence':      _calculate_confidence(df, prediction_time),
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
        label  = "Svømming i Glomma bør ikke gjennomføres"
        color  = "#6B0000"
        ws     = "Ikke aktuelt — for kaldt"
        ws_col = "#6B0000"
        details = [
            f"Predikert temperatur {predicted_temp:.1f} °C er under absolutt minimumsgrense (14 °C).",
            "World Aquatics (FINA) forbyr konkurranser under 16 °C.",
            "Hypotermirisiko er ekstremt høy — svømming bør ikke gjennomføres.",
        ]

    elif predicted_temp < OW_WETSUIT_REQUIRED:
        label  = "Høy risiko – vurder svømming nøye"
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
        is_ses = (df['wind_direction'] >= WIND_SECTOR_MIN) & (df['wind_direction'] <= WIND_SECTOR_MAX)
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
        d = df[df['date'] == date]
        avg_s = d['southerly_wind'].mean()
        risiko_ikon = "🔴" if avg_s >= CRITICAL_WIND_SPEED else \
                      "🟡" if avg_s >= 1.2 else "🟢"
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
    """
    Daglig sammendragstabell for Fetsund lenser – viser vær relevant for
    arrangementet: temperatur, vind, nedbør. Ingen oppvellings-analyse
    (SE/S-vind er ikke relevant her).
    """
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
    """Enkel kombinert graf: lufttemperatur + vind + nedbør for Fetsund."""
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
# PAGE: INFORMASJON
# ============================================================================

def page_informasjon():
    st.title("Om Glommadyppen Temperaturvarsel")

    st.markdown("""
    Glommadyppen er et åpent vannsvømmearrangement fra Bingsfossen til Fetsund lenser
    langs Glomma, arrangert av Fet Svømmeklubb den første lørdagen i august hvert år.
    Distansen er ca. 14 km og gjennomføres uansett vær – men temperaturen i vannet
    kan variere mye fra år til år, og påvirker både sikkerhet og regelverk for våtdrakt.

    Denne siden er laget for å gi arrangører og deltakere bedre grunnlag for å
    planlegge arrangement og treningsturer.
    """)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌊 Hvorfor varierer temperaturen?")
        st.markdown("""
        Kaldt vann fra Mjøsas dyplag kan nå Glomma ved Fetsund gjennom en kjede av hendelser:

        1. **Sørøst/sør-vind** over Mjøsa skaper Ekman-transport som presser overflatevann
           mot nordenden av innsjøen og drar kaldt bunnvann (hypolimnion) opp i sør.
        2. Det kalde vannet strømmer ut i **Vorma ved Minnesund** og transporteres
           sørover i elven.
        3. Etter **ca. 25 timer** (ved typisk augustvannføring) når det kalde vannet
           **samløpet med Glomma** nedenfor Funnefoss.
        4. Her blandes det med Glomma-vannet: bare **~14 %** av temperaturavviket
           overlever fortynningen/dispersjon og når Fetsund.

        Effekten kan likevel gi temperaturfall på 3–5 °C ved arrangementet i år med
        kraftig og vedvarende sørlig vind.
        """)

    with col2:
        st.subheader("📡 Datakilder")
        st.markdown("""
        **NVE HydAPI** – timesverdier for vanntemperatur og vannføring:
        - Svanefoss (2.52.0) — Vorma, 22 km fra Mjøsa
        - Funnefoss (2.410.0) — Vorma, 23,5 km fra Mjøsa
        - Ertesekken (2.197.0) — Vorma, vannføring (brukes i transporttidsmodellen)
        - Blaker (2.17.0) — Glomma, nedenfor samløpet
        - Fetsund (2.587.0) — Målgang / arrangementspunkt

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
          – for eksempel for en treningstur langs arrangementsdistansen
        - Det er innen **1–2 uker** før Glommadyppen
        """)
    with col4:
        st.markdown("""
        **Prediksjonen er *ikke* en langtidsprognose:**
        - Mange måneder før arrangementet reflekterer prediksjonen kun **nåværende
          forhold**, ikke hva som vil skje i august
        - Prediksjonen bør tolkes som *«slik er det nå»*, ikke *«slik blir det
          under Glommadyppen»* dersom det er langt til arrangement
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
    st.markdown("Predikert vanntemperatur under Glommadyppen basert på observasjoner i Mjøsa, Vorma og Glomma")

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
        st.warning(f"⚠️ Siste Vorma-måling er {data_age_days:.1f} dager gammel – stasjonen kan være offline.")

    # ── Nåstatus ─────────────────────────────────────────────────────────────
    st.header("Nåværende status")
    c1, c2, c3, c4 = st.columns(4)

    latest_val = primary_df.iloc[-1]['value']
    delta_24   = f"{latest_val - primary_df.iloc[-24]['value']:+.1f} °C (24t)" if len(primary_df) >= 24 else "–"
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

    # Vis aktuell transporttid som fjerde metric
    t_hours, q_val, q_src = calculate_travel_time(ertesekken_q)
    c4.metric("Transporttid nå", f"{t_hours} t",
              help=f"t = 9700 / {q_val:.0f} m³/s ({q_src})")

    st.divider()

    # ── Prediksjon ────────────────────────────────────────────────────────────
    st.header("Prediksjon for arrangementet")

    # Beregn prediksjonsvindu og vis tydelig hvilken periode som gjelder
    travel_hours_now, q_now_pred, _ = calculate_travel_time(ertesekken_q)
    pred_valid_from = pd.Timestamp.now(tz='UTC')
    pred_valid_to   = pred_valid_from + pd.Timedelta(hours=travel_hours_now)
    pred_from_oslo  = pred_valid_from.tz_convert('Europe/Oslo')
    pred_to_oslo    = pred_valid_to.tz_convert('Europe/Oslo')

    if days_until > 14:
        st.info(f"""
        ℹ️ **Prediksjonen viser nåværende forhold – ikke en prognose for august**

        Prediksjonen er basert på Vorma-målinger fra **nå**, og reflekterer forventet
        temperatur ved Fetsund i løpet av de neste **{travel_hours_now:.0f} timene**
        (dvs. frem til ca. {pred_to_oslo.strftime('%d.%m kl %H:%M')}).

        Det er {days_until} dager til Glommadyppen ({oslo_dt.strftime('%-d. %B %Y')}).
        Prediksjonen for selve arrangementsdagen vil først være meningsfull
        ca. **1–2 uker før** arrangementet.
        """)
    else:
        st.success(f"""
        ✅ **Prediksjon for arrangementet** er nå aktiv ({days_until} dager igjen).
        Prediksjonen er basert på Vorma-temperaturen {travel_hours_now:.0f} timer
        før arrangementet ({oslo_dt.strftime('%-d. %B kl %H:%M')}).
        """)

    prediction = predict_fetsund_temperature(primary_df, ertesekken_q, event_date)

    if data_age_days > 30 and days_until > 30:
        st.info("""
        Sanntidsprediksjon krever ferske Vorma-målinger.
        Prediksjonen aktiveres når stasjonen starter opp igjen (april 2026).
        """)
    elif prediction:
        pred_temp = prediction['predicted_temp']
        risk_label, risk_color, ws_label, ws_color, risk_details = \
            assess_risk_open_water(pred_temp, weather_mjosa)

        st.markdown(f"""
        <div style='background:{risk_color}; padding:20px; border-radius:12px;
                    color:white; text-align:center; margin-bottom:16px;'>
          <div style='font-size:0.85em; opacity:0.8; margin-bottom:2px;'>
            Predikert temperatur ved Fetsund
          </div>
          <div style='font-size:3em; font-weight:700; line-height:1.1;'>
            {pred_temp:.1f} °C
          </div>
          <div style='font-size:1.1em; margin-top:8px; font-weight:600;'>
            {risk_label}
          </div>
          <div style='font-size:0.8em; opacity:0.85; margin-top:6px;'>
            Gjelder: {pred_from_oslo.strftime('%-d. %b kl %H:%M')} –
            {pred_to_oslo.strftime('%-d. %b kl %H:%M')}
            (neste {travel_hours_now:.0f} timer)
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:{ws_color}; padding:10px 16px; border-radius:8px;
                    color:white; display:inline-block; font-weight:600;
                    font-size:1.05em; margin-bottom:16px;'>
          🏊 Våtdrakt: {ws_label}
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Vis risikovurdering og Open Water-regler (draft)", expanded=False):
            for d in risk_details:
                st.markdown(f"- {d}")
            st.markdown("---")
            st.markdown("""
            **Temperaturgrenser (World Aquatics / FINA):**
            | Temp | Vurdering | Våtdrakt |
            |------|-----------|----------|
            | < 14 °C | Svømming bør ikke gjennomføres | Ikke aktuelt |
            | 14–16 °C | Høy risiko | Obligatorisk |
            | 16–18 °C | Moderat risiko | Sterkt anbefalt |
            | 18–20 °C | Lav risiko | Anbefalt |
            | 20–24 °C | Gode forhold | Valgfritt |
            | > 24 °C | Varmt vann | Frarådes |
            """)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Vorma (oppstrøms)", f"{prediction['vorma_temp']:.1f} °C")
        c2.metric("Baseline (48t snitt)", f"{prediction['baseline_temp']:.1f} °C")
        c3.metric("Transporttid brukt",
                  f"{prediction['travel_hours']} t",
                  help=f"t = 9700 / {prediction['q_used']:.0f} m³/s — {prediction['q_source']}")
        c4.metric("Pålitelighet", f"{prediction['confidence']*100:.0f} %")

        std_err = 2.0
        margin  = std_err * 1.96
        st.info(f"""
        **95 % konfidensintervall:** {pred_temp - margin:.1f} – {pred_temp + margin:.1f} °C

        Modell: t = 9700 / {prediction['q_used']:.0f} = **{prediction['travel_hours']} t** transporttid
        ({prediction['q_source']}) · 14 % kaldtvann · {len(primary_df)} målinger
        """)
        st.warning("⚠️ Modellen er validert opp mot data fra juli og august. Bruk med forsiktighet utenfor sommermånedene.")
    else:
        st.warning("⚠️ Ikke nok data for prediksjon.")

    st.divider()

    st.subheader("Temperaturhistorikk – siste 7 dager")
    temp_fig = _temp_chart({
        'Svanefoss': svanefoss_temp,
        'Blaker':    blaker_temp,
        'Fetsund':   fetsund_temp,
    }, "Vanntemperatur (siste 7 dager)")
    st.plotly_chart(temp_fig, use_container_width=True)

    if not weather_mjosa.empty:
        st.divider()
        st.subheader("Vindvarsel – Mjøsa (5 dager)")
        next_48h = weather_mjosa.head(48)
        avg_wind = next_48h['wind_speed'].mean()
        max_wind = next_48h['wind_speed'].max()
        avg_ses  = next_48h['southerly_wind'].mean()

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
        st.caption("Timesverdier fra stasjonene langs Vorma og Glomma. Bare data med de to høyeste kvalitetene vises.")

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

    # ── TAB 2: Vannføring ─────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Vannføring – siste 7 dager (NVE HydAPI)")
        st.caption("Timesverdier i m³/s. Vannføring forbi Ertesekken brukes for å beregne transporttid t = 9700/Q.")

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
            'Ertesekken': er_q,
            'Funnefoss':  fn_q,
            'Blaker':     bl_q,
        }, "Vannføring – siste 7 dager")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Transporttid-kalkulator")
        q_now = er_q.iloc[-1]['value'] if not er_q.empty else FALLBACK_DISCHARGE
        q_val = st.slider("Vannføring ved Ertesekken (m³/s)",
                          min_value=100, max_value=1200,
                          value=int(q_now), step=10)
        t_calc = round(TRANSPORT_COEFF / q_val, 1)
        st.info(f"""
        **t = 9700 / {q_val} = {t_calc} timer** (Svanefoss → Fetsund, 45 km)  
        *(t = 6871 / Q for Svanefoss → Blaker)*
        """)

        st.caption("""
        **Stasjoner:**  
        - **Ertesekken** (2.197.0) — Vorma, nøkkelstasjon for transporttidsmodellen.  
        - **Funnefoss nedre** (2.279.0) — Glomma, oppstrøms samløp med Vorma.  
        - **Blaker** (2.17.0) — Glomma, nedenfor samløp (typisk 1,45 × Ertesekken).
        """)

    # ── TAB 3: Vind ved Mjøsa ─────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Vindmålinger – Kise, søndre Mjøsa (siste 7 dager)")
        st.caption(f"Kilde: MET.no Frost API · Stasjon {FROST_STATION_KISE} (Kise) · Timesverdier.")

        if frost_vind.empty:
            st.warning("Vindmålinger fra Frost API ikke tilgjengelig.")
        else:
            if 'wind_direction' in frost_vind.columns:
                is_ses    = ((frost_vind['wind_direction'] >= WIND_SECTOR_MIN) &
                             (frost_vind['wind_direction'] <= WIND_SECTOR_MAX))
                avg_ses   = frost_vind.loc[is_ses, 'wind_speed'].mean() if is_ses.any() else 0.0
                ses_hours = int(is_ses.sum())

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Vindhastighet nå",       f"{frost_vind.iloc[-1]['wind_speed']:.1f} m/s")
                c2.metric("Gj.snitt total (7d)",    f"{frost_vind['wind_speed'].mean():.1f} m/s")
                c3.metric("Timer SE/S-vind",         f"{ses_hours} t")
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
                disp['time'] = disp['time'].dt.tz_convert('Europe/Oslo').dt.strftime('%Y-%m-%d %H:%M')
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
        # Klikkbar logo → glommadyppen.no
        st.markdown(
            '<a href="https://glommadyppen.no" target="_blank">'
            + '<img src="data:image/jpeg;base64,{}" style="width:100%;cursor:pointer;">'
            .format(__import__('base64').b64encode(open('Samensatt_logo_GlommDyppen.jpg','rb').read()).decode())
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
            "Utviklet av Fet Svømmeklubb for Glommadyppen.no"
        )

    if page == "Om siden":
        page_informasjon()
    elif page == "Observasjoner og værvarsel":
        page_data_varsel()
    else:
        page_prediksjon()


if __name__ == "__main__":
    main()
