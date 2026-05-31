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
STATION_SVANEFOSS        = "2.52.0"
STATION_FUNNEFOSS_TEMP   = "2.410.0"
STATION_ERTESEKKEN_Q     = "2.197.0"
STATION_BLAKER           = "2.17.0"
STATION_FUNNEFOSS_Q      = "2.412.0"
STATION_FETSUND          = "2.587.0"

# ── Frost (met.no observations) ──────────────────────────────────────────────
FROST_STATION_KISE = "SN12680"

# ── Met.no koordinater ───────────────────────────────────────────────────────
MJOSA_LAT,       MJOSA_LON       = 60.78,   10.72
BINGSFOSSEN_LAT, BINGSFOSSEN_LON = 60.2172, 11.5528
FETSUND_LAT,     FETSUND_LON     = 59.9297, 11.5833

# ── Modellparametere ─────────────────────────────────────────────────────────
# Transportkoeffisienter t = k / Q (timer), der Q = vannføring Ertesekken (m³/s)
TRANSPORT_COEFF         = 9700   # Svanefoss → Fetsund   (45,0 km) – empirisk kalibrert
TRANSPORT_COEFF_BLA     = 6871   # Svanefoss → Blaker    (31,8 km) – empirisk kalibrert (R²=0,73, n=19)
TRANSPORT_COEFF_FLOTERN = 7670   # Svanefoss → Fløter'n  (35,5 km) – avledet: 6871 × 35.5/31.8
FALLBACK_DISCHARGE      = 437.0  # August-median Ertesekken (m³/s)

TEMPERATURE_SURVIVAL = 0.63      # Empirisk fortynningskoeffisient Svanefoss→Fetsund
MODEL_SIGMA          = 2.0       # °C – prediksjonsstandardavvik ved transporttidshorisonten

# ── Vindenergi-konfigurasjon ──────────────────────────────────────────────────
WIND_SECTOR_MIN      = 135
WIND_SECTOR_MAX      = 225
WIND_WINDOW_HOURS    = 48
WIND_LEAD_HOURS      = 24
CRITICAL_WIND_SPEED  = 1.9       # m/s
ENERGY_THRESHOLD     = 70.0      # m·h – alarm
ENERGY_WARN          = 45.0      # m·h – advarsel

# ── Seiche-ettereffekt konfigurasjon ─────────────────────────────────────────
# Etter en bekreftet kald oppvellingsepisode ved Minnesund oscillerer
# sprangsjiktet i Mjøsa med ~8–9 dagers halvperiode (Thendrup 1978).
# Sekundær kaldepuls opptrer typisk 5–12 dager etter primær bunn.
# Validert mot 61 episoder 2015–2025: +22 pst.poeng sensitivitet, +15 FP (daglig).
SEICHE_WINDOW_START_DAYS = 5    # dager etter primær kaldbunn
SEICHE_WINDOW_END_DAYS   = 12   # dager etter primær kaldbunn
SEICHE_COLD_THRESHOLD    = 10.0 # °C – absolutt tak for å telle som "kald episode"
SEICHE_ANOMALY_MIN       = 3.0  # °C – minimum ΔT (bunn vs. 7-dagers baseline)

# ── Open Water temperaturgrenser (World Athletics / FINA) ────────────────────
OW_ABORT            = 14.0
OW_WETSUIT_REQUIRED = 16.0
OW_WETSUIT_STRONG   = 18.0
OW_WETSUIT_OPTIONAL = 20.0
OW_TOO_WARM         = 24.0

# ── Arrangement ──────────────────────────────────────────────────────────────
EVENT_YEAR        = 2026
EVENT_MONTH       = 8
EVENT_DAY_OF_WEEK = 5   # lørdag


# ============================================================================
# DATA FETCHING
# ============================================================================

@st.cache_data(ttl=3600)
def fetch_nve_data(station_id, parameter, hours_back=168):
    """
    Henter data fra NVE HydAPI.
    Parameter-koder: 1001 = vassføring (m³/s), 1003 = vanntemperatur (°C)
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
    Beregner transporttid fra Svanefoss til Fløter'n og Fetsund.

    Koeffisienter (t = k / Q, timer):
        Fløter'n (35,5 km):  k = 7670  (avledet fra empirisk 6871 × 35.5/31.8)
        Fetsund  (45,0 km):  k = 9700  (empirisk kalibrert mot 19 kalde episoder, R²=0.73)

    Returnerer (t_flotern, t_fetsund, q_used, source_label).
    """
    if discharge_df is not None and not discharge_df.empty:
        recent = discharge_df.copy()
        recent['time'] = pd.to_datetime(recent['time'])
        cutoff = recent['time'].max() - pd.Timedelta(hours=24)
        last24 = recent[recent['time'] >= cutoff]['value']
        if len(last24) > 0:
            q = last24.median()
            return (round(TRANSPORT_COEFF_FLOTERN / q, 1),
                    round(TRANSPORT_COEFF / q, 1),
                    round(q, 0),
                    "siste 24t (Ertesekken)")
    q = FALLBACK_DISCHARGE
    return (round(TRANSPORT_COEFF_FLOTERN / q, 1),
            round(TRANSPORT_COEFF / q, 1),
            q,
            f"august-median ({FALLBACK_DISCHARGE:.0f} m³/s)")


def detect_seiche_risk(vorma_df, hours_back_history=336):
    """
    Sjekker om det finnes en bekreftet kald oppvellingsepisode (ΔT ≥ 3 °C,
    bunn < 10 °C) ved Minnesund i perioden 5–12 dager tilbake i tid.

    Seiche-mekanisme (Thendrup 1978): etter at sørlig vind setter sprangsjiktet
    i Mjøsa på skrå, vil termoklinen oscillere frem og tilbake med ~8–9 dagers
    halvperiode når vinden avtar. Dette gir sekundære kaldpulser selv uten nytt
    vindpådriv, typisk 5–12 dager etter primær bunn.

    Validering 2015–2025 (daglig, Fetsund < 18 °C som "kaldt"):
        Modell A (kun vind):          Sensitivitet 0.70, F1 0.756
        Modell B (vind + seiche):     Sensitivitet 0.92, F1 0.876
        Seiche bidrar med +22 pst.p. sensitivitet og kun +15 FP-dager (av 682).

    Returnerer dict med:
        'active'         : bool – seiche-risiko er aktiv nå
        'episode_date'   : Timestamp eller None – dato for primær kaldbunn
        'episode_min_T'  : float – minimums-temperatur i episoden
        'episode_dT'     : float – ΔT (baseline − bunn)
        'days_ago'       : float – dager siden primær bunn
        'days_remaining' : float – dager til slutt på seiche-vindu (dag 12)
    """
    result = {
        'active': False,
        'episode_date':  None,
        'episode_min_T': None,
        'episode_dT':    None,
        'days_ago':      None,
        'days_remaining': None,
    }

    if vorma_df is None or vorma_df.empty:
        return result

    df = vorma_df.copy()
    df['time'] = pd.to_datetime(df['time'])
    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize('UTC')
    df = df.sort_values('time').reset_index(drop=True)

    now_utc = pd.Timestamp.now(tz='UTC')

    # Hent siste `hours_back_history` timer for å ha nok historikk til baseline
    cutoff = now_utc - timedelta(hours=hours_back_history)
    df = df[df['time'] >= cutoff].copy()
    if len(df) < 24:
        return result

    # Rullende 3h-gjennomsnitt for å dempe sensorstøy
    df = df.set_index('time')
    df['T_s'] = df['value'].rolling('3h', min_periods=1).mean()

    # Definer seiche-vinduet: [nå - 12 dager, nå - 5 dager]
    window_end   = now_utc - timedelta(days=SEICHE_WINDOW_START_DAYS)
    window_start = now_utc - timedelta(days=SEICHE_WINDOW_END_DAYS)

    window_data = df[(df.index >= window_start) & (df.index <= window_end)]
    if len(window_data) < 6:
        return result

    # Finn det absolutte minimumet i vinduet
    t_min_idx = window_data['T_s'].idxmin()
    T_min_val  = float(window_data.loc[t_min_idx, 'T_s'])

    # Absolutt temperaturkrav
    if T_min_val >= SEICHE_COLD_THRESHOLD:
        return result

    # Beregn baseline: 7-dagers median FØR episoden
    baseline_data = df[(df.index >= t_min_idx - timedelta(days=7)) &
                       (df.index <  t_min_idx - timedelta(hours=12))]
    if len(baseline_data) < 24:
        return result

    baseline = float(baseline_data['T_s'].median())
    dT       = baseline - T_min_val

    if dT < SEICHE_ANOMALY_MIN:
        return result

    days_ago       = (now_utc - t_min_idx).total_seconds() / 86400
    days_remaining = SEICHE_WINDOW_END_DAYS - days_ago

    result.update({
        'active':          True,
        'episode_date':    t_min_idx,
        'episode_min_T':   round(T_min_val, 1),
        'episode_dT':      round(dT, 1),
        'days_ago':        round(days_ago, 1),
        'days_remaining':  round(max(0.0, days_remaining), 1),
    })
    return result


def predict_fetsund_temperature(vorma_temp_df, discharge_df, event_datetime,
                                fetsund_temp_df=None):
    """
    Predikerer temperatur ved Fløter'n / Fetsund for arrangementet.

    Modell:
        T_pred = fetsund_baseline + (vorma_now - vorma_baseline) × κ

    der κ = TEMPERATURE_SURVIVAL (empirisk ≈ 0.63).
    Transporttid brukt: Fetsund (9700/Q) for prediksjonspunkt.
    Fløter'n-tid (7670/Q) rapporteres separat i returverdien.
    """
    if vorma_temp_df.empty:
        return None
    if event_datetime.tzinfo is None:
        event_datetime = event_datetime.replace(tzinfo=pd.Timestamp.now(tz='UTC').tzinfo)

    t_flotern, travel_hours, q_used, q_source = calculate_travel_time(discharge_df)
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

    vorma_baseline = df[df['time'] >= (vorma_time - timedelta(hours=48))]['value'].mean()
    anomaly        = vorma_temp - vorma_baseline

    if fetsund_temp_df is not None and not fetsund_temp_df.empty:
        fe = fetsund_temp_df.copy()
        fe['time'] = pd.to_datetime(fe['time'])
        if fe['time'].dt.tz is None:
            fe['time'] = fe['time'].dt.tz_localize('UTC')
        latest_fe = fe['time'].max()
        recent_fe = fe[fe['time'] >= latest_fe - timedelta(hours=48)]
        fetsund_baseline = recent_fe['value'].median()
        baseline_source  = "Fetsund 48t-median"
    else:
        fetsund_baseline = vorma_baseline + 1.5
        baseline_source  = "Vorma + 1.5 °C (Fetsund-data mangler)"

    fetsund_temp = fetsund_baseline + anomaly * TEMPERATURE_SURVIVAL

    return {
        'predicted_temp':       fetsund_temp,
        'vorma_temp':           vorma_temp,
        'vorma_baseline':       vorma_baseline,
        'fetsund_baseline':     fetsund_baseline,
        'baseline_source':      baseline_source,
        'anomaly':              anomaly,
        'vorma_time':           vorma_time,
        'travel_hours':         travel_hours,
        'travel_hours_flotern': t_flotern,
        'q_used':               q_used,
        'q_source':             q_source,
        'confidence':           _calculate_confidence(df, prediction_time),
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


def assess_risk_open_water(predicted_temp, weather_forecast=None,
                           seiche_risk=None):
    """
    Risikovurdering basert på World Athletics / FINA OW-regler og
    Glommadyppens lokale regler.

    Glommadyppen-regel: Våtdrakt er obligatorisk uansett temperatur.
    Unntak kan søkes arrangøren. Arrangøren følger World Athletics-terskler
    for vurdering av gjennomføring, men kan skjønnsmessig justere den nedre
    grensen basert på helhetsvurdering (vær, sikt, strøm, deltakermassen).

    seiche_risk: dict fra detect_seiche_risk() – legger til advarsel om
    sekundær kaldpuls dersom aktiv.
    """
    WETSUIT_ALWAYS = "🧥 Obligatorisk (Glommadyppen-regel)"
    WETSUIT_COLOR  = "#2c6e9e"

    southerly_risk = False
    if weather_forecast is not None and not weather_forecast.empty:
        df_wf = weather_forecast.copy()
        if 'southerly_wind' not in df_wf.columns:
            df_wf = add_southerly_component(df_wf)
        avg_s = df_wf.head(48)['southerly_wind'].mean()
        southerly_risk = avg_s >= CRITICAL_WIND_SPEED

    if predicted_temp < OW_ABORT:
        label, color = "Svømming bør ikke gjennomføres", "#6B0000"
        details = [
            f"Predikert temperatur {predicted_temp:.1f} °C — under absolutt minimumsgrense (14 °C).",
            "World Athletics forbyr konkurranser i åpent vann under 16 °C.",
            "Hypotermirisiko er svært høy — arrangementet bør ikke gjennomføres.",
            "Arrangøren har fullmakt til å avlyse basert på en helhetsvurdering.",
        ]
    elif predicted_temp < OW_WETSUIT_REQUIRED:
        label, color = "Høy risiko – vurder avlysning", "#dc3545"
        details = [
            f"Predikert temperatur {predicted_temp:.1f} °C — under World Athletics-minimum (16 °C).",
            "Arrangøren bør vurdere avlysning eller utsettelse.",
            "Vurderingen kan påvirkes av lufttemperatur, sol/skydekke og antatt svømmetid.",
        ]
    elif predicted_temp < OW_WETSUIT_STRONG:
        label, color = "Moderat risiko – kjølig vann", "#e07b00"
        details = [
            f"Predikert temperatur {predicted_temp:.1f} °C — World Athletics tillater gjennomføring.",
            "Arrangøren kan senke den nedre grensen noe ved gunstige forhold (sol, varm luft).",
        ]
    elif predicted_temp < OW_WETSUIT_OPTIONAL:
        label, color = "Lav risiko – friskt vann", "#f0a500"
        details = [
            f"Predikert temperatur {predicted_temp:.1f} °C — gode vilkår for langdistansesvømming.",
        ]
    elif predicted_temp < OW_TOO_WARM:
        label, color = "Gode forhold", "#28a745"
        details = [
            f"Predikert temperatur {predicted_temp:.1f} °C — ideelle vilkår.",
            "Selv ved disse temperaturene er våtdrakt obligatorisk i Glommadyppen for sikkerhetens skyld.",
        ]
    else:
        label, color = "Uvanlig varmt vann", "#17a2b8"
        details = [
            f"Predikert temperatur {predicted_temp:.1f} °C — varmere enn normalt for Glomma i august.",
            "Vanlig i Glomma er 16–22 °C. Over 24 °C er sjeldent.",
            "Kontakt arrangøren for vurdering — standard våtdraktpåbud gjelder inntil annet bestemmes.",
        ]

    # Felles merknad om Glommadyppen-regelen
    details.append(
        "🧥 Glommadyppen krever våtdrakt uansett temperatur av sikkerhetsmessige grunner. "
        "Unntak kan søkes arrangøren individuelt."
    )

    if southerly_risk:
        details.append(
            "⚠️ Vedvarende sørlig vind er varslet — temperaturfall fra Mjøsa-oppvelling er mulig."
        )

    if seiche_risk is not None and seiche_risk.get('active'):
        days_ago      = seiche_risk['days_ago']
        days_rem      = seiche_risk['days_remaining']
        ep_date_oslo  = seiche_risk['episode_date'].tz_convert(
            'Europe/Oslo').strftime('%-d. %b kl %H:%M')
        details.append(
            f"🌊 Seiche-ettereffekt aktiv: kald episode ved Minnesund "
            f"({seiche_risk['episode_min_T']:.1f} °C, ΔT={seiche_risk['episode_dT']:.1f} °C) "
            f"for {days_ago:.1f} dager siden ({ep_date_oslo}). "
            f"Forhøyet risiko for sekundær kaldpuls i ca. {days_rem:.0f} dager til "
            f"(sprangsjikt-oscillasjon i Mjøsa, ~8–9 dagers halvperiode)."
        )

    return label, color, WETSUIT_ALWAYS, WETSUIT_COLOR, details


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
    for temp, label, color in [(16, "16 °C – WA minimum", "red"),
                                (18, "18 °C", "orange"),
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


def _daily_forecast_table(df, days=10):
    if df.empty:
        return None
    df = df.copy()
    if 'southerly_wind' not in df.columns:
        df = add_southerly_component(df)
    df['date'] = pd.to_datetime(df['time']).dt.tz_convert('Europe/Oslo').dt.date
    rows = []
    for date in sorted(df['date'].unique())[:days]:
        d     = df[df['date'] == date]
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


# ============================================================================
# WIND ENERGY FUNCTIONS
# ============================================================================

def build_wind_energy_series(frost_df, forecast_df,
                             window_hours=None, lead_hours=None):
    """
    Beregner rullende kumulativ SE/S-vindenergi (E) som driver oppvelling.
    Standardverdier: window=48t, lead=24t – optimalt kalibrert mot 3500+ obs.
    """
    window_hours = window_hours or WIND_WINDOW_HOURS
    lead_hours   = lead_hours   or WIND_LEAD_HOURS

    combined_parts = []
    if frost_df is not None and not frost_df.empty:
        obs = frost_df.copy()
        obs['is_forecast'] = False
        combined_parts.append(obs)
    if forecast_df is not None and not forecast_df.empty:
        fc = forecast_df.copy()
        fc['is_forecast'] = True
        combined_parts.append(fc)
    if not combined_parts:
        return pd.DataFrame()

    df = pd.concat(combined_parts, ignore_index=True)
    df['time'] = pd.to_datetime(df['time'])
    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize('UTC')
    df = df.sort_values('time').reset_index(drop=True)

    if 'southerly_wind' not in df.columns:
        df = add_southerly_component(df)

    df['dt'] = df['time'].diff().dt.total_seconds().div(3600).fillna(1.0).clip(lower=0.5, upper=7.0)
    df['is_ses']    = ((df['wind_direction'] >= WIND_SECTOR_MIN) &
                       (df['wind_direction'] <= WIND_SECTOR_MAX))
    df['v_ses']     = np.where(df['is_ses'], df['wind_speed'], 0.0)
    df['e_contrib'] = df['v_ses'] * df['dt']

    # Dedupliser tidsstempler (kan oppstå i overlapp mellom Frost-obs og Met.no-prognose).
    # Behold siste rad per tidsstempel; obs ble lagt inn først og sort er stabil,
    # så 'last' gir faktisk observasjonen forrang fremfor prognose.
    df = df.drop_duplicates(subset='time', keep='last').reset_index(drop=True)

    df_idx = df.set_index('time')
    df_idx['E_raw'] = (df_idx['e_contrib']
                       .rolling(f'{window_hours}h', min_periods=1)
                       .sum())

    # shift(freq=...) krever unik datetimeindex. Etter dedup er dette garantert,
    # men som ekstra sikkerhet faller vi tilbake til integer-shift ved ValueError.
    try:
        df_idx['E'] = df_idx['E_raw'].shift(freq=f'{lead_hours}h').round(2)
    except ValueError:
        median_dt = float(df['dt'].median()) or 1.0
        n_shift   = max(1, round(lead_hours / median_dt))
        df_idx['E'] = df_idx['E_raw'].shift(n_shift).round(2)

    df = df_idx[['wind_speed', 'wind_direction', 'is_forecast',
                 'v_ses', 'e_contrib', 'dt', 'E']].reset_index()
    df['E'] = df['E'].fillna(0.0)

    now_utc  = pd.Timestamp.now(tz='UTC')
    max_fc_h = 120.0
    df['E_upper'] = df['E']
    df['E_lower'] = df['E']
    fc_mask = df['is_forecast'].values
    if fc_mask.any():
        h_ahead = ((df.loc[fc_mask, 'time'] - now_utc)
                   .dt.total_seconds().div(3600).clip(lower=0).values)
        unc = 12.0 * np.sqrt(h_ahead / max_fc_h)
        df.loc[fc_mask, 'E_upper'] = np.round(df.loc[fc_mask, 'E'].values + unc, 2)
        df.loc[fc_mask, 'E_lower'] = np.round(
            np.maximum(0, df.loc[fc_mask, 'E'].values - unc), 2)
    return df


def _wind_energy_chart(energy_df,
                       title="Kumulativ SE/S-vindenergi – oppvellingsrisiko"):
    if energy_df is None or energy_df.empty:
        return None

    obs = energy_df[~energy_df['is_forecast']].copy()
    fc  = energy_df[ energy_df['is_forecast']].copy()
    now_utc = pd.Timestamp.now(tz='UTC')

    if not obs.empty and not fc.empty:
        bridge = obs.iloc[-1:].copy()
        bridge['is_forecast'] = True
        fc = pd.concat([bridge, fc]).reset_index(drop=True)

    e_max = max(float(energy_df['E'].max()),
                float(energy_df['E_upper'].max() if 'E_upper' in energy_df else 0),
                ENERGY_THRESHOLD * 1.5)
    y_max = round(e_max * 1.15)

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.65, 0.35], vertical_spacing=0.08,
        shared_xaxes=True,
        subplot_titles=(
            'Akkumulert SE/S-vindenergi – 48 t vindu med 24 t lead',
            'SE/S vindstyrke per tidssteg',
        ),
    )

    fig.add_hrect(y0=ENERGY_THRESHOLD, y1=y_max,
                  fillcolor='rgba(220,53,69,0.09)',  line_width=0, row=1, col=1)
    fig.add_hrect(y0=ENERGY_WARN, y1=ENERGY_THRESHOLD,
                  fillcolor='rgba(239,159,39,0.09)', line_width=0, row=1, col=1)
    fig.add_hrect(y0=0, y1=ENERGY_WARN,
                  fillcolor='rgba(40,167,69,0.07)',  line_width=0, row=1, col=1)

    fig.add_hline(y=ENERGY_THRESHOLD,
                  line_dash='dot', line_color='rgba(163,45,45,0.55)', line_width=1.2,
                  annotation_text=f'{ENERGY_THRESHOLD:.0f} m·h – terskel',
                  annotation_position='right', annotation_font_size=10,
                  annotation_font_color='rgba(163,45,45,0.75)', row=1, col=1)
    fig.add_hline(y=ENERGY_WARN,
                  line_dash='dot', line_color='rgba(186,117,23,0.45)', line_width=1.0,
                  annotation_text=f'{ENERGY_WARN:.0f} m·h – advarsel',
                  annotation_position='right', annotation_font_size=10,
                  annotation_font_color='rgba(186,117,23,0.70)', row=1, col=1)

    if not fc.empty:
        t_fwd = list(fc['time'])
        t_rev = list(fc['time'])[::-1]
        fig.add_trace(go.Scatter(
            x=t_fwd + t_rev,
            y=list(fc['E_upper']) + list(fc['E_lower'])[::-1],
            fill='toself', fillcolor='rgba(56,141,228,0.13)',
            line=dict(color='rgba(0,0,0,0)', width=0),
            name='Usikkerhet (±1σ)', hoverinfo='skip',
        ), row=1, col=1)

    if not obs.empty:
        fig.add_trace(go.Scatter(
            x=obs['time'], y=obs['E'], mode='lines', name='E (Frost-obs)',
            line=dict(color='#185FA5', width=2),
            hovertemplate='<b>E (obs)</b>: %{y:.1f} m·h<extra></extra>',
        ), row=1, col=1)

    if not fc.empty:
        fig.add_trace(go.Scatter(
            x=fc['time'], y=fc['E'], mode='lines', name='E (Met.no-prognose)',
            line=dict(color='#185FA5', width=2, dash='dash'),
            hovertemplate='<b>E (varsel)</b>: %{y:.1f} m·h<extra></extra>',
        ), row=1, col=1)

    now_ms = now_utc.timestamp() * 1000
    for row in [1, 2]:
        fig.add_vline(x=now_ms, line_dash='dot', line_color='rgba(100,100,100,0.45)',
                      line_width=1,
                      annotation_text='Nå' if row == 1 else '',
                      annotation_position='top left', annotation_font_size=11,
                      annotation_font_color='rgba(100,100,100,0.75)', row=row, col=1)

    if not obs.empty:
        fig.add_trace(go.Bar(
            x=obs['time'], y=obs['v_ses'], name='SE/S vind (obs)',
            marker_color='rgba(239,159,39,0.55)',
            hovertemplate='%{y:.1f} m/s<extra></extra>',
        ), row=2, col=1)
    if not fc.empty:
        fig.add_trace(go.Bar(
            x=fc['time'], y=fc['v_ses'], name='SE/S vind (varsel)',
            marker_color='rgba(239,159,39,0.25)',
            hovertemplate='%{y:.1f} m/s<extra></extra>',
        ), row=2, col=1)

    fig.update_layout(title=title, height=520, showlegend=True,
                      yaxis=dict(range=[0, y_max]), **_LAYOUT_BASE)
    return fig


# ============================================================================
# FORECAST FUNCTIONS
# ============================================================================

def build_fetsund_forecast(vorma_df, fetsund_df, discharge_df,
                           hours_ahead=120, step_h=3):
    """
    Beregner tidsserie for predikert temperatur ved Fløter'n / Fetsund
    med usikkerhetsintervaller.

    Modell: T_pred(t) = fetsund_baseline + vorma_anomaly(t - travel_h) × κ
    Usikkerhet vokser lineært til travel_h, deretter som √(1 + (h-travel_h)/24).
    """
    if vorma_df is None or vorma_df.empty:
        return pd.DataFrame()

    _, travel_h, _, _ = calculate_travel_time(discharge_df)

    vorma_df = vorma_df.copy()
    vorma_df['time'] = pd.to_datetime(vorma_df['time'])
    if vorma_df['time'].dt.tz is None:
        vorma_df['time'] = vorma_df['time'].dt.tz_localize('UTC')

    fetsund_baseline = None
    last_fetsund_obs = None
    if fetsund_df is not None and not fetsund_df.empty:
        fe = fetsund_df.copy()
        fe['time'] = pd.to_datetime(fe['time'])
        if fe['time'].dt.tz is None:
            fe['time'] = fe['time'].dt.tz_localize('UTC')
        latest_fe = fe['time'].max()
        recent_fe = fe[fe['time'] >= latest_fe - timedelta(hours=48)]
        fetsund_baseline = recent_fe['value'].median()
        last_fetsund_obs = fe.iloc[-1]['value']

    if fetsund_baseline is None:
        vorma_base_val   = vorma_df.tail(48)['value'].mean()
        fetsund_baseline = vorma_base_val + 1.5
        last_fetsund_obs = fetsund_baseline

    vorma_baseline = vorma_df[
        vorma_df['time'] >= vorma_df['time'].max() - timedelta(hours=48)
    ]['value'].mean()

    now_utc = pd.Timestamp.now(tz='UTC')
    rows = []

    for h_step in range(0, hours_ahead + 1, step_h):
        t_fut    = now_utc + timedelta(hours=h_step)
        h_elapsed = h_step
        t_vorma  = t_fut - timedelta(hours=travel_h)

        close = vorma_df[abs(vorma_df['time'] - t_vorma) <= timedelta(hours=2)]
        if not close.empty:
            anomaly = close.iloc[-1]['value'] - vorma_baseline
        else:
            last_anomaly = vorma_df.iloc[-1]['value'] - vorma_baseline
            extrap_h     = max(0.0,
                               (t_vorma - vorma_df['time'].max()).total_seconds() / 3600)
            anomaly = last_anomaly * np.exp(-extrap_h / 36.0)

        raw_pred = fetsund_baseline + anomaly * TEMPERATURE_SURVIVAL
        alpha    = min(1.0, h_elapsed / travel_h)
        pred     = last_fetsund_obs * (1.0 - alpha) + raw_pred * alpha

        ramp   = min(1.0, h_elapsed / travel_h)
        extrap = max(0.0, h_elapsed - travel_h)
        sigma  = MODEL_SIGMA * ramp * np.sqrt(1.0 + extrap / 24.0)

        rows.append({
            'time':      t_fut,
            'predicted': round(pred,               2),
            'lower_68':  round(pred - sigma,        2),
            'upper_68':  round(pred + sigma,        2),
            'lower_95':  round(pred - 1.96 * sigma, 2),
            'upper_95':  round(pred + 1.96 * sigma, 2),
        })

    return pd.DataFrame(rows)


def _forecast_chart(fetsund_obs_df, forecast_df, travel_hours,
                    title="Temperaturprognose – Fløter'n / Fetsund"):
    """Kombinert graf: historiske Fetsund-målinger + prediksjon med KI."""
    fig = go.Figure()

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

    for temp, label in {14: "14 °C – farlig", 16: "16 °C – WA min.",
                        18: "18 °C", 20: "20 °C", 24: "24 °C – varmt"}.items():
        fig.add_hline(y=temp, line_dash="dot", line_color="rgba(110,110,110,0.28)",
                      line_width=0.8, annotation_text=label, annotation_position="right",
                      annotation_font_size=10, annotation_font_color="rgba(110,110,110,0.65)")

    if forecast_df is not None and not forecast_df.empty:
        t_fwd = list(forecast_df['time'])
        t_rev = list(forecast_df['time'])[::-1]
        fig.add_trace(go.Scatter(
            x=t_fwd + t_rev,
            y=list(forecast_df['upper_95']) + list(forecast_df['lower_95'])[::-1],
            fill='toself', fillcolor='rgba(56,141,228,0.10)',
            line=dict(color='rgba(0,0,0,0)', width=0),
            name='95 % KI', hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=t_fwd + t_rev,
            y=list(forecast_df['upper_68']) + list(forecast_df['lower_68'])[::-1],
            fill='toself', fillcolor='rgba(56,141,228,0.22)',
            line=dict(color='rgba(0,0,0,0)', width=0),
            name='68 % KI', hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['time'], y=forecast_df['predicted'],
            mode='lines', name='Prediksjon',
            line=dict(color='#185FA5', width=2, dash='dash'),
            customdata=forecast_df[['lower_68', 'upper_68',
                                    'lower_95', 'upper_95']].values,
            hovertemplate=(
                '<b>Prediksjon</b>: %{y:.1f} °C<br>'
                '68 % KI: %{customdata[0]:.1f}–%{customdata[1]:.1f} °C<br>'
                '95 % KI: %{customdata[2]:.1f}–%{customdata[3]:.1f} °C'
                '<extra></extra>'
            ),
        ))

        now_ms     = pd.Timestamp.now(tz='UTC').timestamp() * 1000
        horizon_ms = (pd.Timestamp.now(tz='UTC') +
                      timedelta(hours=travel_hours)).timestamp() * 1000
        fig.add_vline(x=now_ms, line_dash='dot', line_color='rgba(100,100,100,0.50)',
                      line_width=1, annotation_text='Nå', annotation_position='top left',
                      annotation_font_size=11, annotation_font_color='rgba(100,100,100,0.80)')
        fig.add_vline(x=horizon_ms, line_dash='dot', line_color='rgba(56,141,228,0.45)',
                      line_width=1, annotation_text=f'Datahorisont (+{travel_hours:.0f} t)',
                      annotation_position='top right', annotation_font_size=10,
                      annotation_font_color='rgba(56,141,228,0.75)')

    if fetsund_obs_df is not None and not fetsund_obs_df.empty:
        fig.add_trace(go.Scatter(
            x=fetsund_obs_df['time'], y=fetsund_obs_df['value'],
            mode='lines', name='Observert (Fetsund)',
            line=dict(color='#185FA5', width=2),
            hovertemplate='<b>Observert</b>: %{y:.1f} °C<extra></extra>',
        ))

    fig.update_layout(
        title=title, xaxis_title='',
        yaxis=dict(title='°C', range=[10, 28], fixedrange=True),
        height=430, hovermode='x unified', template='plotly_white',
        margin=dict(l=50, r=90, t=50, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='center', x=0.5, font=dict(size=10)),
    )
    return fig


# ============================================================================
# PAGE: INFORMASJON
# ============================================================================

def page_informasjon():
    st.title("Om prediksjonsmodellen")
    st.markdown(
        "Temperaturprediksjonen er utviklet for å gi arrangøren av **GlommaDyppen** "
        "et kunnskapsgrunnlag for sikkerhetsvurderinger. Modellen er ikke en offisiell "
        "meteorologisk tjeneste."
    )

    # ── Kart over målestasjoner ───────────────────────────────────────────────
    st.subheader("Kart over målestasjoner og strekninger")
    st.image(
        "kart_malestasjoner.png",
        caption=(
            "Oversikt over NVE-målestasjoner langs Vorma og Glomma med GPS-koordinater "
            "og elveavstander fra Minnesund. Kilde: Anton Vooren / Fet Svømmeklubb."
        ),
        use_container_width=True,
    )

    st.subheader("Prediksjonsmodell")
    st.markdown("""
    Modellen beregner forventet vanntemperatur langs svømmestrekningene:

    **T_pred = Fetsund_baseline + Vorma_anomali × 0,63**

    der *Fetsund_baseline* er median av siste 48 timer ved Fetsund,
    *Vorma_anomali* er gjeldende Vorma-temperatur minus Vormas 48-timers median,
    og koeffisienten 0,63 er empirisk validert mot 35+ kalde episoder fra 2018–2025.

    Prediksjonen gjelder **Fløter'n** (startpunkt Glommadyppen, 35,5 km fra Svanefoss)
    og **Fetsund** (mål, 45 km). Temperaturen er i praksis lik ved begge punkter —
    forskjellen er *når* det kalde vannet ankommer:

    | Punkt | Avstand fra Svanefoss | Transporttid |
    |---|---|---|
    | Fløter'n (start) | 35,5 km | **t = 7670 / Q** timer |
    | Fetsund (mål) | 45,0 km | **t = 9700 / Q** timer |

    Ved typisk augustvannføring (Q ≈ 400 m³/s) ankommer kaldt vann Fløter'n
    **~5 timer tidligere** enn Fetsund. Q = vannføring ved Ertesekken (m³/s).
    """)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        **Prediksjonen er pålitelig når:**
        - Det er aktive målinger fra Svanefoss eller Funnefoss (april–september)
        - Du ønsker å vite temperaturen ved Fløter'n / Fetsund **i dag eller i morgen**
        - Det er innen **1–2 uker** før GlommaDyppen
        """)
    with col4:
        st.markdown("""
        **Prediksjonen er *ikke* en langtidsprognose:**
        - Mange måneder før arrangementet reflekterer den kun **nåværende forhold**
        - Usikkerheten er ±2–3 °C (95 % KI)
        """)

    st.divider()

    st.subheader("🌊 Seiche-ettereffekt – forsinket kaldpuls fra Mjøsa")
    st.markdown("""
    Etter at sørlig vind har presset det varme overflatelaget mot sørenden av Mjøsa
    og drevet kaldt bunnvann (hypolimnion) opp mot Minnesund, vil **sprangsjiktet
    fortsette å oscillere** som en pendel selv etter at vinden har lagt seg.
    Denne indre bølgen (seiché) er beskrevet av Thendrup (1978) med en halvperiode på
    typisk **5–8 dager** ved normal sommerstratifisering.

    Praktisk konsekvens: en ny kaldpuls kan nå Glomma **5–12 dager etter den første**,
    uten nytt vindpådriv. Modellen overvåker dette og viser en forhøyet risikoindikator
    i dette tidsvinduet.

    | Kriterium for seiche-trigger | Verdi |
    |---|---|
    | Primær bunn ved Minnesund | < 10 °C |
    | Minimum temperaturdropp (ΔT) | ≥ 3 °C under 7-dagers baseline |
    | Forhøyet risikovindu | Dag 5–12 etter primær bunn |
    | Typisk halvperiode | 8–9 dager |

    **Validering 2015–2025 (682 juli–august-dager, Fetsund < 18 °C = «kaldt»):**

    | Modell | Sensitivitet | F1-score | FN-dager |
    |---|---|---|---|
    | Kun vindbasert | 0,70 | 0,756 | 167 |
    | Vind + seiche | **0,92** | **0,876** | **46** |

    Seiche-triggeren legger til 121 korrekte alarmflagg og bare 15 falske alarmer.
    """)

    st.divider()

    st.subheader("🧥 Våtdrakt og sikkerhet – Glommadyppen")
    st.info(
        "**Glommadyppen-regel:** Våtdrakt er obligatorisk for alle deltakere, "
        "uavhengig av vanntemperatur. Dette er en sikkerhetsmessig beslutning fra "
        "arrangøren. Unntak kan søkes individuelt hos arrangøren.",
        icon="🧥",
    )
    st.markdown("""
    Arrangøren følger ellers World Athletics-terskler for vurdering av gjennomføring,
    men kan utøve skjønn ved den nedre grensen basert på en helhetsvurdering:
    lufttemperatur, sol/skydekke, forventet svømmetid og deltakersammensetning.

    | Temperatur | World Athletics-vurdering | Glommadyppen – våtdrakt |
    |---|---|---|
    | < 14 °C | Avlysning anbefalt | Obligatorisk – avlysning vurderes |
    | 14–16 °C | Høy risiko – vurder avlysning | Obligatorisk – arrangør vurderer |
    | 16–18 °C | Gjennomføring tillatt | **Obligatorisk** |
    | 18–20 °C | Lav risiko | **Obligatorisk** |
    | 20–24 °C | Gode forhold | **Obligatorisk** |
    | > 24 °C | Varmt – sjeldent i Glomma | **Obligatorisk** – kontakt arrangør |

    Merk: Temperaturer over 20 °C er normalt ikke et problem i Glomma i august.
    Arrangørens helhetsvurdering veier tyngst — denne modellen er et beslutningsstøtteverktøy.
    """)


# ============================================================================
# PAGE: PREDIKSJON
# ============================================================================

def page_prediksjon():
    st.title("Temperaturprediksjon – Fløter'n / Fetsund")
    st.markdown(
        "Predikert vanntemperatur langs svømmestrekningen basert på observasjoner i Mjøsa, "
        "Vorma og Glomma. Primært prediksjonspunkt er **Fløter'n** (startpunkt Glommadyppen, "
        "35,5 km fra Svanefoss, t = 7670/Q). Fetsund bru (45 km, t = 9700/Q) er sekundært "
        "målepunkt. Det kalde vannet ankommer Fløter'n **4–5 timer tidligere** enn Fetsund "
        "ved typisk augustvannføring."
    )

    event_date = calculate_event_date(EVENT_YEAR)
    days_until = (event_date - pd.Timestamp.now(tz='UTC')).days
    oslo_dt    = event_date.tz_convert('Europe/Oslo')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Neste arrangement", oslo_dt.strftime("%d. %b %Y"))
    c2.metric("Dager igjen", str(max(0, days_until)))

    with st.spinner("Henter data…"):
        primary_df   = fetch_nve_data(STATION_SVANEFOSS,    1003, hours_back=168)
        if primary_df.empty:
            primary_df = fetch_nve_data(STATION_FUNNEFOSS_TEMP, 1003, hours_back=168)
        fetsund_temp  = fetch_nve_data(STATION_FETSUND,     1003, hours_back=168)
        ertesekken_q  = fetch_nve_data(STATION_ERTESEKKEN_Q, 1001, hours_back=168)
        frost_vind    = fetch_frost_wind(hours_back=168)
        weather_mjosa = fetch_weather_forecast(MJOSA_LAT, MJOSA_LON)
        if not weather_mjosa.empty:
            weather_mjosa = add_southerly_component(weather_mjosa)
        # 14 dagers historikk for seiche-deteksjon (trenger dag 5–12 tilbake)
        vorma_history = fetch_nve_data(STATION_SVANEFOSS, 1003, hours_back=336)
        if vorma_history.empty:
            vorma_history = fetch_nve_data(STATION_FUNNEFOSS_TEMP, 1003, hours_back=336)
        seiche = detect_seiche_risk(vorma_history)

    if primary_df.empty:
        st.error("Ingen Vorma-data tilgjengelig. Sjekk NVE HydAPI.")
        return

    _last_t = pd.to_datetime(primary_df['time'].max())
    if _last_t.tzinfo is None:
        _last_t = _last_t.tz_localize('UTC')
    data_age_days = (pd.Timestamp.now(tz='UTC') - _last_t).total_seconds() / 86400

    c3.metric("Siste Vorma-data",
              _last_t.tz_convert('Europe/Oslo').strftime('%d.%m %H:%M'))
    c4.metric("Dataalder", f"{data_age_days:.1f} dager",
              delta="⚠️ Gamle data" if data_age_days > 2 else None,
              delta_color="inverse")

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

    t_flotern, t_fetsund, q_val, q_src = calculate_travel_time(ertesekken_q)
    c4.metric(
        "Transporttid Fløter'n",
        f"{t_flotern} t",
        delta=f"Fetsund: {t_fetsund} t",
        delta_color="off",
        help=f"Fløter'n: t = 7670 / {q_val:.0f} m³/s ({q_src}) · "
             f"Fetsund: t = 9700 / {q_val:.0f} m³/s"
    )

    # ── Seiche-ettereffekt banner ─────────────────────────────────────────────
    if seiche['active']:
        ep_date_oslo = seiche['episode_date'].tz_convert(
            'Europe/Oslo').strftime('%-d. %b kl %H:%M')
        days_rem = seiche['days_remaining']
        st.warning(
            f"🌊 **Seiche-ettereffekt aktiv** – forhøyet risiko for sekundær kaldpuls\n\n"
            f"En bekreftet kald episode ble registrert ved Minnesund for "
            f"**{seiche['days_ago']:.1f} dager siden** "
            f"({ep_date_oslo}, min {seiche['episode_min_T']:.1f} °C, "
            f"ΔT = {seiche['episode_dT']:.1f} °C). "
            f"Sprangsjiktet i Mjøsa kan oscillere tilbake og gi en ny kaldpuls – "
            f"typisk opptrer sekundærdroppen 5–12 dager etter primær bunn. "
            f"**Forhøyet risikovindu varer i ca. {days_rem:.0f} dager til.**\n\n"
            f"*Validert 2015–2025: seiche-triggeren øker modellens sensitivitet "
            f"fra 0,70 til 0,92 (F1: 0,756 → 0,876) med minimal økning i falske alarmer.*",
            icon="🌊",
        )

    st.divider()

    # ── Prediksjon for arrangementet ──────────────────────────────────────────
    st.header("Prediksjon for arrangementet")

    t_flotern_now, travel_hours_now, _, _ = calculate_travel_time(ertesekken_q)
    pred_valid_to = (pd.Timestamp.now(tz='UTC') +
                     pd.Timedelta(hours=travel_hours_now)).tz_convert('Europe/Oslo')

    if days_until > 14:
        st.info(
            f"ℹ️ Prediksjonen viser nåværende forhold, ikke en prognose for august. "
            f"Det er {days_until} dager til GlommaDyppen "
            f"({oslo_dt.strftime('%-d. %B %Y')}). "
            f"Prediksjonen er gyldig frem til ca. "
            f"{pred_valid_to.strftime('%-d. %b kl %H:%M')} "
            f"(Fløter'n: neste {t_flotern_now:.0f} t · Fetsund: neste {travel_hours_now:.0f} t)."
        )
    else:
        st.success(
            f"✅ Prediksjon for arrangementet er aktiv ({days_until} dager igjen). "
            f"Kaldt vann fra Svanefoss når **Fløter'n** om {t_flotern_now:.0f} t "
            f"og Fetsund om {travel_hours_now:.0f} t."
        )

    prediction = predict_fetsund_temperature(
        primary_df, ertesekken_q, event_date,
        fetsund_temp_df=fetsund_temp,
    )

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
            assess_risk_open_water(pred_temp, weather_mjosa, seiche_risk=seiche)

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
                        Fløter'n / Fetsund
                    </div>
                </div>
                <div style="flex:1; min-width:0;">
                    <div style="font-weight:600; font-size:0.95em;">
                        {risk_label}
                    </div>
                    <div style="font-size:0.82em; color:#666; margin-top:3px;">
                        {ws_label}
                        &nbsp;·&nbsp;
                        95&nbsp;%&nbsp;KI:&nbsp;{lb:.1f}–{ub:.1f}&nbsp;°C
                        &nbsp;·&nbsp;
                        Fløter'n:&nbsp;{prediction['travel_hours_flotern']:.0f}&nbsp;t
                        &nbsp;·&nbsp;
                        Fetsund:&nbsp;{prediction['travel_hours']:.0f}&nbsp;t
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
            c1.metric("Vorma nå",            f"{prediction['vorma_temp']:.1f} °C",
                      delta=f"{prediction['anomaly']:+.1f} °C avvik")
            c2.metric("Fetsund baseline",    f"{prediction['fetsund_baseline']:.1f} °C",
                      help=prediction['baseline_source'])
            c3.metric("Transporttid Fløter'n", f"{prediction['travel_hours_flotern']} t",
                      help=f"t = 7670 / {prediction['q_used']:.0f} m³/s")
            c4.metric("Pålitelighet",        f"{prediction['confidence']*100:.0f} %")
            st.caption(
                f"Modell: T_pred = Fetsund_baseline + Vorma_anomali × κ "
                f"(κ = {TEMPERATURE_SURVIVAL}). "
                f"Validert mot data fra juli og august 2018–2025. "
                "Bruk med forsiktighet utenfor sommermånedene."
            )
    else:
        st.warning("⚠️ Ikke nok data for prediksjon.")

    st.divider()

    # ── Temperaturprognose ────────────────────────────────────────────────────
    st.subheader("Temperaturprognose – Fløter'n / Fetsund")

    forecast_df = build_fetsund_forecast(primary_df, fetsund_temp, ertesekken_q)
    t_flotern_h, travel_h_now, _, _ = calculate_travel_time(ertesekken_q)

    if not forecast_df.empty:
        fig_fc = _forecast_chart(fetsund_temp, forecast_df, travel_h_now)
        st.plotly_chart(fig_fc, use_container_width=True)
        st.caption(
            "Solid linje: observert (Fetsund) · Stiplet linje: prediksjon. "
            "Prediksjonen gjelder **Fløter'n** (start, 35,5 km fra Svanefoss) "
            "og **Fetsund** (mål, 45 km) — temperaturen er i praksis lik ved begge punkter. "
            f"Datahorisonten (+{travel_h_now:.0f} t) markerer der Vorma-observasjoner gir "
            "direkte grunnlag (σ ≈ 2 °C). Etter dette ekstrapoleres Vorma-anomalien "
            "eksponentielt og usikkerheten vokser tilsvarende."
        )
    else:
        st.warning("Ikke nok data for prognosevisning.")

    # ── Vind og oppvellingsrisiko ─────────────────────────────────────────────
    if not weather_mjosa.empty or not frost_vind.empty:
        st.divider()
        st.subheader("Vind og oppvellingsrisiko – Mjøsa")

        energy_df = build_wind_energy_series(frost_vind, weather_mjosa)

        c1, c2, c3, c4 = st.columns(4)
        if not energy_df.empty:
            obs_e  = energy_df[~energy_df['is_forecast']]
            fc_e   = energy_df[ energy_df['is_forecast']]
            cur_E  = float(obs_e['E'].iloc[-1]) if not obs_e.empty else 0.0
            pct    = round(cur_E / ENERGY_THRESHOLD * 100)
            fc_E   = float(fc_e['E'].iloc[-1])       if not fc_e.empty else cur_E
            fc_Ehi = float(fc_e['E_upper'].max())    if not fc_e.empty else cur_E

            c1.metric("Kumulativ E nå",   f"{cur_E:.1f} m·h",
                      help="Rullende 48-timers SE/S-vindenergi (Frost API), 24 t forskjøvet")
            c2.metric("Andel av terskel", f"{pct} %",
                      help=f"{ENERGY_THRESHOLD:.0f} m·h = 100 % (AUC = 0.86)")
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

        wind_tabs = st.tabs(["Kumulativ oppvellingsrisiko", "Vindretning og -hastighet"])
        with wind_tabs[0]:
            if not energy_df.empty:
                fig_e = _wind_energy_chart(energy_df)
                if fig_e:
                    st.plotly_chart(fig_e, use_container_width=True)
                st.caption(
                    f"E = Σ v_i × Δtᵢ for alle obs der vindretning ∈ 135–225° (SE/S), "
                    f"48-timers rullende vindu med 24 t lead-tid. "
                    f"Terskel {ENERGY_THRESHOLD:.0f} m·h og advarsel {ENERGY_WARN:.0f} m·h "
                    "er empirisk kalibrert mot 3 500+ obs jul–aug 2018–2025 "
                    "(AUC = 0.86 for ΔT < −3 °C)."
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
        _latest(sv_temp, "Svanefoss (Vorma)",  c1)
        _latest(fn_temp, "Funnefoss (Glomma)", c2)
        _latest(bl_temp, "Blaker (Glomma)",    c3)
        _latest(fe_temp, "Fetsund (Glomma)",   c4)

        fig = _temp_chart({
            'Svanefoss': sv_temp, 'Funnefoss': fn_temp,
            'Blaker':    bl_temp, 'Fetsund':   fe_temp,
        }, "Vanntemperatur – siste 7 dager")
        st.plotly_chart(fig, use_container_width=True)

        st.caption("""
        **Stasjoner:**
        Svanefoss (2.52.0) i Vorma ca. 22 km fra Mjøsa (referansepunkt) ·
        Funnefoss (2.410.0) i Glomma ca. 5 km ovenfor samløp ·
        Blaker (2.17.0) i Glomma 31,8 km fra Svanefoss ·
        **Fløter'n** (start Glommadyppen) 35,5 km fra Svanefoss – ingen NVE-stasjon ·
        Fetsund (2.587.0) målpunkt Glommadyppen, 45 km fra Svanefoss.
        """)

    # ── TAB 2: Vannføring ─────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Vannføring – siste 7 dager (NVE HydAPI)")
        st.caption(
            "Timesverdier i m³/s. Ertesekken brukes for transporttid: "
            "t = 7670/Q timer til Fløter'n (start), t = 9700/Q timer til Fetsund (mål)."
        )

        c1, c2, c3 = st.columns(3)
        def _latest_q(df, label, col):
            if df.empty:
                col.metric(label, "N/A")
            else:
                v = df.iloc[-1]['value']
                tf = round(TRANSPORT_COEFF_FLOTERN / v, 1) if v > 0 else None
                col.metric(label, f"{v:.0f} m³/s",
                           help=f"Fløter'n ≈ {tf} t" if tf else None)
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
        tf_calc = round(TRANSPORT_COEFF_FLOTERN / q_val, 1)
        t_calc  = round(TRANSPORT_COEFF / q_val, 1)
        st.info(
            f"**Fløter'n: t = 7670 / {q_val} = {tf_calc} timer** (35,5 km)  \n"
            f"**Fetsund:  t = 9700 / {q_val} = {t_calc} timer** (45,0 km)  \n"
            f"*Fløter'n: {t_calc - tf_calc:.1f} timer tidligere enn Fetsund*"
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
        - Fløter'n (start): t = 7670 / Q
        - Fetsund (mål): t = 9700 / Q
        - Vorma-anomali respons: 63 %
        - Validert 2018–2025 (AUC = 0,87)
        - 🌊 Seiche-ettereffekt: dag 5–12

        **Glommadyppen – våtdrakt**
        - 🧥 Obligatorisk uansett temperatur
        - Unntak: søk arrangøren

        **World Athletics OW-grenser**
        - < 14 °C: avlysning anbefalt
        - 14–16 °C: høy risiko
        - 16–18 °C: moderat risiko
        - 18–20 °C: lav risiko
        - 20–24 °C: gode forhold

        **Datakilder**
        - NVE HydAPI (vann)
        - MET Frost API (vind)
        - Met.no Locationforecast
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
