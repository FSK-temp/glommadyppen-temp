"""
glommadyppen_core.py
Streamlit-uavhengig kjernemodul: datahenting (NVE/Frost/Met.no) og
prediksjonsmodell for GlommaDyppen. Brukes av BÅDE streamlit_app.py
(live-appen) og log_prediction.py (GitHub Actions-cronjobb), slik at
begge alltid kjører nøyaktig samme modellogikk.

Ingen avhengighet til streamlit - inneholder ingen st.* kall, ingen
@st.cache_data. API-nøkler hentes via funksjonsargumenter eller
miljøvariabler (os.environ), ikke st.secrets.

Author: Anton Vooren
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

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
MODEL_SIGMA          = 2.0       # °C – prediksjonsstandardavvik når vi går inn i ren ekstrapolering
MODEL_SIGMA_DATA     = 0.6       # °C – residual innenfor datahorisonten (ekte Vorma-obs), ~1.25×validert MAE
                                  # (0.50-0.58 °C). Foreløpig estimat - bør valideres mot faktiske
                                  # prediksjonsresidualer når historisk prediksjonslogg finnes.
# Empiriske grenser fra Fetsund-historikk (2015–2025, juli dag 15 – august)
# Brukes til å klippe KI-båndene slik at de ikke overskrider fysisk mulig range.
TEMP_HIST_LOWER      = 10.0      # °C – P1 av historiske august-temperaturer ved Fetsund
TEMP_HIST_UPPER      = 24.0      # °C – historisk maksimum (aldri over 23,8 °C målt)

# ── Vindenergi-konfigurasjon ──────────────────────────────────────────────────
WIND_SECTOR_MIN      = 135
WIND_SECTOR_MAX      = 225
WIND_WINDOW_HOURS    = 48
WIND_LEAD_HOURS      = 24
CRITICAL_WIND_SPEED  = 1.9       # m/s
ENERGY_THRESHOLD     = 70.0      # m·h – alarm
ENERGY_WARN          = 45.0      # m·h – advarsel

# ── Vindrisiko-justering av temperaturprognosen ───────────────────────────────
# Basert på empirisk regresjon: kumulativ vindenergi (E, 48t/24t-lag) mot
# Fetsund-anomali (min over [+24t,+96t], 7-dagers baseline for å unngå
# baseline-kontaminering). r ≈ -0.29, R² ≈ 0.08 (n=5004, jul-aug 2015-2025).
# Sammenhengen er for svak til å flytte selve sentralestimatet (derfor brukes
# terskel-klassifikatoren over til det formålet) - men den brukes her til å
# SKJEVE usikkerhetsbåndet nedover og utvide det når værvarselet tilsier økt
# oppvellingsrisiko innenfor den horisonten Met.no-vindvarselet faktisk er
# pålitelig (jf. AUC=0.87 ved 1-3 døgn vs. 0.57 ved 7 døgn).
WIND_RISK_HORIZON_HOURS = 96      # t – utover dette anses vindvarselet for upålitelig
WIND_ANOMALY_SLOPE      = -0.015  # °C per m·h (svak, empirisk - se analysenotat)
WIND_ANOMALY_E_TYPISK   = 32.0    # m·h – median E i datasettet, brukt som nullpunkt
WIND_SIGMA_MULT_WARN    = 1.4     # KI-bredde-multiplikator når E > advarselsterskel
WIND_SIGMA_MULT_ALARM   = 1.8     # KI-bredde-multiplikator når E > alarmterskel

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

def fetch_nve_data(station_id, parameter, hours_back=168, api_key=None):
    """
    Henter data fra NVE HydAPI.
    Parameter-koder: 1001 = vassføring (m³/s), 1003 = vanntemperatur (°C)
    """
    api_key = api_key or os.environ.get("NVE_API_KEY")
    try:
        url = f"{NVE_BASE_URL}/Observations"
        headers = ({"X-API-Key": api_key, "accept": "application/json"}
                   if api_key else {"accept": "application/json"})
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
            print(f"[glommadyppen_core] Datafeil stasjon {station_id}: "
                  f"{str(e)[:100]}", file=sys.stderr)
        return pd.DataFrame(columns=['time', 'value', 'quality'])


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
        print(f"[glommadyppen_core] Feil ved henting av varsel: {e}", file=sys.stderr)
        return pd.DataFrame()


# ============================================================================
# ANALYSIS / MODEL FUNCTIONS
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


PREDICTION_LOG_SHEET_ID = "1P4jzHvGVAIlNaFr_ksw6lw6hdWao1o-bn12TIUBAwWk"
PREDICTION_LOG_WORKSHEET = "prediksjonslogg"


def read_prediction_log(sheet_id=None, worksheet_name=None):
    """
    Leser prediksjonsloggen (skrevet av log_prediction.py) via Googles
    offentlige CSV-eksport - krever INGEN autentisering, siden arket er delt
    som "alle med lenken kan redigere" (som også gir leserettighet). Brukes
    by appen for å vise "prediksjons-evolusjon" mot fasit.

    Returnerer tom DataFrame (ikke exception) hvis arket ikke er tilgjengelig
    ennå eller fanen ikke finnes.
    """
    sheet_id = sheet_id or PREDICTION_LOG_SHEET_ID
    worksheet_name = worksheet_name or PREDICTION_LOG_WORKSHEET
    url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}"
        f"/gviz/tq?tqx=out:csv&sheet={worksheet_name}"
    )
    try:
        df = pd.read_csv(url)
        if df.empty:
            return df
        for col in ('logged_at', 'event_date'):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except Exception as e:
        print(f"[glommadyppen_core] Kunne ikke lese prediksjonslogg: {e}", file=sys.stderr)
        return pd.DataFrame()


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


def build_fetsund_forecast(vorma_df, fetsund_df, discharge_df,
                           hours_ahead=120, step_h=3, energy_df=None):
    """
    Beregner tidsserie for predikert temperatur ved Fløter'n / Fetsund
    med usikkerhetsintervaller.

    Modell: T_pred(t) = fetsund_baseline + vorma_anomaly(t - travel_h) × κ
    Usikkerhet: lav og tilnærmet flat innenfor datahorisonten (travel_h, ekte
    Vorma-observasjoner - kun κ/transporttid-residual, MODEL_SIGMA_DATA), vokser
    deretter med √(ekstrapoleringstid) mot MODEL_SIGMA og videre etter hvert som
    vi går utover datahorisonten uten ny data.

    Hvis energy_df (fra build_wind_energy_series) sendes inn, justeres KI-båndet
    innenfor WIND_RISK_HORIZON_HOURS basert på forventet SE/S-vindenergi:
    sentralestimatet ('predicted') røres IKKE, siden vind-magnitude-sammenhengen
    er for svak (R² ≈ 0.08) til å brukes som punktestimat - se WIND_ANOMALY_SLOPE.
    I stedet utvides/skjeves nedre KI-grense for å reflektere økt nedsiderisiko.
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

    # ── Forbered vindenergi-prognose for oppslag (E mot tid) ───────────────────
    energy_lookup = None
    if energy_df is not None and not energy_df.empty:
        energy_lookup = energy_df[['time', 'E', 'is_forecast']].dropna(subset=['time']).copy()
        energy_lookup['time'] = pd.to_datetime(energy_lookup['time'])
        if energy_lookup['time'].dt.tz is None:
            energy_lookup['time'] = energy_lookup['time'].dt.tz_localize('UTC')
        energy_lookup = energy_lookup.sort_values('time').reset_index(drop=True)

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
        # To regimer: (1) innenfor datahorisonten kjenner vi den faktiske
        # Vorma-temperaturen på vei nedover elva - usikkerheten er kun
        # κ/transporttid-modellens residual (MODEL_SIGMA_DATA), og vokser svakt
        # fra ~0 (h=0, ren rapportering av nå-tilstand) til MODEL_SIGMA_DATA
        # (h=travel_h). (2) Utover dette er det ren ekstrapolering uten ny data,
        # og usikkerheten vokser videre mot/forbi MODEL_SIGMA - her dominerer
        # værprognosen, og det er her vindrisiko-justeringen under slår inn.
        sigma  = (MODEL_SIGMA_DATA * ramp
                  + (MODEL_SIGMA - MODEL_SIGMA_DATA) * np.sqrt(extrap / 24.0))

        # ── Vindrisiko-justering (kun innenfor pålitelig prognosehorisont) ─────
        e_fc        = None
        risk_level  = None
        sigma_mult  = 1.0
        risk_shift  = 0.0
        if energy_lookup is not None and h_elapsed <= WIND_RISK_HORIZON_HOURS:
            nearest = energy_lookup.iloc[
                (energy_lookup['time'] - t_fut).abs().argsort()[:1]
            ]
            if not nearest.empty and abs(
                (nearest.iloc[0]['time'] - t_fut).total_seconds()
            ) <= 5400:  # 90 min toleranse for matching
                e_fc = float(nearest.iloc[0]['E'])
                if e_fc >= ENERGY_THRESHOLD:
                    sigma_mult, risk_level = WIND_SIGMA_MULT_ALARM, 'alarm'
                elif e_fc >= ENERGY_WARN:
                    sigma_mult, risk_level = WIND_SIGMA_MULT_WARN, 'advarsel'
                else:
                    risk_level = 'lav'
                # Kun nedsiderisiko - vind gir aldri grunnlag for å anta varmere.
                risk_shift = min(0.0, WIND_ANOMALY_SLOPE * (e_fc - WIND_ANOMALY_E_TYPISK))

        sigma_eff = sigma * sigma_mult

        rows.append({
            'time':            t_fut,
            'predicted':       round(pred, 2),
            'lower_68':        round(max(pred + risk_shift - sigma_eff,        TEMP_HIST_LOWER), 2),
            'upper_68':        round(min(pred + sigma_eff,                     TEMP_HIST_UPPER), 2),
            'lower_95':        round(max(pred + risk_shift - 1.96 * sigma_eff, TEMP_HIST_LOWER), 2),
            'upper_95':        round(min(pred + 1.96 * sigma_eff,              TEMP_HIST_UPPER), 2),
            'wind_E_forecast': round(e_fc, 1) if e_fc is not None else None,
            'wind_risk_level': risk_level,
        })

    return pd.DataFrame(rows)


__all__ = ['NVE_BASE_URL', 'FROST_CLIENT_ID', 'FROST_BASE_URL', 'STATION_SVANEFOSS', 'STATION_FUNNEFOSS_TEMP', 'STATION_ERTESEKKEN_Q', 'STATION_BLAKER', 'STATION_FUNNEFOSS_Q', 'STATION_FETSUND', 'FROST_STATION_KISE', 'MJOSA_LAT', 'MJOSA_LON', 'BINGSFOSSEN_LAT', 'BINGSFOSSEN_LON', 'FETSUND_LAT', 'FETSUND_LON', 'TRANSPORT_COEFF', 'TRANSPORT_COEFF_BLA', 'TRANSPORT_COEFF_FLOTERN', 'FALLBACK_DISCHARGE', 'TEMPERATURE_SURVIVAL', 'MODEL_SIGMA', 'MODEL_SIGMA_DATA', 'TEMP_HIST_LOWER', 'TEMP_HIST_UPPER', 'WIND_SECTOR_MIN', 'WIND_SECTOR_MAX', 'WIND_WINDOW_HOURS', 'WIND_LEAD_HOURS', 'CRITICAL_WIND_SPEED', 'ENERGY_THRESHOLD', 'ENERGY_WARN', 'WIND_RISK_HORIZON_HOURS', 'WIND_ANOMALY_SLOPE', 'WIND_ANOMALY_E_TYPISK', 'WIND_SIGMA_MULT_WARN', 'WIND_SIGMA_MULT_ALARM', 'SEICHE_WINDOW_START_DAYS', 'SEICHE_WINDOW_END_DAYS', 'SEICHE_COLD_THRESHOLD', 'SEICHE_ANOMALY_MIN', 'OW_ABORT', 'OW_WETSUIT_REQUIRED', 'OW_WETSUIT_STRONG', 'OW_WETSUIT_OPTIONAL', 'OW_TOO_WARM', 'EVENT_YEAR', 'EVENT_MONTH', 'EVENT_DAY_OF_WEEK', 'fetch_nve_data', 'fetch_frost_wind', 'fetch_weather_forecast', 'add_southerly_component', 'detect_temperature_drop', 'calculate_travel_time', 'detect_seiche_risk', 'predict_fetsund_temperature', 'assess_risk_open_water', 'calculate_event_date', 'wind_rose_label', 'build_wind_energy_series', 'build_fetsund_forecast', 'read_prediction_log', 'PREDICTION_LOG_SHEET_ID', 'PREDICTION_LOG_WORKSHEET']
