"""
log_prediction.py
Frittstående script - kjøres via GitHub Actions cron (se .github/workflows/
log_prediction.yml), IKKE av Streamlit-appen selv (Streamlit Cloud sover ved
inaktivitet og kan ikke kjøre planlagte jobber pålitelig).

Henter nåværende observasjoner + bygger prediksjonen nøyaktig slik appen gjør
(samme funksjoner fra glommadyppen_core.py), og legger til én rad i Google
Sheet-loggen. Loggen brukes til (1) etterprøving av prediksjoner mot fasit og
(2) fremtidig modelltrening - se kolonneforklaring i HEADER under.

Miljøvariabler (settes som GitHub Actions secrets):
    GCP_SA_KEY      - full JSON-innhold for Google service-account-nøkkelen
    NVE_API_KEY     - NVE HydAPI-nøkkel
    GOOGLE_SHEET_ID - (valgfri) overstyrer standard ark-ID under

Author: Anton Vooren
"""

import os
import sys
import json
import math
from datetime import timedelta

import pandas as pd
import gspread

import glommadyppen_core as core

# ── Konfigurasjon ────────────────────────────────────────────────────────────
# Ark-ID og fanenavn er definert ETT sted (glommadyppen_core.py) slik at
# skriving (her) og lesing (appen, via core.read_prediction_log) aldri kan
# komme ut av synk.
REQUIRED_CORE_VERSION = "1.7.0"
if getattr(core, "CORE_VERSION", None) != REQUIRED_CORE_VERSION:
    # Feil hardt og tidlig. Skriver vi til arket med en gammel kjerne, blir
    # loggen stille inkonsistent - noen rader med dynamisk κ, andre uten - og
    # da er den ubrukelig som grunnlag for kalibrering.
    raise RuntimeError(
        f"glommadyppen_core.py er versjon "
        f"{getattr(core, 'CORE_VERSION', 'ukjent (< 1.7.0)')}, "
        f"men log_prediction.py krever {REQUIRED_CORE_VERSION}. "
        "Rull ut begge filene sammen."
    )

SHEET_ID       = os.environ.get("GOOGLE_SHEET_ID", core.PREDICTION_LOG_SHEET_ID)
WORKSHEET_NAME = core.PREDICTION_LOG_WORKSHEET
LOG_HORIZONS_H = [24, 48, 72, 96]    # timer frem - matcher WIND_RISK_HORIZON_HOURS

# Nye kolonner legges ALLTID til på slutten, slik at eksisterende rader i arket
# forblir riktig justert. _ensure_header() utvider overskriftsraden ved behov.
HEADER = (
    ["logged_at", "event_date", "days_until_event"]
    + ["vorma_temp_now", "vorma_baseline", "vorma_anomaly",
       "fetsund_temp_now", "fetsund_baseline",
       "discharge_q", "travel_hours", "wind_E_now",
       "seiche_active", "seiche_days_remaining"]
    + [f"{prefix}_h{h}" for h in LOG_HORIZONS_H
       for prefix in ("predicted", "lower68", "upper68", "windE_fc", "windrisk")]
    + ["predicted_event", "lower68_event", "upper68_event"]
    # ── Nytt fra v1.7 ────────────────────────────────────────────────────────
    + ["discharge_q_glomma", "mixing_fraction", "kappa_episode", "kappa_increment",
       "kappa_source", "forecast_mode", "seiche_rejected_reason"]
    + [f"sigma_h{h}" for h in LOG_HORIZONS_H]
    + [f"delta_vorma_h{h}" for h in LOG_HORIZONS_H]
)


def fetch_inputs():
    """Henter alle rådata-serier - identisk med page_prediksjon() i appen."""
    nve_key = os.environ.get("NVE_API_KEY")

    # ÉN henting av Vorma-temperatur med det lengste vinduet som trengs (20 d
    # for seiche-deteksjon); de siste 7 døgnene skjæres ut til prognosen.
    vorma_history = core.fetch_nve_data(core.STATION_SVANEFOSS, 1003,
                                        hours_back=core.SEICHE_HISTORY_HOURS,
                                        api_key=nve_key)
    if vorma_history.empty:
        vorma_history = core.fetch_nve_data(core.STATION_FUNNEFOSS_TEMP, 1003,
                                            hours_back=core.SEICHE_HISTORY_HOURS,
                                            api_key=nve_key)
    primary_df = vorma_history.copy()
    if not primary_df.empty:
        cut = pd.to_datetime(primary_df['time']).max() - pd.Timedelta(hours=168)
        primary_df = primary_df[pd.to_datetime(primary_df['time']) >= cut] \
                         .reset_index(drop=True)

    fetsund_temp = core.fetch_nve_data(core.STATION_FETSUND, 1003, hours_back=168, api_key=nve_key)
    ertesekken_q = core.fetch_nve_data(core.STATION_ERTESEKKEN_Q, 1001, hours_back=168, api_key=nve_key)
    # Glomma-vannføring - nødvendig for dynamisk uttynning
    funnefoss_q  = core.fetch_nve_data(core.STATION_FUNNEFOSS_Q, 1001, hours_back=168, api_key=nve_key)
    frost_vind   = core.fetch_frost_wind(hours_back=168)
    weather_mjosa = core.fetch_weather_forecast(core.MJOSA_LAT, core.MJOSA_LON)
    if not weather_mjosa.empty:
        weather_mjosa = core.add_southerly_component(weather_mjosa)

    return (primary_df, fetsund_temp, ertesekken_q, funnefoss_q,
            frost_vind, weather_mjosa, vorma_history)


def nearest_forecast_row(forecast_df, target_h):
    """Finn forecast_df-raden nærmest target_h timer frem i tid (samme grid som build_fetsund_forecast)."""
    if forecast_df is None or forecast_df.empty:
        return None
    now_utc = pd.Timestamp.now(tz='UTC')
    target_t = now_utc + timedelta(hours=target_h)
    idx = (forecast_df['time'] - target_t).abs().idxmin()
    return forecast_df.loc[idx]


def build_snapshot():
    """Bygger én loggrad (dict) fra nåværende observasjoner + prediksjon."""
    (primary_df, fetsund_temp, ertesekken_q, funnefoss_q,
     frost_vind, weather_mjosa, vorma_history) = fetch_inputs()

    if primary_df.empty:
        print("Ingen Vorma-data tilgjengelig - hopper over denne loggingen.", file=sys.stderr)
        return None

    now_utc = pd.Timestamp.now(tz='UTC')
    event_date = core.calculate_event_date(core.EVENT_YEAR)
    days_until = (event_date - now_utc).days

    latest_vorma = primary_df.iloc[-1]['value']
    # 72t MEDIAN (ikke 48t mean): robust mot at en kaldpuls passerer gjennom
    # selve baselinevinduet og gjør anomalien falskt positiv.
    vorma_baseline = primary_df[
        pd.to_datetime(primary_df['time'])
        >= primary_df['time'].max() - timedelta(hours=core.VORMA_BASELINE_HOURS)
    ]['value'].median()
    vorma_anomaly = latest_vorma - vorma_baseline

    fetsund_now = fetsund_temp.iloc[-1]['value'] if not fetsund_temp.empty else float('nan')
    fetsund_baseline = (
        fetsund_temp[fetsund_temp['time'] >= fetsund_temp['time'].max() - timedelta(hours=48)]['value'].median()
        if not fetsund_temp.empty else float('nan')
    )

    t_flotern, travel_hours, q_used, q_source = core.calculate_travel_time(ertesekken_q)
    q_glomma, _ = core.safe_discharge(funnefoss_q, core.FALLBACK_DISCHARGE_GLOMMA)
    kappa_ep, f_mix, kappa_src = core.dilution_kappa(ertesekken_q, funnefoss_q, mode='episode')
    kappa_in, _, _             = core.dilution_kappa(ertesekken_q, funnefoss_q, mode='increment')
    seiche = core.detect_seiche_risk(vorma_history)

    energy_df = core.build_wind_energy_series(frost_vind, weather_mjosa)
    wind_e_now = None
    if not energy_df.empty:
        obs_e = energy_df[~energy_df['is_forecast']]
        if not obs_e.empty:
            wind_e_now = float(obs_e['E'].iloc[-1])

    forecast_df = core.build_fetsund_forecast(
        primary_df, fetsund_temp, ertesekken_q,
        glomma_q_df=funnefoss_q, energy_df=energy_df,
    )
    forecast_mode = forecast_df['mode'].iloc[0] if not forecast_df.empty else None

    row = {
        "logged_at":            now_utc.isoformat(),
        "event_date":           event_date.isoformat(),
        "days_until_event":     days_until,
        "vorma_temp_now":       round(float(latest_vorma), 2),
        "vorma_baseline":       (round(float(vorma_baseline), 2)
                                 if pd.notna(vorma_baseline) else None),
        "vorma_anomaly":        (round(float(vorma_anomaly), 2)
                                 if pd.notna(vorma_anomaly) else None),
        "fetsund_temp_now":     (round(float(fetsund_now), 2)
                                 if pd.notna(fetsund_now) else None),
        "fetsund_baseline":     (round(float(fetsund_baseline), 2)
                                 if pd.notna(fetsund_baseline) else None),
        "discharge_q":          round(float(q_used), 1),
        "travel_hours":         travel_hours,
        "wind_E_now":           round(wind_e_now, 1) if wind_e_now is not None else None,
        "seiche_active":        bool(seiche['active']),
        "seiche_days_remaining": seiche['days_remaining'],
        # ── Nytt fra v1.7 ────────────────────────────────────────────────────
        "discharge_q_glomma":   round(float(q_glomma), 1),
        "mixing_fraction":      round(float(f_mix), 3),
        "kappa_episode":        round(float(kappa_ep), 3),
        "kappa_increment":      round(float(kappa_in), 3),
        "kappa_source":         kappa_src,
        "forecast_mode":        forecast_mode,
        "seiche_rejected_reason": seiche.get('rejected_reason'),
    }

    for h in LOG_HORIZONS_H:
        r = nearest_forecast_row(forecast_df, h)
        if r is None:
            row.update({f"predicted_h{h}": None, f"lower68_h{h}": None,
                        f"upper68_h{h}": None, f"windE_fc_h{h}": None,
                        f"windrisk_h{h}": None, f"sigma_h{h}": None,
                        f"delta_vorma_h{h}": None})
        else:
            row.update({
                f"predicted_h{h}": r['predicted'],
                f"lower68_h{h}":   r['lower_68'],
                f"upper68_h{h}":   r['upper_68'],
                f"windE_fc_h{h}":  r.get('wind_E_forecast'),
                f"windrisk_h{h}":  r.get('wind_risk_level'),
                f"sigma_h{h}":       r.get('sigma'),
                f"delta_vorma_h{h}": r.get('delta_vorma'),
            })

    event_pred = core.predict_fetsund_temperature(
        primary_df, ertesekken_q, event_date, fetsund_temp_df=fetsund_temp,
        glomma_q_df=funnefoss_q,
    )
    if event_pred:
        # Arrangementet ligger normalt langt utenfor datahorisonten, så
        # usikkerheten er mettet: bruk modellens asymptote i stedet for 2.0.
        sigma = core.MODEL_SIGMA_ASYMPTOTE
        row["predicted_event"] = round(float(event_pred['predicted_temp']), 2)
        row["lower68_event"]   = round(float(event_pred['predicted_temp']) - sigma, 2)
        row["upper68_event"]   = round(float(event_pred['predicted_temp']) + sigma, 2)
    else:
        row["predicted_event"] = row["lower68_event"] = row["upper68_event"] = None

    return row


def get_worksheet():
    sa_key_raw = os.environ.get("GCP_SA_KEY")
    if not sa_key_raw:
        raise RuntimeError("Miljøvariabel GCP_SA_KEY (service-account JSON) er ikke satt.")
    creds_dict = json.loads(sa_key_raw)
    gc = gspread.service_account_from_dict(creds_dict)
    sh = gc.open_by_key(SHEET_ID)
    try:
        ws = sh.worksheet(WORKSHEET_NAME)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=WORKSHEET_NAME, rows=1000, cols=len(HEADER) + 5)
    return ws


def _sanitize(v):
    """
    Gjør en verdi trygg for gspread. NaN/NaT/None → tom streng.

    NB: `if not v` duger IKKE her - NaN er truthy i Python, og en NaN havner
    da i arket som strengen "nan". Bruk eksplisitt isnan-sjekk.
    """
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    if isinstance(v, (bool,)):
        return "TRUE" if v else "FALSE"
    return v


def _ensure_header(ws):
    """
    Sørger for at overskriftsraden inneholder alle kolonnene i HEADER.

    Nye kolonner legges alltid til bakerst i HEADER, så et eksisterende ark
    kan utvides uten at gamle rader forskyves - de får bare tomme celler i de
    nye kolonnene. Uten dette ville nye felter skrives inn under gamle
    overskrifter og hele loggen bli feiljustert.
    """
    existing = ws.get_all_values()
    if not existing or not any(existing[0]):
        ws.append_row(HEADER, value_input_option="RAW")
        return HEADER

    current = existing[0]
    if current == HEADER:
        return current

    missing = [c for c in HEADER if c not in current]
    if not missing:
        # Arket har en annen rekkefølge enn HEADER - respekter arkets rekkefølge.
        return current

    new_header = current + missing
    if ws.col_count < len(new_header):
        ws.add_cols(len(new_header) - ws.col_count)
    ws.update([new_header], f"A1")
    print(f"Overskriftsrad utvidet med {len(missing)} nye kolonner: "
          f"{', '.join(missing)}")
    return new_header


def append_row(ws, row_dict):
    header = _ensure_header(ws)
    values = [_sanitize(row_dict.get(col)) for col in header]
    ws.append_row(values, value_input_option="RAW")


def main():
    snapshot = build_snapshot()
    if snapshot is None:
        sys.exit(0)  # ikke en feil - bare ingen data tilgjengelig akkurat nå

    print(f"Logger snapshot for {snapshot['logged_at']} "
          f"(predicted_event={snapshot['predicted_event']})")

    ws = get_worksheet()
    append_row(ws, snapshot)
    print("Rad lagt til i Google Sheet.")


if __name__ == "__main__":
    main()
