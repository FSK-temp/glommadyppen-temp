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
from datetime import timedelta

import pandas as pd
import gspread

import glommadyppen_core as core

# ── Konfigurasjon ────────────────────────────────────────────────────────────
# Ark-ID og fanenavn er definert ETT sted (glommadyppen_core.py) slik at
# skriving (her) og lesing (appen, via core.read_prediction_log) aldri kan
# komme ut av synk.
SHEET_ID       = os.environ.get("GOOGLE_SHEET_ID", core.PREDICTION_LOG_SHEET_ID)
WORKSHEET_NAME = core.PREDICTION_LOG_WORKSHEET
LOG_HORIZONS_H = [24, 48, 72, 96]    # timer frem - matcher WIND_RISK_HORIZON_HOURS

HEADER = (
    ["logged_at", "event_date", "days_until_event"]
    + ["vorma_temp_now", "vorma_baseline", "vorma_anomaly",
       "fetsund_temp_now", "fetsund_baseline",
       "discharge_q", "travel_hours", "wind_E_now",
       "seiche_active", "seiche_days_remaining"]
    + [f"{prefix}_h{h}" for h in LOG_HORIZONS_H
       for prefix in ("predicted", "lower68", "upper68", "windE_fc", "windrisk")]
    + ["predicted_event", "lower68_event", "upper68_event"]
)


def fetch_inputs():
    """Henter alle rådata-serier - identisk med page_prediksjon() i appen."""
    nve_key = os.environ.get("NVE_API_KEY")
    primary_df = core.fetch_nve_data(core.STATION_SVANEFOSS, 1003, hours_back=168, api_key=nve_key)
    if primary_df.empty:
        primary_df = core.fetch_nve_data(core.STATION_FUNNEFOSS_TEMP, 1003, hours_back=168, api_key=nve_key)

    fetsund_temp = core.fetch_nve_data(core.STATION_FETSUND, 1003, hours_back=168, api_key=nve_key)
    ertesekken_q = core.fetch_nve_data(core.STATION_ERTESEKKEN_Q, 1001, hours_back=168, api_key=nve_key)
    frost_vind   = core.fetch_frost_wind(hours_back=168)
    weather_mjosa = core.fetch_weather_forecast(core.MJOSA_LAT, core.MJOSA_LON)
    if not weather_mjosa.empty:
        weather_mjosa = core.add_southerly_component(weather_mjosa)

    vorma_history = core.fetch_nve_data(core.STATION_SVANEFOSS, 1003, hours_back=336, api_key=nve_key)
    if vorma_history.empty:
        vorma_history = core.fetch_nve_data(core.STATION_FUNNEFOSS_TEMP, 1003, hours_back=336, api_key=nve_key)

    return primary_df, fetsund_temp, ertesekken_q, frost_vind, weather_mjosa, vorma_history


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
    (primary_df, fetsund_temp, ertesekken_q,
     frost_vind, weather_mjosa, vorma_history) = fetch_inputs()

    if primary_df.empty:
        print("Ingen Vorma-data tilgjengelig - hopper over denne loggingen.", file=sys.stderr)
        return None

    now_utc = pd.Timestamp.now(tz='UTC')
    event_date = core.calculate_event_date(core.EVENT_YEAR)
    days_until = (event_date - now_utc).days

    latest_vorma = primary_df.iloc[-1]['value']
    vorma_baseline = primary_df[
        pd.to_datetime(primary_df['time']) >= primary_df['time'].max() - timedelta(hours=48)
    ]['value'].mean()
    vorma_anomaly = latest_vorma - vorma_baseline

    fetsund_now = fetsund_temp.iloc[-1]['value'] if not fetsund_temp.empty else None
    fetsund_baseline = (
        fetsund_temp[fetsund_temp['time'] >= fetsund_temp['time'].max() - timedelta(hours=48)]['value'].median()
        if not fetsund_temp.empty else None
    )

    t_flotern, travel_hours, q_used, q_source = core.calculate_travel_time(ertesekken_q)
    seiche = core.detect_seiche_risk(vorma_history)

    energy_df = core.build_wind_energy_series(frost_vind, weather_mjosa)
    wind_e_now = None
    if not energy_df.empty:
        obs_e = energy_df[~energy_df['is_forecast']]
        if not obs_e.empty:
            wind_e_now = float(obs_e['E'].iloc[-1])

    forecast_df = core.build_fetsund_forecast(primary_df, fetsund_temp, ertesekken_q, energy_df=energy_df)

    row = {
        "logged_at":            now_utc.isoformat(),
        "event_date":           event_date.isoformat(),
        "days_until_event":     days_until,
        "vorma_temp_now":       round(float(latest_vorma), 2),
        "vorma_baseline":       round(float(vorma_baseline), 2),
        "vorma_anomaly":        round(float(vorma_anomaly), 2),
        "fetsund_temp_now":     round(float(fetsund_now), 2) if fetsund_now is not None else None,
        "fetsund_baseline":     round(float(fetsund_baseline), 2) if fetsund_baseline is not None else None,
        "discharge_q":          round(float(q_used), 1),
        "travel_hours":         travel_hours,
        "wind_E_now":           round(wind_e_now, 1) if wind_e_now is not None else None,
        "seiche_active":        bool(seiche['active']),
        "seiche_days_remaining": seiche['days_remaining'],
    }

    for h in LOG_HORIZONS_H:
        r = nearest_forecast_row(forecast_df, h)
        if r is None:
            row.update({f"predicted_h{h}": None, f"lower68_h{h}": None,
                        f"upper68_h{h}": None, f"windE_fc_h{h}": None,
                        f"windrisk_h{h}": None})
        else:
            row.update({
                f"predicted_h{h}": r['predicted'],
                f"lower68_h{h}":   r['lower_68'],
                f"upper68_h{h}":   r['upper_68'],
                f"windE_fc_h{h}":  r.get('wind_E_forecast'),
                f"windrisk_h{h}":  r.get('wind_risk_level'),
            })

    event_pred = core.predict_fetsund_temperature(
        primary_df, ertesekken_q, event_date, fetsund_temp_df=fetsund_temp,
    )
    if event_pred:
        sigma = 2.0
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


def append_row(ws, row_dict):
    existing = ws.get_all_values()
    if not existing or not any(existing[0]):
        ws.append_row(HEADER, value_input_option="RAW")
    # NB: pd.notna() er nødvendig her, ikke bare "is not None" - verdier fra
    # forecast_df.loc[idx] (f.eks. windE_fc_h{h}/windrisk_h{h} utenfor
    # vindrisiko-horisonten) blir NaN, ikke None, når pandas henter ut en rad
    # som blander tall- og strengkolonner. "NaN is not None" er True, så en
    # ren None-sjekk slipper NaN gjennom til gspread (ugyldig JSON / "nan"
    # skrevet til arket). Se samme fiks i streamlit_app.py sin Dagsprognose.
    values = [v if pd.notna(v) else "" for v in
              (row_dict.get(col, "") for col in HEADER)]
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
