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
from datetime import datetime, timedelta, timezone

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Versjonsmerke. streamlit_app.py og log_prediction.py sjekker denne ved
# oppstart. Uten sjekken gir en delvis utrulling (ny app + gammel kjerne) bare
# en sladdet NameError på Streamlit Cloud, som er nesten umulig å feilsøke.
# Øk versjonen hver gang det legges til navn eller endres funksjonssignaturer.
CORE_VERSION = "1.9.0"

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

# ── Uttynning ved samløpet Vorma × Glomma ────────────────────────────────────
# κ er IKKE en fri modellkonstant - den ER Vormas blandingsandel i samløpet:
#       f = Q_Vorma / (Q_Vorma + Q_Glomma)
# Historisk median jul-aug 2017-2025 (Ertesekken 2.197.0 × Funnefoss kraftverk
# 2.412.0) er f = 0.633 - praktisk talt identisk med den empiriske episode-κ
# = 0.63 fra 35 kalde episoder. Det bekrefter tolkningen, og betyr at κ kan
# beregnes fra sist målte vannføring i stedet for å låses til én konstant.
#
# f varierer 0.51-0.76 (5.-95. persentil) i normale somre. Ved et Glomma-
# flomår kan den falle langt lavere, og da er en fast κ = 0.63 direkte
# misvisende - kaldpulsen fra Mjøsa blir da langt kraftigere fortynnet.
MIXING_FRACTION_FALLBACK = 0.633  # median f, brukes når Glomma-vannføring mangler
MIXING_FRACTION_MIN      = 0.20   # sanity-grenser på beregnet f
MIXING_FRACTION_MAX      = 0.92

# κ = η · f. To regimer med hver sin η:
#   EPISODE   – amplituden til en hel kaldepisode (min mot baseline). η = 1.00
#               reproduserer κ = 0.63 ved medianvannføring, dvs. ren blanding.
#   INCREMENT – forsterkningen på time-til-time-ENDRINGER, som prognosen bruker.
#               Empirisk η = 0.59 (regresjon ΔT_Fetsund = η·f·ΔT_Svanefoss uten
#               konstantledd, n = 10 552 timer jul-aug 2017-2025). Lavere enn 1
#               fordi lengdedispersjon i elva glatter ut korte svingninger før
#               de når Fetsund.
DILUTION_ETA_EPISODE   = 1.00
DILUTION_ETA_INCREMENT = 0.59

TEMPERATURE_SURVIVAL = 0.63      # Fallback-κ når vannføringsdata mangler helt

# ── Vannføring: gyldighetsgrenser (sensorfeil / nullverdier) ─────────────────
DISCHARGE_MIN_VALID = 20.0       # m³/s – under dette regnes måling som feil
DISCHARGE_MAX_VALID = 4000.0     # m³/s – over dette regnes måling som feil
FALLBACK_DISCHARGE_GLOMMA = 285.0  # August-median Funnefoss kraftverk (m³/s)

# ── Usikkerhetsmodell ────────────────────────────────────────────────────────
# Residualen er sterkt heteroskedastisk: den avhenger av hvor stor
# temperaturendring som er under transport. Empirisk (jul-aug 2017-2025,
# n = 10 552, horisont = transporttid):
#       |ΔT_Vorma|  0-0.5   0.5-1    1-2    2-3     >3
#       sd (°C)      0.47    0.57   0.72   1.02   1.53
# En flat σ = 0.6 er derfor omtrent riktig i rolige perioder, men 2-3× for
# smal nøyaktig når en stor kaldpuls er under transport - altså i den ene
# situasjonen verktøyet finnes for. Kvadratisk sammensetning:
#       σ = √(SIGMA_BASE² + (SIGMA_PER_DELTA · |ΔT_Vorma|)²)
# gir 71 % dekning i 68 %-båndet og 93.5 % i 95 %-båndet (mot 70.7 % / 90.0 %
# for flat σ = 0.6 - altså særlig bedre i halen).
SIGMA_BASE       = 0.45      # °C – grunnstøy innenfor datahorisonten
SIGMA_PER_DELTA  = 0.33      # °C per °C endring i Vorma under transport
SIGMA_FLOOR      = 0.15      # °C – sensorstøy; hindrer nullbredt bånd ved h = 0

# Utenfor datahorisonten vokser usikkerheten, men den METTES - den vokser ikke
# ubegrenset som √t. Ren persistens ved Fetsund (jul-aug 2015-2025, n ≈ 15 000)
# gir sd = 1.28 / 1.94 / 2.26 / 2.41 / 2.50 °C ved +24/48/72/96/120 t; altså
# en asymptote rundt 2.4-2.5 °C. Modellen skal ikke være mer usikker enn å
# ikke ha en modell i det hele tatt, så:
#       σ_ekstrap = MODEL_SIGMA_ASYMPTOTE · √(1 − e^(−ekstrapolering/τ))
# Dette treffer persistens-sd innenfor 0.15 °C på alle horisonter, litt smalere
# (som det skal være, siden modellen er bedre enn persistens).
MODEL_SIGMA_ASYMPTOTE = 2.4  # °C
SIGMA_EXTRAP_TAU      = 36.0 # t
# Utenfor datahorisonten er residualfordelingen kraftig SKJEV. Målt over 7 010
# prognosepunkter (positiv verdi = modellen predikerte for varmt):
#         h     p2.5    p16  median   p84   p97.5
#       +48    −2.06  −0.76   +0.19  +1.96  +4.38
#       +120   −2.95  −1.32   +0.32  +2.91  +6.28
# Halen på kaldsiden er dobbelt så lang. Det er ikke tilfeldig: en ukjent
# framtidig oppvellingshendelse kan bare gjøre det KALDERE, aldri varmere.
# Ekstrapoleringstillegget er derfor ensidig - bredt nedover, smalt oppover.
ANOMALY_SIGMA_EXTRAP_COLD = 3.0  # °C
ANOMALY_SIGMA_EXTRAP_WARM = 1.4  # °C
# Grunnstøy i anomaliformen. Ikke null ved h = 0 (der er prognosen målingen),
# men den vokser raskere enn en ren rampe, fordi transporttiden i seg selv er
# usikker: treffer pulsen en time før eller etter, blir feilen stor der kurven
# er bratt. Kalibrert slik at 95 %-båndet dekker ~95 % også ved +12 t.
ANOMALY_SIGMA_BASE = 0.60  # °C
ANOMALY_SIGMA_RAMP = 0.35  # andel av grunnstøyen som ligger der allerede ved h→0

# Den grunnstøyen er nødvendig for dekningsgraden, men den er tosidig og ville
# alene løftet øvre båndgrense over det uforstyrrede nivået igjen - altså gjeninn-
# føre nettopp selvmotsigelsen v1.8 fjernet. Løsningen er en FYSISK skranke, ikke
# et smalere bånd: er vannet i Vorma målt kaldere enn uforstyrret, kan Fetsund
# ikke ende varmere enn sitt eget uforstyrrede nivå. Innenfor datahorisonten
# kappes øvre grense derfor der, med et lite påslag for målestøy.
# Etter skranken overstiger øvre 95 %-grense det uforstyrrede nivået med
# median −0,41 °C over 53 kaldepisoder; bare 6 % av episodene overstiger med
# mer enn 0,5 °C, og de tilfellene er marginale dipper rundt én grad der tvilen
# er reell. Uten skranken var mediane overskridelse flere grader.
UNDISTURBED_CAP_MARGIN = 0.25  # °C – slingringsmonn for sensor- og baselinestøy
MODEL_SIGMA      = MODEL_SIGMA_ASYMPTOTE  # alias brukt i visningstekster
MODEL_SIGMA_DATA = 0.6       # Beholdt for bakoverkompatibilitet (visningstekster)

# ── Anomaliform (v1.8, standard) ─────────────────────────────────────────────
# Både inkrement- og nivåformen hadde to feil som ga seg utslag nettopp under
# kaldepisoder - altså i den ene situasjonen modellen finnes for:
#
#   1) Usikkerhetsbåndet var SYMMETRISK. Med σ ∝ |ΔT| ble øvre grense varmere
#      enn utgangspunktet så snart en stor kaldpuls var i transitt: modellen
#      «visste» at kaldt vann var på vei, men båndet sa likevel at det kunne bli
#      varmere. Målt på 51 historiske kaldepisoder skjedde det i 98 % av dem.
#      Empirisk er responsen A_Fetsund/A_Vorma negativ i 0,1 % av tilfellene
#      ved store anomalier (n = 2 248). Usikkerheten ligger i FORSTERKNINGEN,
#      ikke i fortegnet, og båndet bygges derfor nå av forsterkningskvantiler.
#
#   2) Dippen var altfor kortvarig. Inkrementformen lot anomalien relaksere mot
#      null, slik at prognosen spratt tilbake til - og over - utgangspunktet
#      etter drøyt to døgn. Måledata sier noe helt annet: etter en oppvellings-
#      hendelse finner elva et NYTT, KALDERE likevektsnivå. Andel av dippdybden
#      som står igjen, målt mot fryst førhendelsesnivå (51 episoder):
#          +12t 0.85 · +24t 0.76 · +36t 0.51 · +48t 0.47 · +72t 0.42 · +120t 0.48
#      Den planer altså ut rundt 45-48 %, den går ikke mot null.
#
# Anomaliformen:
#       B(t)   = 7-døgns 90-persentil          (uforstyrret elvetemperatur)
#       A_v(t) = T_Vorma(t) − B_Vorma(t)       (kuldeunderskudd i Vorma)
#       T(t+h) = B_Fetsund + κ·A_v(t+h−transporttid) + nivåkorreksjon
#
# 90-persentil over 7 døgn er valgt fordi den er robust mot at kaldpulsen selv
# ligger i baselinevinduet - en kaldepisode kan ikke trekke en høy persentil
# nedover slik den trekker et gjennomsnitt eller en median. Testet mot 5/7/10
# døgn og p90/p95/maks; 7 døgn p90 ga sterkest samvariasjon (r = 0,855).
BASELINE_WINDOW_HOURS = 168     # t – 7 døgn
BASELINE_QUANTILE     = 0.90

# Relaksasjon av anomalien utenfor datahorisonten. Tre ledd, tilpasset kurven
# over (RMSE 0,063): en rask komponent, en treg, og et permanent restnivå.
RELAX_TAU_FAST        = 24.0    # t
RELAX_TAU_SLOW        = 300.0   # t
RELAX_SLOW_FRACTION   = 0.20    # andel som avtar tregt
RELAX_PERSISTENT      = 0.30    # andel som IKKE forsvinner innen prognosehorisonten

# Nivåkorreksjon: median residual siste døgn. Fanger opp at Fetsunds
# uforstyrrede nivå ligger noe over Vormas, og at persentilbaselinen driver
# sakte med sesongen.
OFFSET_WINDOW_HOURS   = 24

# TESTET OG FORKASTET: å framskrive driften i det uforstyrrede nivået. Median
# drift er +0,08 °C/døgn i juli og −0,04 i august, men 5.–95. persentil spenner
# −0,47 til +0,44 - altså nesten bare støy. Ekstrapolert fem døgn fram økte den
# MAE fra 1,19 til 1,25 °C. Nivået holdes derfor fast.

# Forsterkningskvantiler, relativt til κ. Empirisk fordeling av realisert
# respons r = A_Fetsund/A_Vorma ved |A_Vorma| ≥ 1 °C (n = 10 069):
#       p5 0.18 · p16 0.39 · median 0.60 · p84 0.91 · p95 1.30
# Skalert til κ = f gir faktorene under. Fordi A_v har fortegn, tas min/max av
# de to grensene - da fungerer båndet også for en varm anomali.
GAIN_REL_68_LOW  = 0.68
GAIN_REL_68_HIGH = 1.30
GAIN_REL_95_LOW  = 0.40
GAIN_REL_95_HIGH = 1.60

# ── Eldre prognoseformer (beholdt for sammenligning og som reserve) ──────────
# 'increment' : T(t+h) = T_Fetsund(nå) + η·f · [T_Vorma(t_kilde) − T_Vorma(t_ref)]
#               Predikerer ENDRINGEN. Immun mot baseline-kontaminering, fordi
#               ingen baseline inngår. Validert MAE 0.54 °C.
# 'level'     : T(t+h) = Fetsund-baseline + κ·(T_Vorma − Vorma-baseline)
#               Den opprinnelige nivåformen. Validert MAE 0.84 °C - dårligere
#               enn ren persistens (0.74 °C), fordi Fetsund-baselinen allerede
#               inneholder kaldpulsen som Vorma-anomalien legger til på nytt.
# Sett FORECAST_MODE = 'increment' eller 'level' for å rulle tilbake.
FORECAST_MODE            = 'anomaly'   # 'anomaly' | 'increment' | 'level'
FETSUND_ANCHOR_HOURS     = 3      # t – medianvindu for ankerverdien T_Fetsund(nå)
FETSUND_ANCHOR_MAX_AGE_H = 24     # t – eldre Fetsund-data ⇒ fall tilbake til 'level'
VORMA_BASELINE_HOURS     = 72     # t – 72t MEDIAN slår 48t mean (MAE 0.68 vs 0.84)
VORMA_RELAX_HOURS        = 36     # t – e-foldingstid når anomalien ekstrapoleres
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
# Vindrisikoen skal virke KUMULATIVT og VEDVARENDE, ikke som et øyeblikksbilde.
# Fram til v1.8 ble E slått opp punktvis på hvert prognosetidspunkt, med tre
# følger som alle ga urealistiske hakk i båndet:
#   1) Passerte vindtoppen, forsvant risikoen igjen - som om kaldvannet den
#      hadde skapt ble borte. Det motsier den empiriske relaksasjonskurven, der
#      ~48 % av dippen står igjen etter fem døgn.
#   2) σ-multiplikatoren hoppet i trinn (1.0 → 1.4 → 1.8) idet E krysset en
#      terskel, og ga et synlig sprang i båndet mellom to nabopunkter.
#   3) Ved WIND_RISK_HORIZON_HOURS falt hele justeringen bort på ett tidssteg.
#      Målt sprang i nedre 68 %-grense: 3,7 °C på tre timer - den «spiken» som
#      var synlig i grafen.
# Nå bygges risikoen som et løpende maksimum med samme relaksasjon som
# temperaturanomalien, og både multiplikator og horisont tones jevnt.
WIND_RISK_FADE_HOURS = 36.0   # t – utfasing etter vindrisiko-horisonten

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
SEICHE_REBOUND_MIN       = 1.0  # °C – temperaturen må ha steget minst så mye ETTER
                                #      bunnpunktet, ellers er nedkjølingen fortsatt
                                #      pågående og det er ingen "bekreftet episode"
SEICHE_HISTORY_HOURS     = 480  # t (20 d) – bunn kan ligge 12 d tilbake, og baseline
                                #      krever 7 d FØR det ⇒ 19 d minimum historikk

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
        end_dt   = datetime.now(timezone.utc)
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

        # Fysisk områdefilter. NVEs kvalitetskoder fanger ikke alle sensorfeil -
        # Svanefoss har f.eks. levert ~-20.78 °C med kvalitetskode 1 (2021).
        if parameter == 1003:
            df = df[(df['value'] > 0.0) & (df['value'] < 35.0)]
        elif parameter == 1001:
            df = df[(df['value'] >= DISCHARGE_MIN_VALID) &
                    (df['value'] <= DISCHARGE_MAX_VALID)]

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
        end_time   = datetime.now(timezone.utc)
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


def safe_discharge(discharge_df, fallback, hours=24):
    """
    Robust medianvannføring over de siste `hours` timene.

    Returnerer (Q, kilde-etikett). Faller tilbake til `fallback` dersom serien
    mangler, er tom, eller medianen er utenfor fysisk gyldig område. Uten dette
    vernet gir ett døgn med nullstilt sensor Q = 0 → divisjon på null i
    transporttidsberegningen, og hele siden faller.
    """
    if discharge_df is not None and not discharge_df.empty:
        try:
            d = discharge_df.copy()
            d['time'] = pd.to_datetime(d['time'])
            cutoff = d['time'].max() - pd.Timedelta(hours=hours)
            recent = d[d['time'] >= cutoff]['value'].dropna()
            if len(recent) > 0:
                q = float(recent.median())
                if DISCHARGE_MIN_VALID <= q <= DISCHARGE_MAX_VALID:
                    return q, f"siste {hours}t"
        except Exception as e:
            print(f"[glommadyppen_core] safe_discharge: {e}", file=sys.stderr)
    return float(fallback), f"fallback ({fallback:.0f} m³/s)"


def mixing_fraction(vorma_q_df=None, glomma_q_df=None):
    """
    Vormas andel av vannmengden i samløpet med Glomma:

        f = Q_Vorma / (Q_Vorma + Q_Glomma)

    Q_Vorma  : Ertesekken (2.197.0), 10 km over samløpet
    Q_Glomma : Funnefoss kraftverk (2.412.0), 5,4 km over samløpet

    Massebalansen holder: median (Q_Erte + Q_Funne) / Q_Blaker = 1.03 over
    jul-aug 2015-2025, så de to stasjonene beskriver samløpet godt.

    Returnerer (f, kilde-etikett). Faller tilbake til historisk median (0.633)
    hvis Glomma-vannføring mangler.
    """
    if glomma_q_df is None or glomma_q_df.empty:
        return MIXING_FRACTION_FALLBACK, "historisk median (Glomma-data mangler)"

    qv, sv_src = safe_discharge(vorma_q_df,  FALLBACK_DISCHARGE)
    qg, gl_src = safe_discharge(glomma_q_df, FALLBACK_DISCHARGE_GLOMMA)

    if "fallback" in gl_src:
        return MIXING_FRACTION_FALLBACK, "historisk median (Glomma-data ugyldig)"

    total = qv + qg
    if total <= 0:
        return MIXING_FRACTION_FALLBACK, "historisk median (ugyldig sum)"

    f = qv / total
    if not (MIXING_FRACTION_MIN <= f <= MIXING_FRACTION_MAX):
        return MIXING_FRACTION_FALLBACK, f"historisk median (f={f:.2f} utenfor gyldig område)"

    label = f"Q_Vorma {qv:.0f} / (Q_Vorma {qv:.0f} + Q_Glomma {qg:.0f}) m³/s"
    return float(f), label


def dilution_kappa(vorma_q_df=None, glomma_q_df=None, mode='episode'):
    """
    Uttynningskoeffisienten κ beregnet fra sist målte vannføring i Vorma og
    Glomma, i stedet for som fast konstant.

        κ = η · f,   f = Q_Vorma / (Q_Vorma + Q_Glomma)

    mode='episode'   → η = 1.00 (ren blanding; reproduserer κ = 0.63 ved
                       medianvannføring, som episodeanalysen fant)
    mode='increment' → η = 0.59 (empirisk forsterkning på time-til-time-
                       endringer; lavere pga. lengdedispersjon i elva)

    Returnerer (kappa, f, kilde-etikett).
    """
    f, src = mixing_fraction(vorma_q_df, glomma_q_df)
    eta = DILUTION_ETA_INCREMENT if mode == 'increment' else DILUTION_ETA_EPISODE
    return float(eta * f), float(f), src


def undisturbed_baseline(hourly, window_hours=None, quantile=None):
    """
    «Uforstyrret» elvetemperatur: rullende høy persentil over et langt vindu.

    Poenget med en høy persentil framfor et snitt eller en median er at en
    kaldepisode ikke kan trekke den nedover. Et 48-timers gjennomsnitt blir
    kontaminert av selve pulsen man vil måle avviket fra; 90-persentilen over
    7 døgn representerer nivået elva hadde uten hendelsen.
    """
    window_hours = window_hours or BASELINE_WINDOW_HOURS
    quantile     = BASELINE_QUANTILE if quantile is None else quantile
    if hourly is None or hourly.empty:
        return None
    return hourly.rolling(f'{int(window_hours)}h',
                          min_periods=48).quantile(quantile)


def relaxation_factor(hours_ahead):
    """
    Hvor stor andel av en temperaturanomali som står igjen etter `hours_ahead`
    timer uten nye observasjoner.

        f(t) = (1 − p − q)·e^(−t/τ_rask) + p·e^(−t/τ_treg) + q

    Det siste leddet er avgjørende: q > 0 betyr at anomalien IKKE dør ut.
    Etter en oppvellingshendelse er Mjøsas epilimnion blandet, og elva legger
    seg på et nytt og kaldere nivå som holder seg i flere døgn. Målt på 51
    kaldepisoder står ~48 % av dippdybden igjen etter fem døgn. Tidligere
    relakserte modellen mot null, og spratt derfor tilbake til - og forbi -
    utgangspunktet etter drøyt to døgn.
    """
    t = np.asarray(hours_ahead, dtype=float)
    fast = (1.0 - RELAX_SLOW_FRACTION - RELAX_PERSISTENT) * np.exp(-t / RELAX_TAU_FAST)
    slow = RELAX_SLOW_FRACTION * np.exp(-t / RELAX_TAU_SLOW)
    return np.clip(fast + slow + RELAX_PERSISTENT, 0.0, 1.0)


def calculate_travel_time(discharge_df):
    """
    Beregner transporttid fra Svanefoss til Fløter'n og Fetsund.

    Koeffisienter (t = k / Q, timer), der Q = vannføring Ertesekken (Vorma):
        Fløter'n (35,5 km):  k = 7670  (avledet fra empirisk 6871 × 35.5/31.8)
        Fetsund  (45,0 km):  k = 9700  (empirisk kalibrert mot 19 kalde episoder, R²=0.73)

    Merk: koeffisientene er kalibrert MOT Ertesekken-vannføringen, ikke mot
    vannføringen nedstrøms samløpet. De skal derfor fortsatt brukes med Q_Vorma
    selv om elva går raskere etter at Glomma har kommet til.

    Returnerer (t_flotern, t_fetsund, q_used, source_label).
    """
    q, src = safe_discharge(discharge_df, FALLBACK_DISCHARGE)
    label = ("siste 24t (Ertesekken)" if src.startswith("siste")
             else f"august-median ({FALLBACK_DISCHARGE:.0f} m³/s)")
    return (round(TRANSPORT_COEFF_FLOTERN / q, 1),
            round(TRANSPORT_COEFF / q, 1),
            round(q, 0),
            label)


def _hourly_series(df, value_col='value'):
    """
    Konverterer en NVE-serie til en tz-aware Series på fast timesgrid med
    lineær interpolasjon (maks 6 t hull). Erstatter «nærmeste rad innenfor
    ±2 t»-oppslaget, som ga trappetrinn i prognosen ved grovere måleintervall.
    """
    if df is None or df.empty:
        return None
    d = df.copy()
    d['time'] = pd.to_datetime(d['time'])
    if d['time'].dt.tz is None:
        d['time'] = d['time'].dt.tz_localize('UTC')
    else:
        d['time'] = d['time'].dt.tz_convert('UTC')
    d = (d.dropna(subset=[value_col])
           .drop_duplicates(subset='time', keep='last')
           .sort_values('time')
           .set_index('time')[value_col])
    if d.empty:
        return None
    # Tving nanosekund-oppløsning. NVE/pandas kan gi 's'- eller 'us'-oppløsning,
    # og da feiler .asof() med "Cannot losslessly convert units" så snart
    # oppslagstidspunktet har en annen oppløsning (f.eks. fra en Timedelta med
    # desimaltimer). Feilen dukker bare opp i enkelte kombinasjoner, så den er
    # lett å ikke oppdage i drift.
    d.index = pd.DatetimeIndex(d.index).as_unit('ns')
    grid = pd.date_range(d.index.min().floor('h'), d.index.max().ceil('h'),
                         freq='1h', tz='UTC').as_unit('ns')
    return d.reindex(d.index.union(grid)).interpolate(
        method='time', limit=6).reindex(grid)


def detect_seiche_risk(vorma_df, hours_back_history=None):
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
        'rejected_reason': None,
    }

    if vorma_df is None or vorma_df.empty:
        return result

    hours_back_history = hours_back_history or SEICHE_HISTORY_HOURS

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

    # Krav om FAKTISK bunnpunkt. Uten dette kan minimumet ligge på selve
    # vinduskanten mens nedkjølingen fortsatt pågår - og vi ville varslet om
    # en "ettereffekt" av noe som ikke er over. Krev at temperaturen har
    # steget minst SEICHE_REBOUND_MIN etter bunnen.
    after = df[df.index > t_min_idx]['T_s']
    if after.empty or (float(after.max()) - T_min_val) < SEICHE_REBOUND_MIN:
        result['rejected_reason'] = 'ingen bekreftet oppgang etter bunnpunkt'
        return result

    # Beregn baseline: 7-dagers median FØR episoden.
    # NB: krever historikk helt tilbake til t_min − 7 d. Med t_min inntil 12 d
    # tilbake betyr det 19 døgn - derfor SEICHE_HISTORY_HOURS = 480, ikke 336.
    baseline_data = df[(df.index >= t_min_idx - timedelta(days=7)) &
                       (df.index <  t_min_idx - timedelta(hours=12))]
    if len(baseline_data) < 24:
        result['rejected_reason'] = 'for lite historikk til baseline'
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
                                fetsund_temp_df=None, glomma_q_df=None):
    """
    Predikerer temperatur ved Fløter'n / Fetsund for arrangementet (episodeform).

    Modell:
        T_pred = fetsund_baseline + (T_Vorma − Vorma-baseline) × κ

    κ beregnes nå dynamisk fra vannføringen (κ = f = Q_Vorma/(Q_Vorma+Q_Glomma))
    når glomma_q_df er tilgjengelig - se dilution_kappa(). Uten Glomma-data
    brukes historisk median f = 0.633, som gir κ ≈ den gamle konstanten 0.63.

    Vorma-baselinen er 72-timers MEDIAN (ikke 48-timers mean): median er robust
    mot at en kaldpuls passerer gjennom baselinevinduet, og 72 t slår 48 t
    empirisk (MAE 0.68 mot 0.84 °C).
    """
    if vorma_temp_df is None or vorma_temp_df.empty:
        return None
    if event_datetime.tzinfo is None:
        event_datetime = event_datetime.replace(tzinfo=pd.Timestamp.now(tz='UTC').tzinfo)

    t_flotern, travel_hours, q_used, q_source = calculate_travel_time(discharge_df)
    kappa, f_mix, kappa_src = dilution_kappa(discharge_df, glomma_q_df, mode='episode')
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

    vorma_baseline = df[
        df['time'] >= (vorma_time - timedelta(hours=VORMA_BASELINE_HOURS))
    ]['value'].median()
    anomaly = vorma_temp - vorma_baseline

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

    fetsund_temp = fetsund_baseline + anomaly * kappa

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
        'kappa':                round(kappa, 3),
        'mixing_fraction':      round(f_mix, 3),
        'kappa_source':         kappa_src,
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


# ============================================================================
# ETTERPRØVING AV PREDIKSJONSLOGGEN
# Sammenligner logget prediksjon mot faktisk observert temperatur ved Fetsund.
# Brukes av "Treffsikkerhet"-siden i appen, og er grunnlaget for å KALIBRERE
# SIGMA_BASE / SIGMA_PER_DELTA mot faktiske residualer i stedet for mot
# historiske rekonstruksjoner.
# ============================================================================

EVAL_HORIZONS   = (24, 48, 72, 96)
EVAL_TOLERANCE_H = 1.5   # t – hvor nær en observasjon må ligge gyldighetstidspunktet


def _to_num(v):
    """Robust tallkonvertering. Google Sheets CSV gir strenger, tomme celler
    og av og til desimalkomma - alt skal bli NaN eller float."""
    if v is None:
        return float('nan')
    if isinstance(v, (int, float)):
        return float(v)
    t = str(v).strip().replace(',', '.')
    if t == '' or t.lower() in ('nan', 'none', '-', '–'):
        return float('nan')
    try:
        return float(t)
    except ValueError:
        return float('nan')


def evaluate_prediction_log(log_df, fetsund_df, horizons=None,
                            tolerance_h=EVAL_TOLERANCE_H):
    """
    Kobler hver loggede prediksjon til den faktiske Fetsund-observasjonen på
    gyldighetstidspunktet (logged_at + horisont).

    Returnerer en "lang" DataFrame med én rad per (loggrad × horisont) som har
    en matchende observasjon:

        logged_at, horizon_h, valid_time, predicted, lower68, upper68,
        lower95, upper95, sigma, delta_vorma, windrisk, observed,
        error, abs_error, in68, in95

    lower95/upper95 rekonstrueres fra det loggede standardavviket:
        lower95 = lower68 − 0.96·σ,  upper95 = upper68 + 0.96·σ
    (følger direkte av at begge båndene deler samme risikoforskyvning).
    Rader logget før σ ble innført får NaN der.

    Tom DataFrame returneres hvis loggen eller observasjonene mangler - aldri
    exception, siden dette kalles direkte fra appen.
    """
    horizons = horizons or EVAL_HORIZONS
    cols = ['logged_at', 'horizon_h', 'valid_time', 'predicted', 'lower68',
            'upper68', 'lower95', 'upper95', 'sigma', 'delta_vorma',
            'windrisk', 'observed', 'error', 'abs_error', 'in68', 'in95']
    if log_df is None or log_df.empty or 'logged_at' not in log_df.columns:
        return pd.DataFrame(columns=cols)

    obs = _hourly_series(fetsund_df)
    if obs is None or obs.empty:
        return pd.DataFrame(columns=cols)

    log = log_df.copy()
    log['logged_at'] = pd.to_datetime(log['logged_at'], errors='coerce', utc=True)
    log = log.dropna(subset=['logged_at'])
    if log.empty:
        return pd.DataFrame(columns=cols)

    obs_times = obs.index
    rows = []
    for _, r in log.iterrows():
        t0 = r['logged_at']
        for h in horizons:
            pred = _to_num(r.get(f'predicted_h{h}'))
            if not np.isfinite(pred):
                continue
            valid = (t0 + pd.Timedelta(hours=h)).as_unit('ns')
            if valid > obs_times.max() or valid < obs_times.min():
                continue
            pos = obs_times.get_indexer([valid], method='nearest')[0]
            if pos < 0:
                continue
            t_obs = obs_times[pos]
            if abs((t_obs - valid).total_seconds()) > tolerance_h * 3600:
                continue
            observed = float(obs.iloc[pos])
            if not np.isfinite(observed):
                continue

            lo68  = _to_num(r.get(f'lower68_h{h}'))
            hi68  = _to_num(r.get(f'upper68_h{h}'))
            sigma = _to_num(r.get(f'sigma_h{h}'))
            lo95  = lo68 - 0.96 * sigma if np.isfinite(sigma) and np.isfinite(lo68) else float('nan')
            hi95  = hi68 + 0.96 * sigma if np.isfinite(sigma) and np.isfinite(hi68) else float('nan')

            rows.append({
                'logged_at':   t0,
                'horizon_h':   int(h),
                'valid_time':  valid,
                'predicted':   pred,
                'lower68':     lo68,
                'upper68':     hi68,
                'lower95':     lo95,
                'upper95':     hi95,
                'sigma':       sigma,
                'delta_vorma': _to_num(r.get(f'delta_vorma_h{h}')),
                'windrisk':    r.get(f'windrisk_h{h}'),
                'observed':    observed,
                'error':       pred - observed,
                'abs_error':   abs(pred - observed),
                'in68': (bool(lo68 <= observed <= hi68)
                         if np.isfinite(lo68) and np.isfinite(hi68) else None),
                'in95': (bool(lo95 <= observed <= hi95)
                         if np.isfinite(lo95) and np.isfinite(hi95) else None),
            })

    return pd.DataFrame(rows, columns=cols)


def summarize_prediction_skill(eval_df):
    """
    Oppsummerer evaluate_prediction_log() per horisont.

    Returnerer DataFrame med: horizon_h, n, mae, bias, rmse, p90_abs,
    coverage68, coverage95, sigma_mean, sigma_implied.

    `sigma_implied` er standardavviket til de faktiske residualene. Sammenlignet
    med `sigma_mean` (det modellen SA at usikkerheten var) forteller den om
    båndene er for smale eller for vide - og gir tallet SIGMA_BASE /
    SIGMA_PER_DELTA skal kalibreres mot.
    """
    empty = pd.DataFrame(columns=['horizon_h', 'n', 'mae', 'bias', 'rmse',
                                  'p90_abs', 'coverage68', 'coverage95',
                                  'sigma_mean', 'sigma_implied'])
    if eval_df is None or eval_df.empty:
        return empty

    out = []
    for h, g in eval_df.groupby('horizon_h'):
        e = g['error'].dropna()
        if e.empty:
            continue
        c68 = g['in68'].dropna()
        c95 = g['in95'].dropna()
        sg  = g['sigma'].dropna()
        out.append({
            'horizon_h':     int(h),
            'n':             int(len(e)),
            'mae':           float(e.abs().mean()),
            'bias':          float(e.mean()),
            'rmse':          float(np.sqrt((e ** 2).mean())),
            'p90_abs':       float(e.abs().quantile(0.90)),
            'coverage68':    float(c68.mean()) if len(c68) else float('nan'),
            'coverage95':    float(c95.mean()) if len(c95) else float('nan'),
            'sigma_mean':    float(sg.mean()) if len(sg) else float('nan'),
            'sigma_implied': float(e.std(ddof=1)) if len(e) > 1 else float('nan'),
        })
    return pd.DataFrame(out).sort_values('horizon_h').reset_index(drop=True) if out else empty


def prediction_history_series(log_df, horizon_h=24):
    """
    Henter ut «hva modellen sa {horizon_h} timer i forveien», som en tidsserie
    på gyldighetstidspunktet. Brukes til å tegne den stiplede historikklinjen
    i temperaturgrafen ved siden av den faktisk observerte kurven.

    Returnerer DataFrame med kolonnene time, predicted, lower68, upper68.
    """
    cols = ['time', 'predicted', 'lower68', 'upper68']
    if log_df is None or log_df.empty or 'logged_at' not in log_df.columns:
        return pd.DataFrame(columns=cols)
    col = f'predicted_h{horizon_h}'
    if col not in log_df.columns:
        return pd.DataFrame(columns=cols)

    d = log_df.copy()
    d['time'] = (pd.to_datetime(d['logged_at'], errors='coerce', utc=True)
                 + pd.Timedelta(hours=horizon_h))
    d['predicted'] = d[col].map(_to_num)
    d['lower68']   = d.get(f'lower68_h{horizon_h}', pd.Series(index=d.index)).map(_to_num)
    d['upper68']   = d.get(f'upper68_h{horizon_h}', pd.Series(index=d.index)).map(_to_num)
    d = (d.dropna(subset=['time', 'predicted'])
           .sort_values('time')
           .drop_duplicates(subset='time', keep='last'))
    return d[cols].reset_index(drop=True)


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


def _effective_wind_energy(energy_lookup, now_utc, h_step):
    """
    «Virksom» vindenergi ved prognosetidspunktet now + h_step.

    Returnerer det høyeste E som har rukket å virke fram til dette tidspunktet,
    dempet med relaxation_factor() etter at toppen passerte:

        E_eff(h) = max over alle t ≤ h av  E(t) · relaksasjon(h − t)

    Dette er den fysisk riktige formen. E er allerede et 48-timers akkumulat med
    24 timers forsinkelse, så E(t) beskriver oppvelling som slår ut ved t. Når
    den oppvellingen først har skjedd, forsvinner ikke det kalde vannet fordi
    vinden løyer - det er nettopp poenget med det permanente restleddet i
    relaksasjonsmodellen. Punktvis oppslag lot risikoen slå av igjen så snart
    vindtoppen var passert.

    Returnerer None hvis serien ikke dekker tidspunktet.
    """
    if energy_lookup is None or energy_lookup.empty:
        return None
    t_fut = now_utc + timedelta(hours=int(h_step))
    past = energy_lookup[energy_lookup['time'] <= t_fut + pd.Timedelta(minutes=90)]
    if past.empty:
        return None
    age_h = (t_fut - past['time']).dt.total_seconds() / 3600.0
    age_h = age_h.clip(lower=0.0)
    eff = past['E'].astype(float).values * relaxation_factor(age_h.values)
    if len(eff) == 0 or not np.isfinite(eff).any():
        return None
    return float(np.nanmax(eff))


def build_fetsund_forecast(vorma_df, fetsund_df, discharge_df,
                           glomma_q_df=None, hours_ahead=120, step_h=3,
                           energy_df=None, mode=None, now=None):
    """
    Tidsserie for predikert temperatur ved Fløter'n / Fetsund med
    usikkerhetsintervaller.

    ── Modell (FORECAST_MODE = 'increment', standard) ───────────────────────
        T(t+h) = T_Fetsund(nå) + η·f · [ T_Vorma(t_kilde) − T_Vorma(t_ref) ]

        t_ref   = nå − transporttid   (vannet som ankommer Fetsund nå)
        t_kilde = nå + h − transporttid (vannet som ankommer om h timer)
        f       = Q_Vorma / (Q_Vorma + Q_Glomma), fra sist målte vannføring
        η       = DILUTION_ETA_INCREMENT

    Formen predikerer ENDRINGEN i stedet for nivået, og er derfor immun mot
    baseline-kontaminering: den gamle nivåformen la Vorma-anomalien oppå en
    Fetsund-baseline som allerede inneholdt den samme kaldpulsen, og
    dobbelttalte den. Validert mot 10 552 timer jul-aug 2017-2025:

        nivåform (produksjon t.o.m. v1.6)   MAE 0.84 °C
        ren persistens (ingen modell)        MAE 0.74 °C
        inkrementform                        MAE 0.54 °C

    ── Modell (mode='level') ────────────────────────────────────────────────
        T(t+h) = Fetsund-baseline + κ·(T_Vorma(t_kilde) − Vorma-baseline)
        Den opprinnelige formen. Brukes automatisk hvis Fetsund-data mangler
        eller er eldre enn FETSUND_ANCHOR_MAX_AGE_H, og kan tvinges fram ved
        å sette FORECAST_MODE = 'level'.

    ── Usikkerhet ───────────────────────────────────────────────────────────
        σ_data = √( (SIGMA_BASE·ramp)² + (SIGMA_PER_DELTA·|ΔT_Vorma|)² )
        σ      = √( σ_data² + σ_ekstrap² ),
        σ_ekstrap = MODEL_SIGMA_ASYMPTOTE·√(1 − e^(−ekstrapolering/τ))
    Heteroskedastisk: båndet utvides automatisk når en stor temperaturendring
    er under transport. Se kommentaren ved SIGMA_BASE.

    Hvis energy_df sendes inn, skjeves og utvides båndet innenfor
    WIND_RISK_HORIZON_HOURS basert på prognosert SE/S-vindenergi.
    Sentralestimatet røres ikke (R² ≈ 0.08 - for svakt til punktestimat).

    `now` overstyrer «nåtidspunktet». Standard er faktisk klokkeslett; sett den
    eksplisitt for å BACKTESTE modellen mot historikk. Uten dette argumentet er
    funksjonen umulig å etterprøve, siden nåtiden var hardkodet internt.

    Returnerte kolonner: time, predicted, lower_68, upper_68, lower_95,
    upper_95, delta_vorma, sigma, kappa, mode, wind_E_forecast,
    wind_risk_level, is_extrapolated.
    """
    sv = _hourly_series(vorma_df)
    if sv is None or sv.empty:
        return pd.DataFrame()

    now_utc = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    if now_utc.tz is None:
        now_utc = now_utc.tz_localize('UTC')
    now_utc = now_utc.tz_convert('UTC').as_unit('ns')
    _, travel_h, _, _ = calculate_travel_time(discharge_df)
    travel_h = max(1.0, float(travel_h))

    # ── Uttynning fra sist målte vannføring ──────────────────────────────────
    kappa_inc, f_mix, kappa_src = dilution_kappa(discharge_df, glomma_q_df,
                                                 mode='increment')
    kappa_lvl, _, _             = dilution_kappa(discharge_df, glomma_q_df,
                                                 mode='episode')

    sv_last_t   = sv.index.max()
    sv_last_val = float(sv.loc[sv_last_t])
    sv_base = float(sv[sv.index >= sv_last_t -
                       pd.Timedelta(hours=VORMA_BASELINE_HOURS)].median())

    # ── Uforstyrret nivå og anomali (anomaliformen) ──────────────────────────
    sv_undist   = undisturbed_baseline(sv)
    sv_anom     = (sv - sv_undist) if sv_undist is not None else None
    sv_anom_last = 0.0
    if sv_anom is not None and sv_anom.notna().any():
        sv_anom = sv_anom.dropna()
        sv_anom_last = float(sv_anom.iloc[-1])
        sv_anom_last_t = sv_anom.index.max()
    else:
        sv_anom = None
        sv_anom_last_t = sv_last_t

    def _anom_at(t):
        """Vorma-anomali på tidspunkt t. Utenfor dataserien: relaksasjon mot
        et permanent restnivå (se relaxation_factor)."""
        t = pd.Timestamp(t).as_unit('ns')
        if sv_anom is None:
            return 0.0, True
        if t <= sv_anom_last_t:
            if t < sv_anom.index.min():
                return float(sv_anom.iloc[0]), True
            v = sv_anom.asof(t)
            return (float(v), False) if pd.notna(v) else (0.0, True)
        dt = (t - sv_anom_last_t).total_seconds() / 3600.0
        return sv_anom_last * float(relaxation_factor(dt)), True

    def _vorma_at(t):
        """T_Vorma på tidspunkt t (brukt av inkrement- og nivåformen)."""
        t = pd.Timestamp(t).as_unit('ns')
        if t <= sv_last_t and t >= sv.index.min():
            v = sv.asof(t)
            if pd.notna(v):
                return float(v), False
        if t < sv.index.min():
            return float(sv.iloc[0]), True
        extrap_h = (t - sv_last_t).total_seconds() / 3600.0
        return (sv_base + (sv_last_val - sv_base) *
                float(np.exp(-extrap_h / VORMA_RELAX_HOURS))), True

    # ── Anker: Fetsund nå (robust 3t-median, ikke ett enkelt punkt) ──────────
    anchor      = None
    anchor_age  = None
    fetsund_baseline = None
    fe_undist_now    = None
    fe_h = _hourly_series(fetsund_df)
    if fe_h is not None and not fe_h.empty:
        fe_last_t = fe_h.index.max()
        anchor_age = (now_utc - fe_last_t).total_seconds() / 3600.0
        anchor = float(fe_h[fe_h.index >= fe_last_t -
                            pd.Timedelta(hours=FETSUND_ANCHOR_HOURS)].median())
        fetsund_baseline = float(fe_h[fe_h.index >= fe_last_t -
                                      pd.Timedelta(hours=48)].median())
        fe_ud = undisturbed_baseline(fe_h)
        if fe_ud is not None and fe_ud.notna().any():
            fe_undist_now = float(fe_ud.dropna().iloc[-1])

    use_mode = mode or FORECAST_MODE
    if (anchor is None or not np.isfinite(anchor)
            or anchor_age is None or anchor_age > FETSUND_ANCHOR_MAX_AGE_H):
        use_mode = 'level'
    if use_mode == 'anomaly' and (fe_undist_now is None or sv_anom is None):
        use_mode = 'increment'
    if fetsund_baseline is None or not np.isfinite(fetsund_baseline):
        fetsund_baseline = sv_base + 1.5
    if anchor is None or not np.isfinite(anchor):
        anchor = fetsund_baseline

    # ── Nivåkorreksjon for anomaliformen ────────────────────────────────────
    offset = 0.0
    if use_mode == 'anomaly':
        resid = []
        fe_ud_series = undisturbed_baseline(fe_h)
        for s_t in pd.date_range(fe_last_t - pd.Timedelta(hours=OFFSET_WINDOW_HOURS),
                                 fe_last_t, freq='3h'):
            if s_t in fe_h.index and pd.notna(fe_ud_series.get(s_t, np.nan)):
                a_src, _ = _anom_at(s_t - pd.Timedelta(hours=travel_h))
                resid.append(float(fe_h.at[s_t]) - float(fe_ud_series.at[s_t])
                             - kappa_lvl * a_src)
        if resid:
            offset = float(np.median(resid))

    # Referansepunkt for inkrementformen
    sv_ref, _ = _vorma_at(now_utc - pd.Timedelta(hours=travel_h))

    # ── Vindenergi-prognose for oppslag ─────────────────────────────────────
    energy_lookup = None
    if energy_df is not None and not energy_df.empty:
        energy_lookup = energy_df[['time', 'E', 'is_forecast']].dropna(subset=['time']).copy()
        energy_lookup['time'] = pd.to_datetime(energy_lookup['time'])
        if energy_lookup['time'].dt.tz is None:
            energy_lookup['time'] = energy_lookup['time'].dt.tz_localize('UTC')
        energy_lookup = energy_lookup.sort_values('time').reset_index(drop=True)

    rows = []
    for h_step in range(0, int(hours_ahead) + 1, int(step_h)):
        t_fut  = now_utc + timedelta(hours=h_step)
        t_src  = t_fut - timedelta(hours=travel_h)

        band_lo = band_hi = None

        if use_mode == 'anomaly':
            a_src, is_extrap = _anom_at(t_src)
            k_used  = kappa_lvl
            level   = fe_undist_now
            model   = level + k_used * a_src + offset
            # Ved h = 0 VET vi temperaturen; da skal prognosen være målingen.
            # Vekt gradvis over til anomalimodellen fram til datahorisonten.
            alpha   = min(1.0, h_step / travel_h)
            pred    = anchor * (1.0 - alpha) + model * alpha
            blend   = lambda v: anchor * (1.0 - alpha) + v * alpha
            d_vorma = a_src
            # Asymmetrisk bånd fra forsterkningskvantilene. Fordi anomalien har
            # fortegn, tas min/max - da holder båndet også for varm anomali.
            g68 = (k_used * GAIN_REL_68_LOW * a_src, k_used * GAIN_REL_68_HIGH * a_src)
            g95 = (k_used * GAIN_REL_95_LOW * a_src, k_used * GAIN_REL_95_HIGH * a_src)
            band_lo = (blend(level + min(g68) + offset),
                       blend(level + min(g95) + offset))
            band_hi = (blend(level + max(g68) + offset),
                       blend(level + max(g95) + offset))
        elif use_mode == 'increment':
            sv_src, is_extrap = _vorma_at(t_src)
            d_vorma = sv_src - sv_ref
            pred    = anchor + kappa_inc * d_vorma
            k_used  = kappa_inc
        else:
            sv_src, is_extrap = _vorma_at(t_src)
            d_vorma = sv_src - sv_base
            raw     = fetsund_baseline + d_vorma * kappa_lvl
            alpha   = min(1.0, h_step / travel_h)
            pred    = anchor * (1.0 - alpha) + raw * alpha
            k_used  = kappa_lvl

        # ── Usikkerhet ──────────────────────────────────────────────────────
        ramp    = min(1.0, h_step / travel_h)
        extrap  = max(0.0, h_step - travel_h)
        ext_ramp = float(np.sqrt(1.0 - np.exp(-extrap / SIGMA_EXTRAP_TAU)))
        if use_mode == 'anomaly':
            # Forsterkningsusikkerheten ligger allerede i det asymmetriske
            # båndet; her legges bare måle- og relaksasjonsstøy til, pluss et
            # ENSIDIG ekstrapoleringstillegg (se kommentaren ved konstantene).
            sigma_data  = max(SIGMA_FLOOR, ANOMALY_SIGMA_BASE *
                              (ANOMALY_SIGMA_RAMP + (1.0 - ANOMALY_SIGMA_RAMP) * ramp))
            sigma_cold  = float(np.hypot(sigma_data, ANOMALY_SIGMA_EXTRAP_COLD * ext_ramp))
            sigma_warm  = float(np.hypot(sigma_data, ANOMALY_SIGMA_EXTRAP_WARM * ext_ramp))
        else:
            sigma_data = max(SIGMA_FLOOR,
                             float(np.hypot(SIGMA_BASE * ramp,
                                            SIGMA_PER_DELTA * abs(d_vorma))))
            sigma_cold = sigma_warm = float(np.hypot(
                sigma_data, MODEL_SIGMA_ASYMPTOTE * ext_ramp))
        sigma = max(sigma_cold, sigma_warm)

        # ── Vindrisiko-justering ────────────────────────────────────────────
        # E_eff er ikke E på dette tidspunktet, men det høyeste E som har
        # rukket å virke fram til hit, relaksert med samme kurve som
        # temperaturanomalien. En vindtopp som har passert fortsetter altså å
        # holde båndet åpent nedover, i stedet for å slå av og på.
        e_fc, risk_level = None, None
        sigma_mult, risk_shift = 1.0, 0.0
        if energy_lookup is not None:
            e_fc = _effective_wind_energy(energy_lookup, now_utc, h_step)
        if e_fc is not None:
            # Jevn utfasing i stedet for en klippekant ved horisonten
            if h_step <= WIND_RISK_HORIZON_HOURS:
                horizon_w = 1.0
            else:
                horizon_w = float(np.exp(
                    -(h_step - WIND_RISK_HORIZON_HOURS) / WIND_RISK_FADE_HOURS))

            # Kontinuerlig multiplikator - ingen trinn ved tersklene
            if e_fc <= ENERGY_WARN:
                mult = 1.0 + (WIND_SIGMA_MULT_WARN - 1.0) * max(0.0, e_fc / ENERGY_WARN)
                risk_level = 'lav'
            elif e_fc <= ENERGY_THRESHOLD:
                frac = (e_fc - ENERGY_WARN) / max(ENERGY_THRESHOLD - ENERGY_WARN, 1e-6)
                mult = WIND_SIGMA_MULT_WARN + frac * (WIND_SIGMA_MULT_ALARM - WIND_SIGMA_MULT_WARN)
                risk_level = 'advarsel'
            else:
                mult = WIND_SIGMA_MULT_ALARM
                risk_level = 'alarm'

            sigma_mult = 1.0 + (mult - 1.0) * horizon_w
            # Kun nedsiderisiko - vind gir aldri grunnlag for å anta varmere.
            risk_shift = horizon_w * min(
                0.0, WIND_ANOMALY_SLOPE * (e_fc - WIND_ANOMALY_E_TYPISK))
            e_fc = round(e_fc, 1)

        # Vindrisiko utvider kun kaldsiden - vind gir aldri grunnlag for å anta
        # varmere vann.
        cold_eff  = sigma_cold * sigma_mult
        warm_eff  = sigma_warm
        sigma_eff = max(cold_eff, warm_eff)

        base_lo, base_hi = (band_lo, band_hi) if band_lo is not None else \
                           ((pred, pred), (pred, pred))
        lo68 = base_lo[0] + risk_shift - cold_eff
        hi68 = base_hi[0] + warm_eff
        lo95 = base_lo[1] + risk_shift - 1.96 * cold_eff
        hi95 = base_hi[1] + 1.96 * warm_eff

        # Fysisk skranke: er kaldvannet OBSERVERT i Vorma, kan Fetsund ikke ende
        # varmere enn sitt eget uforstyrrede nivå. Gjelder bare innenfor
        # datahorisonten - utenfor den vet vi ikke hva som kommer.
        if (use_mode == 'anomaly' and not is_extrap
                and d_vorma < 0 and fe_undist_now is not None):
            cap = fe_undist_now + offset + UNDISTURBED_CAP_MARGIN
            hi68 = min(hi68, cap)
            hi95 = min(hi95, cap)
            # Behold båndet velformet dersom skranken biter hardt
            hi68 = max(hi68, pred)
            hi95 = max(hi95, hi68)

        rows.append({
            'time':            t_fut,
            'predicted':       round(pred, 2),
            'lower_68':        round(max(lo68, TEMP_HIST_LOWER), 2),
            'upper_68':        round(min(hi68, TEMP_HIST_UPPER), 2),
            'lower_95':        round(max(lo95, TEMP_HIST_LOWER), 2),
            'upper_95':        round(min(hi95, TEMP_HIST_UPPER), 2),
            'delta_vorma':     round(d_vorma, 2),
            'sigma':           round(sigma_eff, 2),
            'kappa':           round(k_used, 3),
            'mode':            use_mode,
            'is_extrapolated': bool(is_extrap),
            'wind_E_forecast': e_fc,
            'wind_risk_level': risk_level,
        })

    out = pd.DataFrame(rows)
    out.attrs['mixing_fraction'] = round(f_mix, 3)
    out.attrs['kappa_source']    = kappa_src
    out.attrs['undisturbed_level'] = (round(fe_undist_now, 2)
                                      if fe_undist_now is not None else None)
    out.attrs['travel_hours']    = travel_h
    return out


__all__ = ['CORE_VERSION', 'NVE_BASE_URL', 'FROST_CLIENT_ID', 'FROST_BASE_URL', 'STATION_SVANEFOSS', 'STATION_FUNNEFOSS_TEMP', 'STATION_ERTESEKKEN_Q', 'STATION_BLAKER', 'STATION_FUNNEFOSS_Q', 'STATION_FETSUND', 'FROST_STATION_KISE', 'MJOSA_LAT', 'MJOSA_LON', 'BINGSFOSSEN_LAT', 'BINGSFOSSEN_LON', 'FETSUND_LAT', 'FETSUND_LON', 'TRANSPORT_COEFF', 'TRANSPORT_COEFF_BLA', 'TRANSPORT_COEFF_FLOTERN', 'FALLBACK_DISCHARGE', 'TEMPERATURE_SURVIVAL', 'MIXING_FRACTION_FALLBACK', 'DILUTION_ETA_EPISODE', 'DILUTION_ETA_INCREMENT', 'DISCHARGE_MIN_VALID', 'DISCHARGE_MAX_VALID', 'FALLBACK_DISCHARGE_GLOMMA', 'SIGMA_BASE', 'SIGMA_PER_DELTA', 'SIGMA_FLOOR', 'MODEL_SIGMA_ASYMPTOTE', 'SIGMA_EXTRAP_TAU', 'ANOMALY_SIGMA_EXTRAP_COLD', 'ANOMALY_SIGMA_EXTRAP_WARM', 'ANOMALY_SIGMA_BASE', 'ANOMALY_SIGMA_RAMP', 'UNDISTURBED_CAP_MARGIN', 'FORECAST_MODE', 'BASELINE_WINDOW_HOURS', 'BASELINE_QUANTILE', 'RELAX_TAU_FAST', 'RELAX_TAU_SLOW', 'RELAX_SLOW_FRACTION', 'RELAX_PERSISTENT', 'OFFSET_WINDOW_HOURS', 'GAIN_REL_68_LOW', 'GAIN_REL_68_HIGH', 'GAIN_REL_95_LOW', 'GAIN_REL_95_HIGH', 'VORMA_BASELINE_HOURS', 'VORMA_RELAX_HOURS', 'MODEL_SIGMA', 'MODEL_SIGMA_DATA', 'TEMP_HIST_LOWER', 'TEMP_HIST_UPPER', 'WIND_SECTOR_MIN', 'WIND_SECTOR_MAX', 'WIND_WINDOW_HOURS', 'WIND_LEAD_HOURS', 'CRITICAL_WIND_SPEED', 'ENERGY_THRESHOLD', 'ENERGY_WARN', 'WIND_RISK_HORIZON_HOURS', 'WIND_ANOMALY_SLOPE', 'WIND_ANOMALY_E_TYPISK', 'WIND_SIGMA_MULT_WARN', 'WIND_SIGMA_MULT_ALARM', 'SEICHE_WINDOW_START_DAYS', 'SEICHE_WINDOW_END_DAYS', 'SEICHE_COLD_THRESHOLD', 'SEICHE_ANOMALY_MIN', 'SEICHE_REBOUND_MIN', 'SEICHE_HISTORY_HOURS', 'OW_ABORT', 'OW_WETSUIT_REQUIRED', 'OW_WETSUIT_STRONG', 'OW_WETSUIT_OPTIONAL', 'OW_TOO_WARM', 'EVENT_YEAR', 'EVENT_MONTH', 'EVENT_DAY_OF_WEEK', 'fetch_nve_data', 'fetch_frost_wind', 'fetch_weather_forecast', 'add_southerly_component', 'detect_temperature_drop', 'calculate_travel_time', 'detect_seiche_risk', 'predict_fetsund_temperature', 'assess_risk_open_water', 'calculate_event_date', 'wind_rose_label', 'safe_discharge', 'mixing_fraction', 'dilution_kappa', 'undisturbed_baseline', 'relaxation_factor', 'build_wind_energy_series', 'build_fetsund_forecast', 'read_prediction_log', 'evaluate_prediction_log', 'summarize_prediction_skill', 'prediction_history_series', 'EVAL_HORIZONS', 'PREDICTION_LOG_SHEET_ID', 'PREDICTION_LOG_WORKSHEET']
