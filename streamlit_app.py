"""
GlommaDyppen Vanntemperatur Prediksjon
Real-time water temperature prediction for GlommaDyppen swimming event

Author: Anton Vooren
Date: 2026
"""

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Delt kjernemodul (datahenting + prediksjonsmodell) - brukes også av
# log_prediction.py (GitHub Actions-cronjobb) for å garantere at appen og
# loggingen alltid kjører nøyaktig samme modellogikk. Se glommadyppen_core.py.
import glommadyppen_core as _core
from glommadyppen_core import *  # noqa: F401,F403 - konstanter + modellfunksjoner

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="GlommaDyppen Temperatur",
    page_icon="🏊‍♂️",
    layout="wide",
    initial_sidebar_state="auto"
)

# ============================================================================
# VERSJONSSJEKK AV KJERNEMODULEN
# streamlit_app.py og glommadyppen_core.py må rulles ut sammen. Blir bare den
# ene oppdatert, feiler appen med en NameError som Streamlit Cloud sladder
# ("original error message is redacted to prevent data leaks") - praktisk talt
# umulig å feilsøke fra brukersiden. Denne sjekken bytter den ut med en presis
# melding om nøyaktig hva som er ute av synk.
# ============================================================================

REQUIRED_CORE_VERSION = "1.8.0"

_REQUIRED_CORE_ATTRS = [
    # v1.7 - dynamisk uttynning og robusthet
    "CORE_VERSION", "undisturbed_baseline", "relaxation_factor", "dilution_kappa", "mixing_fraction", "safe_discharge",
    "SEICHE_HISTORY_HOURS", "SIGMA_BASE", "SIGMA_PER_DELTA",
    "MODEL_SIGMA_ASYMPTOTE", "VORMA_RELAX_HOURS", "FORECAST_MODE",
    # v1.7 - etterprøving av prediksjonsloggen
    "EVAL_HORIZONS", "evaluate_prediction_log", "summarize_prediction_skill",
    "prediction_history_series",
]


def _core_fingerprint():
    """
    Identifiserer NØYAKTIG hvilken glommadyppen_core.py som faktisk er lastet.

    Ligger det to kopier av filen i repoet, eller bygger Streamlit Cloud fra en
    annen branch enn man tror, sier versjonsnummeret alene ingenting om hvor
    filen kom fra. Stien gjør det.
    """
    import os
    from datetime import datetime as _dt
    info = {"sti": getattr(_core, "__file__", "ukjent")}
    try:
        stt = os.stat(info["sti"])
        info["størrelse"] = f"{stt.st_size} byte"
        info["endret"] = _dt.fromtimestamp(stt.st_mtime).strftime("%d.%m.%Y %H:%M")
        with open(info["sti"], encoding="utf-8") as fh:
            lines = fh.readlines()
        info["linjer"] = str(len(lines))
        hit = next((f"linje {i}: {ln.strip()}" for i, ln in enumerate(lines, 1)
                    if ln.startswith("CORE_VERSION")), None)
        info["CORE_VERSION"] = hit or "★ finnes ikke i filen ★"
    except OSError as e:
        info["lesefeil"] = str(e)
    return info


def _check_core_version():
    """Stopper appen med en lesbar melding hvis kjernemodulen er utdatert."""
    missing = [n for n in _REQUIRED_CORE_ATTRS if not hasattr(_core, n)]
    found   = getattr(_core, "CORE_VERSION", None)

    sig_ok = True
    try:
        import inspect
        sig_ok = 'glomma_q_df' in inspect.signature(
            _core.build_fetsund_forecast).parameters
    except (AttributeError, ValueError, TypeError):
        sig_ok = False

    if not missing and sig_ok and found == REQUIRED_CORE_VERSION:
        return

    st.error(
        f"**glommadyppen_core.py er ikke i synk med streamlit_app.py.**\n\n"
        f"Appen krever kjerneversjon `{REQUIRED_CORE_VERSION}`, men fant "
        f"`{found or 'ingen versjon (eldre enn 1.7.0)'}`.",
        icon="🔧",
    )

    st.markdown("**Dette er filen appen faktisk leste:**")
    st.code("\n".join(f"{k:14s} {v}" for k, v in _core_fingerprint().items()),
            language=None)
    st.markdown(
        "Sjekk at nettopp *denne* filen er den du lastet opp. Riktig fil har "
        "`CORE_VERSION = \"1.8.0\"` på linje 30 og er cirka 1 340 linjer lang. "
        "Ligger det flere kopier i repoet, er det stien over som gjelder."
    )

    if missing:
        with st.expander(f"{len(missing)} navn mangler i kjernemodulen"):
            st.code("\n".join(missing))
            if not sig_ok:
                st.markdown(
                    "I tillegg mangler `build_fetsund_forecast()` argumentet "
                    "`glomma_q_df`."
                )
    st.stop()


# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

try:
    NVE_API_KEY = st.secrets["nve_api_key"]
except (KeyError, FileNotFoundError):
    NVE_API_KEY = None


# ============================================================================
# DATA FETCHING
# Tynne, Streamlit-cachede wrappere rundt glommadyppen_core sine funksjoner.
# All faktisk hente-logikk bor i glommadyppen_core.py.
# ============================================================================

@st.cache_data(ttl=3600)
def fetch_nve_data(station_id, parameter, hours_back=168):
    return _core.fetch_nve_data(station_id, parameter, hours_back, api_key=NVE_API_KEY)


@st.cache_data(ttl=3600)
def fetch_frost_wind(hours_back=168):
    return _core.fetch_frost_wind(hours_back)


@st.cache_data(ttl=1800)
def read_prediction_log():
    """Prediksjonsloggen fra Google Sheets. Cachet i 30 min - den oppdateres
    bare én gang i døgnet av GitHub Actions-jobben."""
    return _core.read_prediction_log()


@st.cache_data(ttl=21600)
def fetch_weather_forecast(lat, lon, days_ahead=14):
    return _core.fetch_weather_forecast(lat, lon, days_ahead)



# ============================================================================
# VISUALIZATION HELPERS
# (analyse-/prediksjonsfunksjoner ligger i glommadyppen_core.py)
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
# MOBILE CSS
# Injiserer responsiv CSS slik at 4-kolonner brytes til 2×2 på smale skjermer,
# heading-størrelser skaleres ned og tabeller får horisontal scroll.
# ============================================================================

def _inject_mobile_css():
    """Inject responsive CSS for better mobile/tablet display."""
    st.markdown("""
    <style>
    /* ═══════════════════════════════════════════════════════════════════════
       GlommaDyppen – mobilvisning
       Tillater at Streamlit-kolonner brytes over flere linjer på smale
       skjermer, slik at 4-kolonne layouts blir til 2×2 (og 1 kolonne
       på svært smale skjermer).
    ═══════════════════════════════════════════════════════════════════════ */

    @media screen and (max-width: 768px) {

        /* ── Kolonne-wrapping ─────────────────────────────────────────── */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.25rem 0.5rem !important;
        }
        [data-testid="column"] {
            min-width: calc(48% - 0.5rem) !important;
            flex: 1 1 calc(48% - 0.5rem) !important;
        }

        /* ── Hoved-padding ────────────────────────────────────────────── */
        .main .block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-top: 0.75rem !important;
            max-width: 100% !important;
        }

        /* ── Overskrifter ─────────────────────────────────────────────── */
        h1 { font-size: 1.35rem !important; line-height: 1.25 !important; }
        h2 { font-size: 1.15rem !important; }
        h3 { font-size: 1.05rem !important; }

        /* ── Metric-widgets ───────────────────────────────────────────── */
        [data-testid="stMetricValue"] {
            font-size: 1.05rem !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.68rem !important;
            line-height: 1.2 !important;
        }
        [data-testid="stMetricDelta"] {
            font-size: 0.65rem !important;
        }

        /* ── Dataframe – horisontal scroll på mobil ───────────────────── */
        [data-testid="stDataFrame"] > div {
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
        }

        /* ── Tabs: mindre font ────────────────────────────────────────── */
        [data-testid="stTabs"] [data-testid="stMarkdownContainer"] p {
            font-size: 0.85rem !important;
        }
        button[data-baseweb="tab"] {
            padding: 6px 10px !important;
            font-size: 0.80rem !important;
        }

        /* ── Caption / expander ───────────────────────────────────────── */
        [data-testid="stCaptionContainer"] p {
            font-size: 0.76rem !important;
        }
        [data-testid="stExpander"] summary {
            font-size: 0.88rem !important;
        }

        /* ── Mobile-navigasjonshint: vis ──────────────────────────────── */
        .gd-mobile-hint {
            display: flex !important;
            align-items: center;
            justify-content: center;
            gap: 6px;
            background: #eef3fb;
            border-radius: 8px;
            padding: 7px 14px;
            font-size: 0.82rem;
            color: #4472C4;
            margin-bottom: 0.75rem;
            border: 1px solid #c8d8f0;
        }
    }

    /* Skjul hint på desktop */
    .gd-mobile-hint { display: none; }

    @media screen and (max-width: 480px) {
        /* Svært smal skjerm (eldre telefoner): én kolonne */
        [data-testid="column"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        h1 { font-size: 1.15rem !important; }
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# WIND ENERGY FUNCTIONS
# ============================================================================



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



def _forecast_chart(fetsund_obs_df, forecast_df, travel_hours,
                    history_df=None, history_horizon=24,
                    title="Temperaturprognose – Fløter'n / Fetsund"):
    """
    Kombinert graf: historiske Fetsund-målinger + prediksjon med usikkerhetsbånd.

    `history_df` (fra core.prediction_history_series) tegnes som en stiplet
    linje bakover i tid: hva modellen SA `history_horizon` timer i forveien,
    plassert på gyldighetstidspunktet. Avstanden mellom den stiplede linjen og
    den heltrukne observasjonslinjen er treffsikkerheten, direkte avlesbar.
    """
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
        forecast_df = forecast_df.copy()
        if 'wind_E_forecast' in forecast_df.columns:
            forecast_df['wind_E_forecast'] = forecast_df['wind_E_forecast'].apply(
                lambda v: f"{v:.1f} m·h" if pd.notna(v) else "ingen prognose")
            forecast_df['wind_risk_level'] = forecast_df['wind_risk_level'].fillna('–')

        # Filtrer ut rader der KI-båndet har nullbredde (sigma=0 ved t=0),
        # ellers tegner Plotly fill='toself'-polygoner som usynlige linjer.
        band_df = forecast_df[forecast_df['upper_95'] > forecast_df['lower_95']].copy()
        t_fwd = list(band_df['time'])
        t_rev = list(band_df['time'])[::-1]
        if t_fwd:
            fig.add_trace(go.Scatter(
                x=t_fwd + t_rev,
                y=list(band_df['upper_95']) + list(band_df['lower_95'])[::-1],
                fill='toself', fillcolor='rgba(56,141,228,0.10)',
                line=dict(color='rgba(0,0,0,0)', width=0),
                name='95 % område', hoverinfo='skip',
            ))
        if t_fwd:
            fig.add_trace(go.Scatter(
                x=t_fwd + t_rev,
                y=list(band_df['upper_68']) + list(band_df['lower_68'])[::-1],
                fill='toself', fillcolor='rgba(56,141,228,0.22)',
                line=dict(color='rgba(0,0,0,0)', width=0),
                name='68 % område', hoverinfo='skip',
            ))
        hover_cols = ['lower_68', 'upper_68', 'lower_95', 'upper_95']
        hover_template = (
            '<b>Dipp Prediksjon</b>: %{y:.1f} °C<br>'
            '68 %: %{customdata[0]:.1f}–%{customdata[1]:.1f} °C<br>'
            '95 %: %{customdata[2]:.1f}–%{customdata[3]:.1f} °C'
        )
        if 'wind_E_forecast' in forecast_df.columns:
            hover_cols += ['wind_E_forecast', 'wind_risk_level']
            hover_template += (
                '<br>Vindenergi (varsel): %{customdata[4]} (%{customdata[5]})'
            )
        hover_template += '<extra></extra>'

        fig.add_trace(go.Scatter(
            x=forecast_df['time'], y=forecast_df['predicted'],
            mode='lines', name='Prediksjon',
            line=dict(color='#185FA5', width=2, dash='dash'),
            customdata=forecast_df[hover_cols].values,
            hovertemplate=hover_template,
        ))

        if 'wind_risk_level' in forecast_df.columns:
            risk_end = forecast_df[forecast_df['wind_risk_level'].notna()]
            if not risk_end.empty:
                risk_horizon_ms = risk_end['time'].max().timestamp() * 1000
                fig.add_vline(
                    x=risk_horizon_ms, line_dash='dot',
                    line_color='rgba(186,117,23,0.45)', line_width=1,
                    annotation_text='Vindrisiko-horisont', annotation_position='bottom right',
                    annotation_font_size=10, annotation_font_color='rgba(186,117,23,0.75)',
                )

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

    # ── Uforstyrret nivå: referansen dippen måles mot ────────────────────────
    _undist = forecast_df.attrs.get('undisturbed_level') if forecast_df is not None else None
    if _undist is not None:
        fig.add_hline(
            y=_undist, line_dash='dot', line_width=1.4,
            line_color='rgba(24,95,165,0.55)',
            annotation_text=f"uforstyrret nivå {_undist:.1f} °C",
            annotation_position='top left', annotation_font_size=10,
        )

    # ── Tidligere prediksjoner (fasit-linjen) ────────────────────────────────
    if history_df is not None and not history_df.empty:
        hd = history_df.copy()
        hd['time'] = pd.to_datetime(hd['time'])
        if hd['time'].dt.tz is None:
            hd['time'] = hd['time'].dt.tz_localize('UTC')
        # Begrens til samme tidsrom som observasjonene, slik at grafen ikke
        # zoomer ut til hele loggens levetid.
        if fetsund_obs_df is not None and not fetsund_obs_df.empty:
            _t0 = pd.to_datetime(fetsund_obs_df['time']).min()
            if _t0.tzinfo is None:
                _t0 = _t0.tz_localize('UTC')
            hd = hd[hd['time'] >= _t0]
        if not hd.empty:
            fig.add_trace(go.Scatter(
                x=hd['time'], y=hd['predicted'],
                mode='lines+markers',
                name=f'Predikert {history_horizon} t i forveien',
                line=dict(color='#BA7517', width=1.8, dash='dash'),
                marker=dict(size=4),
                hovertemplate=(f'<b>Prediksjon {history_horizon} t i forveien</b>'
                               ': %{y:.1f} °C<extra></extra>'),
            ))

    fig.update_layout(
        title=title, xaxis_title='',
        yaxis=dict(title='°C', range=[8, 25], fixedrange=True),
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
    # Samme prinsipp som logoen: en manglende bildefil skal ikke ta ned siden.
    try:
        st.image(
            "kart_malestasjoner.png",
            caption=(
                "Oversikt over NVE-målestasjoner langs Vorma og Glomma med "
                "GPS-koordinater og elveavstander fra Minnesund. "
                "Kilde: Anton Vooren / Fet Svømmeklubb."
            ),
            use_container_width=True,
        )
    except Exception:
        st.info("Kartet over målestasjoner er ikke tilgjengelig.")

    st.subheader("Prediksjonsmodell")
    st.markdown("""
    Modellen beskriver kuldeunderskuddet i elva i forhold til det nivået den ville
    hatt uten hendelsen:

    **T(t+h) = uforstyrret nivå + κ · Vorma-anomali(t + h − transporttid)**

    Det uforstyrrede nivået er 90-persentilen over sju døgn. En høy persentil er
    valgt framfor et gjennomsnitt eller en median fordi en kaldepisode ikke kan
    trekke den nedover — et 48-timers gjennomsnitt blir kontaminert av selve pulsen
    man vil måle avviket fra. Testet mot 5, 7 og 10 døgn og p90, p95 og maks; sju
    døgn med p90 ga sterkest samvariasjon mellom stasjonene (r = 0,855).

    #### Uttynningen κ beregnes, den er ikke en konstant

    κ er ikke en fri modellparameter — den **er** Vormas blandingsandel i samløpet
    med Glomma:

    **κ = Q_Vorma / (Q_Vorma + Q_Glomma)**

    målt ved Ertesekken (2.197.0) og Funnefoss kraftverk (2.412.0). Medianen over
    juli–august 2017–2025 er **0,633** — praktisk talt identisk med den empiriske
    episode-κ = 0,63 fra 35 kalde episoder. Den sammenfallende verdien er den
    sterkeste bekreftelsen på at mekanismen er ren blanding.

    Andelen varierer 0,51–0,76 mellom 5. og 95. persentil i normale somre, og kan
    falle langt lavere i et Glomma-flomår. Modellen leser derfor vannføringen ved
    hver kjøring i stedet for å låse κ til 0,63. Mangler Glomma-data, brukes den
    historiske medianen.

    Prediksjonen gjelder startpunktet til **Fløter'n** (Glommadyppen), 35,5 km fra Svanefoss
    og slutpunktet 11 km nedstrøms i Glomma. Temperaturen er i praksis lik ved begge punkter —
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
        - Utenfor datahorisonten metter usikkerheten mot σ ≈ 2,4 °C
        - Båndene er risikojusterte, ikke symmetriske konfidensintervaller
        """)

    st.markdown("""
    #### To ting modellen tok feil av før v1.8

    **Båndet motsa seg selv.** Med et symmetrisk ± σ ble øvre grense varmere enn
    utgangspunktet så snart en stor kaldpuls var i transitt. Modellen «visste» at
    kaldt vann var på vei, men båndet sa samtidig at det kunne bli varmere. Målt
    på 51 historiske kaldepisoder skjedde det i 43 % av dem selv der kaldvannet
    allerede var **observert** i Vorma. Empirisk er responsen
    A_Fetsund / A_Vorma negativ i 0,1 % av tilfellene ved store anomalier
    (n = 2 248) – usikkerheten ligger i hvor kraftig utslaget blir, ikke i
    fortegnet. Båndet bygges derfor nå av forsterkningskvantiler:

    | | κ-faktor | ved κ = 0,63 |
    |---|---|---|
    | 68 %-området | 0,68 – 1,30 | 0,43 – 0,82 |
    | 95 %-området | 0,40 – 1,60 | 0,25 – 1,01 |

    Andelen episoder der båndet motsier sin egen dipp innenfor datahorisonten er
    dermed nede i 11 %, og de gjenværende tilfellene er marginale dipper rundt
    én grad, der tvilen er reell.

    #### Dippen varer nå like lenge som i virkeligheten

    Den gamle modellen lot anomalien relaksere mot null, så prognosen spratt
    tilbake til – og forbi – utgangspunktet etter drøyt to døgn. Måledata sier
    noe helt annet. Andel av dippdybden som står igjen, målt mot fryst
    førhendelsesnivå over 51 episoder:

    | Etter | +12 t | +24 t | +36 t | +48 t | +72 t | +120 t |
    |---|---|---|---|---|---|---|
    | Igjen av dippen | 0,85 | 0,76 | 0,51 | 0,47 | 0,42 | **0,48** |

    Den planer ut rundt 45–48 %. Den går ikke mot null: etter en oppvelling er
    Mjøsas epilimnion blandet, og elva legger seg på et nytt og kaldere nivå som
    holder seg i flere døgn. Relaksasjonen har derfor tre ledd – en rask
    komponent (τ = 24 t), en treg (τ = 300 t) og et permanent restnivå på 30 %.

    Resultat på de samme 51 episodene, alt målt mot uforstyrret nivå:

    | | v1.6 | v1.7 | **v1.8** | Fasit |
    |---|---|---|---|---|
    | Dippdybde | 3,4 °C | 2,5 °C | **3,6 °C** | 4,9 °C |
    | Varighet under −1 °C | 84 t | 120 t | **126 t** | 112 t |

    #### Usikkerheten skalerer med hvor mye som er i bevegelse

    Residualen er sterkt heteroskedastisk. Målt over de samme 10 552 timene:

    | \\|ΔT_Vorma\\| under transport | 0–0,5 | 0,5–1 | 1–2 | 2–3 | > 3 |
    |---|---|---|---|---|---|
    | Standardavvik (°C) | 0,47 | 0,57 | 0,72 | 1,02 | **1,53** |

    En fast σ = 0,6 °C er derfor omtrent riktig i rolige perioder, men altfor smal
    nettopp når en stor kaldpuls er under transport. Modellen bruker i stedet
    σ = √(0,45² + (0,33 · |ΔT_Vorma|)²), som gir 71 % dekning i 68 %-båndet og
    93,5 % i 95 %-båndet. Utenfor datahorisonten vokser σ mot en asymptote på
    2,4 °C, kalibrert mot hvor stor feilen blir ved ren persistens (1,3 / 1,9 /
    2,3 / 2,4 °C ved +24/48/72/96 timer).
    """)

    st.divider()

    st.subheader("Ettereffekt – forsinket kaldpuls fra Mjøsa")
    st.markdown("""
    Etter at vind fra sør har presset det varme overflatelaget mot nordenden av Mjøsa
    og drevet kaldt bunnvann opp mot Minnesund, vil sjiktet mellom kaldt og varmt vann 
    fortsette å pendle opp og ned selv etter at vinden har lagt seg.
    Denne indre bølgen (seiché) er har en halvperiode på typisk 5–8 dager på en normal sommer.

    Praktisk konsekvens: en ny kaldpuls kan nå Glomma 5–12 dager etter den første,
    uten nytt vindpådriv. Modellen overvåker dette og viser en forhøyet risikoindikator
    i dette tidsvinduet.

    | Kriterium for ettereffekt | Verdi |
    |---|---|
    | Primær bunntemp ved Minnesund | < 10 °C |
    | Minimum temperaturdropp (ΔT) | ≥ 3 °C under en 7-dagers periode |
    | Forhøyet risikovindu | Dag 5–12 etter primær bunn |
    | Typisk halvperiode | 5-8 dager |

    **Validering 2015–2025 (682 juli–august-dager, Fetsund < 18 °C = «kaldt»):**

    | Modell | Sensitivitet | F1-score | FN-dager |
    |---|---|---|---|
    | Kun vindbasert | 0,70 | 0,756 | 167 |
    | Vind + ettereffekt | **0,92** | **0,876** | **46** |

    Ettereffekt triggeren legger til 121 korrekte alarmflagg og bare 15 falske alarmer.
    """)

    st.divider()

    st.subheader("Våtdrakt og sikkerhet – Glommadyppen")
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
        "Predikert vanntemperatur langs den lengste svømmestrekningen i Glommadyppen basert på observasjoner i Mjøsa, "
        "Vorma og Glomma. Primært prediksjonspunkt er startpunktet til **Fløter'n** (Glommadyppen), "
        "35,5 km fra Svanefoss. Fetsund bru 10,5 km lengre nedstrøms Glomma er sekundært "
        "målepunkt. Det kalde vannet ankommer startpunktet til Fløter'n **4–5 timer tidligere** enn ved sluttpunktet"
        "ved typisk augustvannføring."
    )

    event_date = calculate_event_date(EVENT_YEAR)
    days_until = (event_date - pd.Timestamp.now(tz='UTC')).days
    oslo_dt    = event_date.tz_convert('Europe/Oslo')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Neste arrangement", oslo_dt.strftime("%d. %b %Y"))
    c2.metric("Dager igjen", str(max(0, days_until)))

    with st.spinner("Henter data…"):
        # ÉN henting av Vorma-temperatur med det lengste vinduet som trengs.
        # Seiche-deteksjonen krever 20 døgn (bunnpunkt inntil 12 d tilbake +
        # 7 d baseline før det); prognosen bruker de siste 7 døgnene av samme
        # serie. Tidligere ble stasjonen hentet to ganger (168 t og 336 t).
        vorma_history = fetch_nve_data(STATION_SVANEFOSS, 1003,
                                       hours_back=SEICHE_HISTORY_HOURS)
        if vorma_history.empty:
            vorma_history = fetch_nve_data(STATION_FUNNEFOSS_TEMP, 1003,
                                           hours_back=SEICHE_HISTORY_HOURS)
        primary_df = vorma_history.copy()
        if not primary_df.empty:
            _cut = pd.to_datetime(primary_df['time']).max() - pd.Timedelta(hours=168)
            primary_df = primary_df[pd.to_datetime(primary_df['time']) >= _cut] \
                             .reset_index(drop=True)

        fetsund_temp  = fetch_nve_data(STATION_FETSUND,      1003, hours_back=168)
        ertesekken_q  = fetch_nve_data(STATION_ERTESEKKEN_Q, 1001, hours_back=168)
        # Glomma-vannføring: nødvendig for dynamisk uttynning κ = Q_V/(Q_V+Q_G)
        funnefoss_q   = fetch_nve_data(STATION_FUNNEFOSS_Q,  1001, hours_back=168)
        frost_vind    = fetch_frost_wind(hours_back=168)
        weather_mjosa = fetch_weather_forecast(MJOSA_LAT, MJOSA_LON)
        if not weather_mjosa.empty:
            weather_mjosa = add_southerly_component(weather_mjosa)
        seiche = detect_seiche_risk(vorma_history)

    if primary_df.empty:
        st.error("Ingen Vorma-data tilgjengelig. Sjekk NVE HydAPI.")
        return

    _last_t = pd.to_datetime(primary_df['time'].max())
    if _last_t.tzinfo is None:
        _last_t = _last_t.tz_localize('UTC')
    data_age_hours = (pd.Timestamp.now(tz='UTC') - _last_t).total_seconds() / 3600

    c3.metric("Siste Vorma-data",
              _last_t.tz_convert('Europe/Oslo').strftime('%d.%m %H:%M'))
    c4.metric("Dataalder", f"{data_age_hours:.0f} t",
              delta="⚠️ Gamle data" if data_age_hours > 48 else None,
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

    # ── Uttynning ved samløpet – beregnet, ikke konstant ─────────────────────
    kappa_now, f_now, kappa_src = dilution_kappa(ertesekken_q, funnefoss_q,
                                                 mode='episode')
    k1, k2, k3 = st.columns(3)
    k1.metric(
        "Uttynning κ nå", f"{kappa_now:.2f}",
        delta=("beregnet fra vannføring" if "historisk" not in kappa_src
               else "historisk median"),
        delta_color="off",
        help=("κ er Vormas blandingsandel i samløpet: "
              "κ = Q_Vorma / (Q_Vorma + Q_Glomma). "
              f"Nå: {kappa_src}. Historisk median jul–aug er 0,633 – praktisk "
              "talt identisk med den empiriske episode-κ = 0,63 fra 35 kalde "
              "episoder, som bekrefter tolkningen."),
    )
    _qg_txt = (f"{funnefoss_q.iloc[-1]['value']:.0f} m³/s"
               if not funnefoss_q.empty else "N/A")
    k2.metric("Vannføring Vorma (Ertesekken)", f"{q_val:.0f} m³/s")
    k3.metric("Vannføring Glomma (Funnefoss)", _qg_txt,
              delta=None if not funnefoss_q.empty else "⚠️ mangler – bruker median κ",
              delta_color="off")
    if f_now < 0.50:
        st.warning(
            f"**Glomma dominerer samløpet akkurat nå** (Vorma-andel {f_now:.0%}, "
            "normalt 63 %). En kaldpuls fra Mjøsa blir kraftigere fortynnet enn "
            "vanlig før den når Fløter'n, og prediksjonen er tilsvarende dempet.",
            icon="💧",
        )

    # ── Seiche-ettereffekt banner ─────────────────────────────────────────────
    # Vorma anses "stigende" når temperaturen har økt > 0,2 °C de siste 24 t.
    # I så fall er en ny (tredje) kalddipp i samme periode lite sannsynlig, og
    # varselet nedgraderes til en info om at oppgangen forventes å forplante
    # seg nedstrøms til Fløter'n/Fetsund.
    vorma_rising = (len(primary_df) >= 24 and
                     (latest_val - primary_df.iloc[-24]['value']) > 0.2)

    if seiche['active'] and vorma_rising:
        ep_date_oslo = seiche['episode_date'].tz_convert(
            'Europe/Oslo').strftime('%-d. %b kl %H:%M')
        st.info(
            f"**Vorma-temperaturen stiger igjen** – ingen ny kaldpuls forventet nå\n\n"
            f"Det er fortsatt ca. **{seiche['days_remaining']:.0f} dager** igjen av det "
            f"forhøyede risikovinduet etter kaldepisoden {ep_date_oslo}, men temperaturen "
            f"i Vorma har begynt å stige. En tredje kalddipp i denne perioden regnes som "
            f"lite sannsynlig. Oppgangen forventes å gi en tilsvarende temperaturøkning "
            f"ved Fløter'n om ca. **{t_flotern:.0f} t** og ved Fetsund om ca. "
            f"**{t_fetsund:.0f} t**.",
            icon="📈",
        )
    elif seiche['active']:
        ep_date_oslo = seiche['episode_date'].tz_convert(
            'Europe/Oslo').strftime('%-d. %b kl %H:%M')
        days_rem = seiche['days_remaining']
        st.warning(
            f"**Seiche-ettereffekt aktiv** – forhøyet risiko for sekundær kaldpuls\n\n"
            f"En bekreftet kald episode ble registrert ved Minnesund for "
            f"**{seiche['days_ago']:.1f} dager siden** "
            f"({ep_date_oslo}, min {seiche['episode_min_T']:.1f} °C, "
            f"ΔT = {seiche['episode_dT']:.1f} °C). "
            f"Sprangsjiktet i Mjøsa kan oscillere tilbake og gi en ny kaldpuls – "
            f"typisk opptrer sekundærdroppen 5–12 dager etter primær bunn. "
            f"**Forhøyet risikovindu varer i ca. {days_rem:.0f} dager til.**\n\n",
            icon="🌊",
        )


    # ── Bygg prognose tidlig så den er tilgjengelig i hele seksjonen ─────────
    energy_df   = build_wind_energy_series(frost_vind, weather_mjosa)
    forecast_df = build_fetsund_forecast(primary_df, fetsund_temp, ertesekken_q,
                                         glomma_q_df=funnefoss_q,
                                         energy_df=energy_df)
    _mode = forecast_df['mode'].iloc[0] if not forecast_df.empty else None
    if _mode == 'level' and FORECAST_MODE == 'increment':
        st.info(
            "Prognosen kjører i **nivåmodus** (reservemodell) fordi Fetsund-"
            "målingene mangler eller er eldre enn 24 t. Nivåformen har høyere "
            "forventet feil (MAE ≈ 0,84 mot 0,54 °C) – tolk båndet med tilsvarende "
            "forsiktighet.",
            icon="ℹ️",
        )
    if energy_df is None or energy_df.empty:
        st.warning(
            "**Vinddata er utilgjengelig** (Frost API / Met.no svarte ikke). "
            "Prognosen under er IKKE vindjustert, og oppvellingsrisiko kan ikke "
            "vurderes. Kun transportmodellen vises.",
            icon="🌬️",
        )
    t_flotern_h, travel_h_now, _, _ = calculate_travel_time(ertesekken_q)

    # ── Prediksjon for 1-4 dager ───────────────────────────────────────────────
    st.header("Prediksjon for 1-4 dager")
    st.subheader("Dagsprognose")
    st.caption(
        f"Frem til datahorisonten (+{travel_h_now:.0f} t) er prediksjonen basert på "
        "**observert vann i Vorma**, og predikerer endringen fra dagens Fetsund-måling: "
        "T = uforstyrret nivå + κ · Vorma-anomali, der κ beregnes fra sist målte "
        f"vannføring (nå {kappa_now:.2f}). "
        "Etter datahorisonten er det ekstrapolering – kun vindenergi-signalet (E) gir "
        "reell fremoverskuende informasjon (AUC = 0,87 for ΔT < −3 °C)."
    )

    _now_oslo = pd.Timestamp.now(tz='UTC').tz_convert('Europe/Oslo')

    def _dato_label(h):
        target = _now_oslo + pd.Timedelta(hours=h)
        return target.strftime('%a %d.%m')

    HORIZONS = [
        (f"Nå–+{travel_h_now:.0f} t\n(databasert)", travel_h_now),
        (_dato_label(48), 48),
        (_dato_label(72), 72),
        (_dato_label(96), 96),
    ]

    _RISK_EMOJI = {"lav": "🟢", "advarsel": "🟡", "alarm": "🔴"}

    if data_age_hours > 720 and days_until > 30:
        st.info(
            "Sanntidsprediksjon krever ferske Vorma-målinger. "
            "Aktiveres når stasjonen starter opp igjen (april 2026)."
        )
    elif not forecast_df.empty:
        fcols = st.columns(4)
        for i, (label, h) in enumerate(HORIZONS):
            # Finn raden nærmest h timer frem i tid
            now_utc = pd.Timestamp.now(tz='UTC')
            target_t = now_utc + pd.Timedelta(hours=h)
            fc_t = pd.to_datetime(forecast_df['time'])
            if fc_t.dt.tz is None:
                fc_t = fc_t.dt.tz_localize('UTC')
            idx = (fc_t - target_t).abs().idxmin()
            row = forecast_df.loc[idx]

            pred     = row['predicted']
            lo68     = row['lower_68']
            hi68     = row['upper_68']
            risk_raw = row.get('wind_risk_level')
            # NB: pd.notna() er kritisk her - "risk_raw or default" ville feilet
            # fordi NaN er "truthy" i Python, og resultatet ble literally "nan".
            risk     = risk_raw if pd.notna(risk_raw) else ('–' if h > travel_h_now else 'databasert')
            e_fc     = row.get('wind_E_forecast')

            if h <= travel_h_now:
                # Innenfor datahorisonten: vis lav, pålitelig usikkerhet
                delta_str  = f"{lo68:.1f}–{hi68:.1f} °C  ✅ databasert"
                delta_col  = "off"
                _sig = row.get('sigma')
                _dv  = row.get('delta_vorma')
                help_str   = (
                    "Basert på observert vann i Vorma (anomaliform)."
                    + (f" Vorma-anomali under transport = {_dv:+.1f} °C." if pd.notna(_dv) else "")
                    + (f" σ = {_sig:.2f} °C." if pd.notna(_sig) else "")
                )
            else:
                emoji      = _RISK_EMOJI.get(risk, "⚪")
                # Samme NaN-fiks som over: pd.notna() istedenfor "e_fc and ..."
                e_str      = f"  E={e_fc}" if pd.notna(e_fc) and e_fc != "ingen prognose" else ""
                delta_str  = f"{lo68:.1f}–{hi68:.1f} °C  {emoji} {risk}{e_str}"
                delta_col  = ("inverse"
                              if risk in ("advarsel", "alarm") else "off")
                _sig = row.get('sigma')
                help_str   = (
                    "Ekstrapolering: Vorma-anomalien relakserer mot et permanent restnivå "
                    f"({100*RELAX_PERSISTENT:.0f} % av dybden står igjen). Vindrisiko-nivå fra prognosert "
                    "SE/S-vindenergi (Met.no, AUC = 0,87 for ΔT < −3 °C)."
                    + (f" σ = {_sig:.2f} °C." if pd.notna(_sig) else "")
                )

            fcols[i].metric(
                label, f"{pred:.1f} °C",
                delta=delta_str,
                delta_color=delta_col,
                help=help_str,
            )
    else:
        st.warning("Ikke nok data for dagsprognose.")

    st.divider()

    # ── Temperaturprognose (graf) ─────────────────────────────────────────────
    st.subheader("Temperaturprognose – Fløter'n / Fetsund")

    if not forecast_df.empty:
        # Tidligere prediksjoner fra loggen, tegnet stiplet ved siden av fasit
        pred_log = read_prediction_log()
        hist_h   = 24
        hist_df  = pd.DataFrame()
        if not pred_log.empty:
            hcol1, _ = st.columns([1, 3])
            hist_h = hcol1.selectbox(
                "Vis tidligere prediksjon med varsel",
                options=list(_core.EVAL_HORIZONS), index=0,
                format_func=lambda h: f"{h} timer i forveien",
                help=("Stiplet oransje linje viser hva modellen predikerte for hvert "
                      "tidspunkt, gitt så mange timer i forveien. Avstanden til den "
                      "heltrukne blå observasjonslinjen er treffsikkerheten."),
            )
            hist_df = _core.prediction_history_series(pred_log, hist_h)

        fig_fc = _forecast_chart(fetsund_temp, forecast_df, travel_h_now,
                                 history_df=hist_df, history_horizon=hist_h)
        st.plotly_chart(fig_fc, use_container_width=True, config={"responsive": True})
        if pred_log.empty:
            st.caption(
                "Prediksjonsloggen er tom eller utilgjengelig, så tidligere "
                "prediksjoner kan ikke vises. Loggen fylles av GitHub Actions-"
                "jobben kl. 06:00 UTC."
            )
        _sig_max = forecast_df['sigma'].max() if 'sigma' in forecast_df.columns else None
        st.caption(
            "Prikket vannrett linje: **uforstyrret nivå** (7-døgns 90-persentil) – "
            "temperaturen elva ville hatt uten kaldepisoden. Dippen måles mot den. "
            "Solid linje: observert (Fetsund) · stiplet linje: prediksjon · "
            "mørkt bånd: 68 % sannsynlig område · lyst bånd: 95 %. "
            "Båndene er **usymmetriske med hensikt**. Når en kaldpuls er observert i "
            "Vorma, er usikkerheten hvor KRAFTIG den slår ut – ikke om den kommer. "
            "Empirisk er responsen negativ i 0,1 % av tilfellene ved store anomalier, "
            "så båndet bygges av forsterkningskvantiler (κ × 0,4 til κ × 1,6) i stedet "
            "for et symmetrisk ± σ. Utenfor datahorisonten er tillegget ensidig: en "
            "ukjent framtidig oppvellingshendelse kan bare gjøre det kaldere. "
            f"**Frem til datahorisonten (+{travel_h_now:.0f} t)** er prediksjonen basert på "
            "observert vann i Vorma. Bredden følger hvor stor temperaturendring som er "
            f"under transport (σ = √({SIGMA_BASE}² + ({SIGMA_PER_DELTA}·|ΔT_Vorma|)²)) – "
            "den er smal i rolige perioder og utvides automatisk når en kraftig kaldpuls "
            "er på vei. **Etter datahorisonten** ekstrapoleres anomalien mot 72-timers "
            f"medianen, og usikkerheten metter mot σ ≈ {MODEL_SIGMA_ASYMPTOTE} °C "
            f"(kalibrert mot persistensfeilen ved Fetsund)"
            + (f"; maks i denne prognosen er σ = {_sig_max:.1f} °C. " if _sig_max else ". ")
            + f"Innenfor vindrisiko-horisonten (+{WIND_RISK_HORIZON_HOURS} t) skjeves båndet "
            "nedover dersom SE/S-vindvarselet overskrider advarsel- "
            f"({ENERGY_WARN:.0f} m·h) eller alarmterskelen ({ENERGY_THRESHOLD:.0f} m·h)."
        )
    else:
        st.warning("Ikke nok data for prognosevisning.")

    # ── Vind og oppvellingsrisiko ─────────────────────────────────────────────
    if not weather_mjosa.empty or not frost_vind.empty:
        st.divider()
        st.subheader("Vind og oppvellingsrisiko – Mjøsa")

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
                    st.plotly_chart(fig_e, use_container_width=True, config={"responsive": True})
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
                st.plotly_chart(chart, use_container_width=True, config={"responsive": True})


# ============================================================================
# PAGE: DATA & VARSEL
# ============================================================================

def page_data_varsel():
    st.title("Observasjoner og Værvarsler")
    st.markdown(
        "Faktiske målinger fra NVE og met.no. "
        "Bruk denne siden for å se rådata og standard værvarsler."
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
        st.plotly_chart(fig, use_container_width=True, config={"responsive": True})

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
        st.plotly_chart(fig, use_container_width=True, config={"responsive": True})

        _f, _fsrc = mixing_fraction(er_q, fn_q)
        st.caption(
            f"**Blandingsandel i samløpet: f = {_f:.3f}** ({_fsrc}). "
            "Dette tallet ER uttynningskoeffisienten κ i prediksjonsmodellen – "
            "andelen av vannet ved Fløter'n/Fetsund som kommer fra Vorma, og dermed "
            "hvor mye av en kaldpuls fra Mjøsa som overlever samløpet. "
            "Historisk median juli–august er 0,633. Massebalansen stemmer: "
            "median (Q_Ertesekken + Q_Funnefoss) / Q_Blaker = 1,03 over 2015–2025."
        )

        st.subheader("Transporttid-kalkulator")
        q_now, _ = safe_discharge(er_q, FALLBACK_DISCHARGE)
        q_now = min(max(q_now, 100), 1200)
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
                st.plotly_chart(chart, use_container_width=True, config={"responsive": True})

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
                    st.plotly_chart(chart, use_container_width=True, config={"responsive": True})

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
                    st.plotly_chart(chart, use_container_width=True, config={"responsive": True})

        st.info("""
        **Om oppvellings-indikatoren (SE/S-vind):**
        🟢 Lav risiko (< 1,2 m/s) · 🟡 Moderat (1,2–1,9 m/s) · 🔴 Høy (≥ 1,9 m/s vedvarende SE/S-vind)
        Vind over tid fra sørøst–sør (135–225°) kan føre til kaldt vann fra Mjøsa til Glomma.
        """)


# ============================================================================
# PAGE: TREFFSIKKERHET
# Etterprøver den loggede prediksjonen mot faktisk observert temperatur ved
# Fetsund. Dette er grunnlaget for å kalibrere SIGMA_BASE / SIGMA_PER_DELTA mot
# ekte residualer i stedet for mot historiske rekonstruksjoner.
# ============================================================================

def page_treffsikkerhet():
    st.title("Treffsikkerhet")
    st.markdown(
        "Hver morgen logges prediksjonen for 24, 48, 72 og 96 timer frem. Her "
        "sammenlignes hver av dem med temperaturen som faktisk ble målt ved "
        "Fetsund på gyldighetstidspunktet."
    )

    with st.spinner("Henter logg og observasjoner…"):
        log = read_prediction_log()
        # Hent Fetsund-observasjoner så langt tilbake som loggen rekker
        hours_back = 720
        if not log.empty and 'logged_at' in log.columns:
            _t0 = pd.to_datetime(log['logged_at'], errors='coerce', utc=True).min()
            if pd.notna(_t0):
                span = (pd.Timestamp.now(tz='UTC') - _t0).total_seconds() / 3600
                hours_back = int(min(max(span + 168, 336), 2160))  # 14 d–90 d
        fetsund_obs = fetch_nve_data(STATION_FETSUND, 1003, hours_back=hours_back)

    if log.empty:
        st.info(
            "**Loggen er tom eller utilgjengelig ennå.**\n\n"
            "Radene skrives av GitHub Actions-jobben `log_prediction.py` "
            "kl. 06:00 UTC hver dag. Siden fylles automatisk etter hvert som "
            "prediksjoner rekker å bli innhentet av virkeligheten – de første "
            "tallene for 24-timers horisonten kommer etter to døgn, og for "
            "96-timers horisonten etter fem.",
            icon="🗓️",
        )
        return

    if fetsund_obs.empty:
        st.error("Ingen Fetsund-observasjoner tilgjengelig – kan ikke etterprøve.")
        return

    ev = _core.evaluate_prediction_log(log, fetsund_obs)
    if ev.empty:
        _n = len(log)
        st.info(
            f"Loggen har {_n} rad(er), men ingen av prediksjonene har rukket å "
            "bli innhentet av en tilsvarende observasjon ennå. Prøv igjen om "
            "noen døgn.",
            icon="⏳",
        )
        return

    summary = _core.summarize_prediction_skill(ev)

    # ── Nøkkeltall ───────────────────────────────────────────────────────────
    st.header("Nøkkeltall")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Etterprøvde prediksjoner", f"{len(ev)}",
              help=f"Fra {ev['logged_at'].min():%d.%m.%Y} til {ev['logged_at'].max():%d.%m.%Y}")
    c2.metric("Samlet MAE", f"{ev['abs_error'].mean():.2f} °C",
              help="Gjennomsnittlig absoluttavvik over alle horisonter")
    _bias = ev['error'].mean()
    c3.metric("Systematisk skjevhet", f"{_bias:+.2f} °C",
              delta=("modellen predikerer for varmt" if _bias > 0.3 else
                     "modellen predikerer for kaldt" if _bias < -0.3 else
                     "ingen vesentlig skjevhet"),
              delta_color="off",
              help="Positiv verdi = prediksjonen ligger over observasjonen")
    _c68 = ev['in68'].dropna()
    c4.metric("Dekning i 68 %-båndet",
              f"{100 * _c68.mean():.0f} %" if len(_c68) else "N/A",
              delta="mål: 68 %", delta_color="off")

    # ── Per horisont ─────────────────────────────────────────────────────────
    st.divider()
    st.header("Per prognosehorisont")

    disp = summary.copy()
    disp['Horisont']     = disp['horizon_h'].map(lambda h: f"+{h} t")
    disp['n']            = disp['n']
    disp['MAE']          = disp['mae'].map(lambda v: f"{v:.2f} °C")
    disp['Skjevhet']     = disp['bias'].map(lambda v: f"{v:+.2f} °C")
    disp['RMSE']         = disp['rmse'].map(lambda v: f"{v:.2f} °C")
    disp['P90 avvik']    = disp['p90_abs'].map(lambda v: f"{v:.2f} °C")
    disp['Dekning 68 %'] = disp['coverage68'].map(
        lambda v: f"{100 * v:.0f} %" if pd.notna(v) else "–")
    disp['Dekning 95 %'] = disp['coverage95'].map(
        lambda v: f"{100 * v:.0f} %" if pd.notna(v) else "–")
    disp['σ oppgitt']    = disp['sigma_mean'].map(
        lambda v: f"{v:.2f} °C" if pd.notna(v) else "–")
    disp['σ faktisk']    = disp['sigma_implied'].map(
        lambda v: f"{v:.2f} °C" if pd.notna(v) else "–")
    st.dataframe(
        disp[['Horisont', 'n', 'MAE', 'Skjevhet', 'RMSE', 'P90 avvik',
              'Dekning 68 %', 'Dekning 95 %', 'σ oppgitt', 'σ faktisk']],
        hide_index=True, use_container_width=True,
    )

    st.caption(
        "**σ oppgitt** er usikkerheten modellen selv anga; **σ faktisk** er "
        "standardavviket til de virkelige residualene. Er σ faktisk vesentlig "
        "større enn σ oppgitt, er båndene for smale, og `SIGMA_BASE` / "
        "`SIGMA_PER_DELTA` i glommadyppen_core.py bør økes tilsvarende – og "
        "omvendt. Dekningstallene bør nærme seg 68 % og 95 % når antallet "
        "observasjoner blir stort nok; med under ~30 punkter per horisont er "
        "de fortsatt svært usikre."
    )

    # ── MAE-graf ─────────────────────────────────────────────────────────────
    if len(summary) > 0:
        fig_mae = go.Figure()
        fig_mae.add_trace(go.Bar(
            x=[f"+{h} t" for h in summary['horizon_h']], y=summary['mae'],
            name='MAE', marker_color='#185FA5',
            hovertemplate='<b>MAE</b>: %{y:.2f} °C<extra></extra>',
        ))
        fig_mae.add_trace(go.Bar(
            x=[f"+{h} t" for h in summary['horizon_h']], y=summary['sigma_implied'],
            name='σ faktisk', marker_color='rgba(186,117,23,0.75)',
            hovertemplate='<b>σ faktisk</b>: %{y:.2f} °C<extra></extra>',
        ))
        fig_mae.add_trace(go.Scatter(
            x=[f"+{h} t" for h in summary['horizon_h']], y=summary['sigma_mean'],
            mode='lines+markers', name='σ oppgitt av modellen',
            line=dict(color='#6B0000', width=2, dash='dot'),
            hovertemplate='<b>σ oppgitt</b>: %{y:.2f} °C<extra></extra>',
        ))
        fig_mae.update_layout(
            title="Feil og usikkerhet per horisont", barmode='group',
            yaxis=dict(title='°C'), height=380, template='plotly_white',
            margin=dict(l=50, r=30, t=50, b=40), hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        xanchor='center', x=0.5, font=dict(size=10)),
        )
        st.plotly_chart(fig_mae, use_container_width=True,
                        config={"responsive": True})

    # ── Predikert mot observert ──────────────────────────────────────────────
    st.divider()
    st.header("Predikert mot observert")

    sel_h = st.selectbox(
        "Horisont", options=sorted(ev['horizon_h'].unique()),
        format_func=lambda h: f"{h} timer i forveien",
    )
    sub = ev[ev['horizon_h'] == sel_h].sort_values('valid_time')

    if sub.empty:
        st.info("Ingen etterprøvde prediksjoner på denne horisonten ennå.")
        return

    fig_ts = go.Figure()
    if sub['lower68'].notna().any() and sub['upper68'].notna().any():
        band = sub.dropna(subset=['lower68', 'upper68'])
        fig_ts.add_trace(go.Scatter(
            x=list(band['valid_time']) + list(band['valid_time'])[::-1],
            y=list(band['upper68']) + list(band['lower68'])[::-1],
            fill='toself', fillcolor='rgba(186,117,23,0.16)',
            line=dict(color='rgba(0,0,0,0)'), name='68 % område',
            hoverinfo='skip',
        ))
    fig_ts.add_trace(go.Scatter(
        x=sub['valid_time'], y=sub['predicted'], mode='lines+markers',
        name=f'Predikert {sel_h} t i forveien',
        line=dict(color='#BA7517', width=1.8, dash='dash'),
        marker=dict(size=5),
        hovertemplate='<b>Predikert</b>: %{y:.1f} °C<extra></extra>',
    ))
    fig_ts.add_trace(go.Scatter(
        x=sub['valid_time'], y=sub['observed'], mode='lines',
        name='Observert (Fetsund)', line=dict(color='#185FA5', width=2.2),
        hovertemplate='<b>Observert</b>: %{y:.1f} °C<extra></extra>',
    ))
    fig_ts.update_layout(
        title=f"Prediksjon {sel_h} t i forveien mot fasit",
        yaxis=dict(title='°C'), xaxis_title='', height=400,
        template='plotly_white', hovermode='x unified',
        margin=dict(l=50, r=30, t=50, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='center', x=0.5, font=dict(size=10)),
    )
    st.plotly_chart(fig_ts, use_container_width=True, config={"responsive": True})

    # ── Feilfordeling ────────────────────────────────────────────────────────
    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(
        x=sub['valid_time'], y=sub['error'], mode='markers',
        marker=dict(size=7, color=sub['error'], colorscale='RdBu',
                    cmid=0, cmin=-3, cmax=3, showscale=False),
        name='Avvik', hovertemplate='<b>Avvik</b>: %{y:+.2f} °C<extra></extra>',
    ))
    fig_err.add_hline(y=0, line_color='rgba(80,80,80,0.6)', line_width=1)
    _mb = sub['error'].mean()
    fig_err.add_hline(y=_mb, line_dash='dot', line_color='#BA7517',
                      annotation_text=f"snitt {_mb:+.2f} °C",
                      annotation_position='top left', annotation_font_size=10)
    fig_err.update_layout(
        title=f"Avvik over tid (+{sel_h} t) – positiv = predikert for varmt",
        yaxis=dict(title='°C'), xaxis_title='', height=300,
        template='plotly_white', margin=dict(l=50, r=30, t=50, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig_err, use_container_width=True, config={"responsive": True})

    # ── Treff fordelt på vindrisikonivå ──────────────────────────────────────
    if 'windrisk' in sub.columns and sub['windrisk'].notna().any():
        grp = (sub.dropna(subset=['windrisk'])
                  .groupby('windrisk')['abs_error']
                  .agg(['count', 'mean']).reset_index())
        grp = grp[grp['count'] >= 3]
        if not grp.empty:
            st.subheader("Treffsikkerhet fordelt på vindrisikonivå")
            grp['Nivå'] = grp['windrisk']
            grp['Antall'] = grp['count']
            grp['MAE'] = grp['mean'].map(lambda v: f"{v:.2f} °C")
            st.dataframe(grp[['Nivå', 'Antall', 'MAE']], hide_index=True,
                         use_container_width=True)
            st.caption(
                "Er MAE vesentlig høyere på «advarsel» og «alarm» enn på «lav», "
                "bekrefter det at vindsituasjonene er de vanskelige – og at "
                "sigma-multiplikatorene WIND_SIGMA_MULT_WARN / _ALARM gjør en "
                "reell jobb."
            )

    with st.expander("Alle etterprøvde prediksjoner (rådata)"):
        raw = ev.copy().sort_values(['valid_time', 'horizon_h'], ascending=[False, True])
        raw['logged_at']  = raw['logged_at'].dt.tz_convert('Europe/Oslo').dt.strftime('%d.%m %H:%M')
        raw['valid_time'] = raw['valid_time'].dt.tz_convert('Europe/Oslo').dt.strftime('%d.%m %H:%M')
        st.dataframe(
            raw[['logged_at', 'horizon_h', 'valid_time', 'predicted',
                 'observed', 'error', 'lower68', 'upper68', 'in68', 'windrisk']]
            .round(2),
            hide_index=True, use_container_width=True,
        )
        st.download_button(
            "Last ned som CSV", ev.to_csv(index=False).encode('utf-8'),
            file_name="glommadyppen_treffsikkerhet.csv", mime="text/csv",
        )


# ============================================================================
# MAIN – navigasjon
# ============================================================================

def main():
    _check_core_version()
    _inject_mobile_css()
    # Navigasjonshint: vises kun på smale skjermer (CSS display:none på desktop)
    st.markdown(
        '''<div class="gd-mobile-hint">
        ☰&nbsp; Trykk på menyen øverst til venstre for å navigere mellom sidene
        </div>''',
        unsafe_allow_html=True,
    )
    with st.sidebar:
        # Logoen leses fra disk. Mangler filen (eller er repoet sjekket ut uten
        # den), skal det IKKE ta ned hele appen - da vises bare en tekstlenke.
        try:
            _logo_b64 = __import__('base64').b64encode(
                open('Samensatt_logo_GlommDyppen.jpg', 'rb').read()).decode()
            st.markdown(
                '<a href="https://glommadyppen.no" target="_blank">'
                + f'<img src="data:image/jpeg;base64,{_logo_b64}" '
                  'style="width:100%;cursor:pointer;">'
                + '</a>',
                unsafe_allow_html=True
            )
        except OSError:
            st.markdown("### [GlommaDyppen](https://glommadyppen.no)")
        st.markdown("---")
        page = st.radio(
            "Navigasjon",
            options=["Om siden", "Observasjoner og værvarsel", "Dipp Prediksjon",
                     "Treffsikkerhet"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption(f"App 1.8.0 · kjerne {getattr(_core, 'CORE_VERSION', '?')}")
        st.markdown("""
        **Modell**
        - Fløter'n (start): t = 7670 / Q
        - Fetsund (mål): t = 9700 / Q
        - Anomaliform (v1.8)
        - κ = Q_Vorma/(Q_Vorma+Q_Glomma), beregnet
        - Validert 2018–2025 (AUC = 0,87)
        - Ettereffekt: dag 5–12 etter vindepisode

        **Glommadyppen – våtdrakt**
        - Våtdrakt er obligatorisk uansett temperatur
        - For unntak: Søk arrangøren

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
    elif page == "Treffsikkerhet":
        page_treffsikkerhet()
    else:
        page_prediksjon()


if __name__ == "__main__":
    main()
