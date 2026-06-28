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
    initial_sidebar_state="expanded"
)

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
                name='95 % KI', hoverinfo='skip',
            ))
        if t_fwd:
            fig.add_trace(go.Scatter(
                x=t_fwd + t_rev,
                y=list(band_df['upper_68']) + list(band_df['lower_68'])[::-1],
                fill='toself', fillcolor='rgba(56,141,228,0.22)',
                line=dict(color='rgba(0,0,0,0)', width=0),
                name='68 % KI', hoverinfo='skip',
            ))
        hover_cols = ['lower_68', 'upper_68', 'lower_95', 'upper_95']
        hover_template = (
            '<b>Prediksjon</b>: %{y:.1f} °C<br>'
            '68 % KI: %{customdata[0]:.1f}–%{customdata[1]:.1f} °C<br>'
            '95 % KI: %{customdata[2]:.1f}–%{customdata[3]:.1f} °C'
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
        - Usikkerheten er ±2–3 °C (95 % KI)
        """)

    st.divider()

    st.subheader("Ettereffekt – forsinket kaldpuls fra Mjøsa")
    st.markdown("""
    Etter at sørlig vind har presset det varme overflatelaget mot sørenden av Mjøsa
    og drevet kaldt bunnvann (hypolimnion) opp mot Minnesund, vil **sprangsjiktet
    fortsette å oscillere** som en pendel selv etter at vinden har lagt seg.
    Denne indre bølgen (seiché) er beskrevet av Thendrup (1978) med en halvperiode på
    typisk **5–8 dager** ved normal sommerstratifisering.

    Praktisk konsekvens: en ny kaldpuls kan nå Glomma **5–12 dager etter den første**,
    uten nytt vindpådriv. Modellen overvåker dette og viser en forhøyet risikoindikator
    i dette tidsvinduet.

    | Kriterium for ettereffekt | Verdi |
    |---|---|
    | Primær bunn ved Minnesund | < 10 °C |
    | Minimum temperaturdropp (ΔT) | ≥ 3 °C under 7-dagers baseline |
    | Forhøyet risikovindu | Dag 5–12 etter primær bunn |
    | Typisk halvperiode | 8–9 dager |

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
                                         energy_df=energy_df)
    t_flotern_h, travel_h_now, _, _ = calculate_travel_time(ertesekken_q)

    # ── Prediksjon for arrangementet ──────────────────────────────────────────
    st.header("Prediksjon for arrangementet")

    # Gyldighetsvindu: vannet som er observert i Vorma nå ankommer Fløter'n om
    # travel_h_now timer. Innenfor det vinduet er prediksjonen databasert og
    # pålitelig (σ ≈ MODEL_SIGMA_DATA). Utenfor er det ren ekstrapolering.
    valid_from_oslo = pd.Timestamp.now(tz='UTC').tz_convert('Europe/Oslo')
    valid_to_oslo   = (pd.Timestamp.now(tz='UTC') +
                       pd.Timedelta(hours=travel_h_now)).tz_convert('Europe/Oslo')

    if days_until > 14:
        st.info(
            f"ℹ️ Prediksjonen nedenfor viser **nåværende forhold**, ikke en prognose for august. "
            f"Det er {days_until} dager til GlommaDyppen ({oslo_dt.strftime('%-d. %B %Y')}). "
            f"Dagens temperaturmåling i Vorma ved Svanefoss er **direkte gyldig** for "
            f"Fløter'n frem til ca. **{valid_to_oslo.strftime('%-d. %b kl %H:%M')}** "
            f"(transporttid {travel_h_now:.0f} t). Etter det er prognosen ekstrapolering "
            f"med økende usikkerhet – se dagsprognose-rad og graf under."
        )
    else:
        st.success(
            f"✅ Prediksjon for arrangementet er aktiv ({days_until} dager igjen). "
            f"Vorma-observasjoner er direkte gyldige for **Fløter'n** de neste "
            f"{travel_h_now:.0f} t (frem til {valid_to_oslo.strftime('%-d. %b kl %H:%M')})."
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
        pred_temp = prediction['predicted_temp']

        # To-regime sigma: innenfor datahorisonten er usikkerheten liten og
        # flat (MODEL_SIGMA_DATA); utenfor vokser den med √(ekstrapoleringstid).
        h_until    = max(0.0, (event_date - pd.Timestamp.now(tz='UTC')
                               ).total_seconds() / 3600)
        ramp_ev    = min(1.0, h_until / max(travel_h_now, 1))
        extrap_ev  = max(0.0, h_until - travel_h_now)
        sigma      = (MODEL_SIGMA_DATA * ramp_ev
                      + (MODEL_SIGMA - MODEL_SIGMA_DATA) * np.sqrt(extrap_ev / 24.0))
        lb         = max(pred_temp - 1.96 * sigma, TEMP_HIST_LOWER)
        ub         = min(pred_temp + 1.96 * sigma, TEMP_HIST_UPPER)

        risk_label, risk_color, ws_label, ws_color, risk_details = \
            assess_risk_open_water(pred_temp, weather_mjosa, seiche_risk=seiche)

        # Gyldighets-tag i kortet
        if h_until <= travel_h_now:
            validity_tag = (f"Databasert · gyldig for Fløter'n frem til "
                            f"{valid_to_oslo.strftime('%-d. %b kl %H:%M')} · σ ≈ {sigma:.1f} °C")
        else:
            validity_tag = (f"Ekstrapolering · Vorma-data gyldig til "
                            f"{valid_to_oslo.strftime('%-d. %b kl %H:%M')} · σ ≈ {sigma:.1f} °C")

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
                    </div>
                    <div style="font-size:0.78em; color:#999; margin-top:2px;">
                        {validity_tag}
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
            c1.metric("Vorma nå",              f"{prediction['vorma_temp']:.1f} °C",
                      delta=f"{prediction['anomaly']:+.1f} °C avvik")
            c2.metric("Fetsund baseline",       f"{prediction['fetsund_baseline']:.1f} °C",
                      help=prediction['baseline_source'])
            c3.metric("Transporttid Fløter'n",  f"{prediction['travel_hours_flotern']} t",
                      help=f"t = 7670 / {prediction['q_used']:.0f} m³/s")
            c4.metric("Pålitelighet",           f"{prediction['confidence']*100:.0f} %")
            st.caption(
                f"Modell: T_pred = Fetsund_baseline + Vorma_anomali × κ "
                f"(κ = {TEMPERATURE_SURVIVAL}). "
                f"Validert mot data fra juli og august 2018–2025. "
                "Bruk med forsiktighet utenfor sommermånedene."
            )
    else:
        st.warning("⚠️ Ikke nok data for prediksjon.")

    # ── Dagsprognose – eksplisitt +2/+3/+4 dager ─────────────────────────────
    st.subheader("Dagsprognose")
    st.caption(
        f"Frem til datahorisonten (+{travel_h_now:.0f} t) er prediksjonen basert på "
        f"**observert vann i Vorma** og er relativt pålitelig (σ ≈ {MODEL_SIGMA_DATA} °C). "
        f"Etter det er den ekstrapolering – kun vindenergi-signalet (E) gir "
        f"reell fremoverskuende informasjon (AUC = 0,87 for ΔT < −3 °C)."
    )

    HORIZONS = [
        (f"Nå–+{travel_h_now:.0f} t\n(databasert)", travel_h_now),
        ("+2 dager",  48),
        ("+3 dager",  72),
        ("+4 dager",  96),
    ]

    _RISK_EMOJI = {"lav": "🟢", "advarsel": "🟡", "alarm": "🔴"}

    if not forecast_df.empty:
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

            pred   = row['predicted']
            lo68   = row['lower_68']
            hi68   = row['upper_68']
            risk   = row.get('wind_risk_level') or ('–' if h > travel_h_now else 'databasert')
            e_fc   = row.get('wind_E_forecast')

            if h <= travel_h_now:
                # Innenfor datahorisonten: vis lav, pålitelig usikkerhet
                delta_str  = f"{lo68:.1f}–{hi68:.1f} °C  ✅ databasert"
                delta_col  = "off"
                help_str   = (f"Basert på observert vann i Vorma – "
                              f"σ ≈ {MODEL_SIGMA_DATA} °C (validert MAE ~0,5–0,6 °C)")
            else:
                emoji      = _RISK_EMOJI.get(risk, "⚪")
                e_str      = f"  E={e_fc}" if e_fc and e_fc != "ingen prognose" else ""
                delta_str  = f"{lo68:.1f}–{hi68:.1f} °C  {emoji} {risk}{e_str}"
                delta_col  = ("inverse"
                              if risk in ("advarsel", "alarm") else "off")
                help_str   = (f"Ekstrapolering fra nå-tilstand med eksponentielt avtagende "
                              f"anomali. Vindrisiko-nivå basert på prognosert SE/S-vindenergi "
                              f"fra Met.no (AUC = 0,87 for ΔT < −3 °C).")

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
        fig_fc = _forecast_chart(fetsund_temp, forecast_df, travel_h_now)
        st.plotly_chart(fig_fc, use_container_width=True)
        st.caption(
            "Solid linje: observert (Fetsund) · stiplet linje: prediksjon · "
            f"grått bånd: 68 % KI · lyst bånd: 95 % KI. "
            f"**Frem til datahorisonten (+{travel_h_now:.0f} t)** er prediksjonen basert på "
            f"observert vann i Vorma og har lav usikkerhet (σ ≈ {MODEL_SIGMA_DATA} °C). "
            f"**Etter datahorisonten** ekstrapoleres anomalien eksponentielt og usikkerheten "
            f"vokser mot σ ≈ {MODEL_SIGMA} °C og videre. "
            f"Innenfor vindrisiko-horisonten (+{WIND_RISK_HORIZON_HOURS} t) vil båndet "
            "skjeves nedover og utvides dersom SE/S-vindvarselet overskrider "
            f"advarsel- ({ENERGY_WARN:.0f} m·h) eller alarmterskelen ({ENERGY_THRESHOLD:.0f} m·h)."
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
    else:
        page_prediksjon()


if __name__ == "__main__":
    main()
