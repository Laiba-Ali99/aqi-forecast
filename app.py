"""
app.py  —  AQI Forecast Dashboard
A professional, feature-rich Streamlit dashboard with:
  • Real-time 72-hour PM2.5 forecast
  • Day-by-day breakdown
  • Hazardous AQI alerts
  • SHAP feature importance
  • EDA (historical trends, correlations, distributions)
  • Model performance metrics
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
from datetime import datetime

# ── Must be first Streamlit call ───────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Forecast · Karachi",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for a polished dark-accented design ────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
  }

  /* Page background */
  .main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
  }

  /* Hide default Streamlit header */
  header[data-testid="stHeader"] { display: none; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2330;
  }
  section[data-testid="stSidebar"] * { color: #e0e4f0 !important; }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stRadio label { color: #9aa0b8 !important; }

  /* KPI cards */
  .kpi-card {
    background: #1a1d2e;
    border: 1px solid #252840;
    border-radius: 16px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
  }
  .kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, #6366f1);
    border-radius: 16px 16px 0 0;
  }
  .kpi-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b7299;
    margin-bottom: 8px;
  }
  .kpi-value {
    font-size: 32px;
    font-weight: 700;
    color: #e8eaf6;
    line-height: 1;
    font-family: 'JetBrains Mono', monospace;
  }
  .kpi-sub {
    font-size: 12px;
    color: #6b7299;
    margin-top: 6px;
  }
  .kpi-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    margin-top: 8px;
  }

  /* Alert banner */
  .alert-banner {
    background: linear-gradient(135deg, #2d1b1b, #1f1010);
    border: 1px solid #ff4444;
    border-left: 4px solid #ff4444;
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 16px;
  }
  .alert-item {
    background: #1e1010;
    border: 1px solid #3d2020;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 13px;
    color: #ffcdd2;
  }

  /* Good air banner */
  .good-banner {
    background: linear-gradient(135deg, #0d2018, #091a10);
    border: 1px solid #00c853;
    border-left: 4px solid #00c853;
    border-radius: 12px;
    padding: 14px 20px;
    color: #b9f6ca;
    font-size: 14px;
  }

  /* Section headers */
  .section-header {
    font-size: 18px;
    font-weight: 600;
    color: #c5cae9;
    padding-bottom: 8px;
    border-bottom: 1px solid #252840;
    margin-bottom: 16px;
  }

  /* Metric comparison table */
  .metric-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #1e2330;
    font-size: 14px;
  }
  .metric-name { color: #9aa0b8; }
  .metric-val  { color: #e8eaf6; font-family: 'JetBrains Mono', monospace; }

  /* Day tab cards */
  .day-card {
    background: #1a1d2e;
    border: 1px solid #252840;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
  }
  .day-card-title { font-size: 12px; color: #6b7299; text-transform: uppercase; letter-spacing: 0.08em; }
  .day-card-val   { font-size: 24px; font-weight: 700; color: #e8eaf6; font-family: 'JetBrains Mono', monospace; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: #0f1117;
    padding: 4px;
    border-radius: 12px;
    border: 1px solid #1e2330;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #6b7299 !important;
    font-size: 13px;
    font-weight: 500;
    padding: 6px 18px;
  }
  .stTabs [aria-selected="true"] {
    background: #252840 !important;
    color: #c5cae9 !important;
  }

  /* Plotly chart backgrounds */
  .js-plotly-plot { border-radius: 12px; }

  /* Divider */
  hr { border-color: #1e2330 !important; }

  /* DataFrame */
  .stDataFrame { border-radius: 10px; overflow: hidden; }

  /* Sidebar nav items */
  .nav-item {
    padding: 8px 12px;
    border-radius: 8px;
    cursor: pointer;
    margin: 2px 0;
    font-size: 14px;
    color: #9aa0b8;
  }
  .nav-item:hover { background: #1e2330; color: #c5cae9; }
  .nav-item.active { background: #252840; color: #818cf8; }
</style>
""", unsafe_allow_html=True)


# ── Paths & constants ─────────────────────────────────────────────────────────
FEATURES_PATH = "data/processed/features.csv"
RAW_PATH      = "data/raw/aqi_data.csv"
MODEL_PATH    = "models/aqi_model.pkl"
METRICS_PATH  = "models/metrics.json"
SHAP_PATH     = "models/shap_summary.png"
FI_PATH       = "models/feature_importance.json"
CITY          = os.getenv("CITY", "Karachi")

AQI_COLORS = {
    "Good":                   "#00e676",
    "Moderate":               "#ffee58",
    "Unhealthy for Sensitive":"#ffa726",
    "Unhealthy":              "#ef5350",
    "Very Unhealthy":         "#ab47bc",
    "Hazardous":              "#b71c1c",
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#12141f",
    font=dict(family="Space Grotesk", color="#9aa0b8", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="#1e2330", zerolinecolor="#1e2330"),
    yaxis=dict(gridcolor="#1e2330", zerolinecolor="#1e2330"),
)


# ── Helper functions ──────────────────────────────────────────────────────────
def aqi_category(pm25: float) -> tuple[str, str]:
    breakpoints = [
        (0,   12.0,  "Good",                   "#00e676"),
        (12.1, 35.4, "Moderate",               "#ffee58"),
        (35.5, 55.4, "Unhealthy for Sensitive","#ffa726"),
        (55.5, 150.4,"Unhealthy",              "#ef5350"),
        (150.5,250.4,"Very Unhealthy",         "#ab47bc"),
        (250.5,9999, "Hazardous",              "#b71c1c"),
    ]
    for lo, hi, label, color in breakpoints:
        if lo <= pm25 <= hi:
            return label, color
    return "Hazardous", "#b71c1c"


@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv(FEATURES_PATH) if os.path.exists(FEATURES_PATH) else pd.DataFrame()
    if not df.empty and "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
    return df


@st.cache_data(ttl=300)
def load_raw():
    df = pd.read_csv(RAW_PATH) if os.path.exists(RAW_PATH) else pd.DataFrame()
    if not df.empty and "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


@st.cache_data
def load_feature_importance():
    if os.path.exists(FI_PATH):
        with open(FI_PATH) as f:
            return json.load(f)
    return None


def predict_72h(df: pd.DataFrame, model) -> list[dict]:
    """Generate 72-hour rolling forecast."""
    from datetime import timedelta
    DROP_COLS    = ["datetime", "target"]
    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    base_row       = df.iloc[-1].copy()
    current_pm25   = float(base_row.get("pm2_5", 15.0))
    rolling_window = df["pm2_5"].tail(24).tolist() if "pm2_5" in df.columns else [current_pm25]
    predictions    = []

    for h in range(1, 73):
        future_time = datetime.now() + timedelta(hours=h)
        row = base_row.copy()

        row["hour"]        = future_time.hour
        row["day"]         = future_time.day
        row["month"]       = future_time.month
        row["day_of_week"] = future_time.weekday()
        row["is_weekend"]  = int(future_time.weekday() >= 5)
        row["hour_sin"]    = np.sin(2 * np.pi * future_time.hour / 24)
        row["hour_cos"]    = np.cos(2 * np.pi * future_time.hour / 24)
        row["month_sin"]   = np.sin(2 * np.pi * future_time.month / 12)
        row["month_cos"]   = np.cos(2 * np.pi * future_time.month / 12)

        if rolling_window:
            row["pm25_lag_1"] = rolling_window[-1]
            row["pm25_lag_3"] = rolling_window[-3] if len(rolling_window) >= 3 else rolling_window[-1]
            row["pm25_rolling_mean_3"] = np.mean(rolling_window[-3:])
            row["pm25_rolling_mean_6"] = np.mean(rolling_window[-6:])
            row["pm25_rolling_std_3"]  = np.std(rolling_window[-3:]) if len(rolling_window) >= 2 else 0

        available = [c for c in feature_cols if c in row.index]
        X_row     = pd.DataFrame([row[available].fillna(0)])

        try:
            pm25_pred = max(0.0, float(model.predict(X_row)[0]))
        except Exception:
            pm25_pred = current_pm25

        label, color = aqi_category(pm25_pred)
        predictions.append({
            "hour":      h,
            "timestamp": future_time,
            "date":      future_time.strftime("%b %d"),
            "time":      future_time.strftime("%H:%M"),
            "pm2_5":     round(pm25_pred, 2),
            "label":     label,
            "color":     color,
            "day":       (h - 1) // 24 + 1,
        })
        rolling_window.append(pm25_pred)
        current_pm25 = pm25_pred

    return predictions


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px 0'>
      <div style='font-size:20px;font-weight:700;color:#818cf8;letter-spacing:-0.02em'>🌬️ AQI Forecast</div>
      <div style='font-size:12px;color:#6b7299;margin-top:4px'>Powered by Random Forest ML</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background:#1a1d2e;border:1px solid #252840;border-radius:10px;padding:12px 14px;margin:8px 0 16px 0'>
      <div style='font-size:11px;color:#6b7299;text-transform:uppercase;letter-spacing:0.08em'>City</div>
      <div style='font-size:18px;font-weight:600;color:#c5cae9;margin-top:4px'>{CITY}</div>
      <div style='font-size:11px;color:#6b7299;margin-top:2px'>{datetime.now().strftime("%d %b %Y · %H:%M")} UTC</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["📊 Live Forecast", "📈 EDA & History", "🤖 Model Performance", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("<div style='font-size:11px;color:#3d4466;padding:8px 0'>Data: OpenWeatherMap API<br>Model: Random Forest Regressor<br>Forecast horizon: 72 hours</div>", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
df    = load_data()
raw   = load_raw()
model = load_model()

if df.empty or model is None:
    st.error("⚠️ Data or model not found. Run these commands first:")
    st.code("python fetch_data.py\npython build_features.py\npython train_model.py")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: LIVE FORECAST
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Live Forecast":

    # Generate predictions
    with st.spinner("Generating 72-hour forecast..."):
        preds    = predict_72h(df, model)
        pred_df  = pd.DataFrame(preds)

    # Alerts
    alerts = [p for p in preds if p["pm2_5"] >= 35.5]

    # ── Dashboard header ───────────────────────────────────────────────────
    col_title, col_refresh = st.columns([5, 1])
    with col_title:
        st.markdown(f"""
        <div style='margin-bottom:4px'>
          <span style='font-size:28px;font-weight:700;color:#e8eaf6;letter-spacing:-0.03em'>
            Air Quality Forecast
          </span>
          <span style='font-size:16px;color:#6b7299;margin-left:12px'>{CITY} · Next 72 Hours</span>
        </div>
        """, unsafe_allow_html=True)
    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ── Alert or good-air banner ───────────────────────────────────────────
    if alerts:
        st.markdown(f"""
        <div class="alert-banner">
          <div style="font-size:14px;font-weight:600;color:#ff6b6b;margin-bottom:8px">
            ⚠️ {len(alerts)} hour(s) of elevated PM2.5 forecast in the next 72 hours
          </div>
          {''.join(f'<div class="alert-item">🔶 {p["label"]} — {p["pm2_5"]} µg/m³ at {p["time"]} on {p["date"]}</div>' for p in alerts[:5])}
          {'<div style="font-size:12px;color:#ff6b6b;margin-top:6px">+ more alerts...</div>' if len(alerts) > 5 else ''}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="good-banner">
          ✅ <strong>Air quality looks good</strong> for the next 72 hours. No hazardous periods forecast.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── KPI cards ──────────────────────────────────────────────────────────
    current_pm25 = float(df["pm2_5"].iloc[-1]) if "pm2_5" in df.columns else 0.0
    current_label, current_color = aqi_category(current_pm25)
    avg_72  = round(pred_df["pm2_5"].mean(), 1)
    peak_72 = round(pred_df["pm2_5"].max(),  1)
    min_72  = round(pred_df["pm2_5"].min(),  1)
    avg_lbl, avg_col  = aqi_category(avg_72)
    peak_lbl, pk_col  = aqi_category(peak_72)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "CURRENT PM2.5", f"{current_pm25:.1f}", "µg/m³", current_label, current_color, "#818cf8"),
        (c2, "72-H AVERAGE",  f"{avg_72}",           "µg/m³", avg_lbl,       avg_col,        "#34d399"),
        (c3, "72-H PEAK",     f"{peak_72}",           "µg/m³", peak_lbl,      pk_col,         "#f472b6"),
        (c4, "72-H MINIMUM",  f"{min_72}",            "µg/m³", "Best period", "#00e676",      "#60a5fa"),
    ]
    for col, label, val, unit, badge, badge_bg, accent in cards:
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="--accent:{accent}">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{val}</div>
              <div class="kpi-sub">{unit}</div>
              <div class="kpi-badge" style="background:{badge_bg}22;color:{badge_bg};border:1px solid {badge_bg}44">
                {badge}
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main 72-hour chart ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">72-Hour PM2.5 Forecast</div>', unsafe_allow_html=True)

    fig = go.Figure()

    # AQI bands (background)
    bands = [(0,12,"#00e676",0.04),(12,35.4,"#ffee58",0.04),(35.4,55.5,"#ffa726",0.06),(55.5,150,"#ef5350",0.08)]
    for lo, hi, col, alpha in bands:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=col, opacity=alpha, line_width=0, layer="below")

    # Confidence band (±15%)
    fig.add_trace(go.Scatter(
        x=pd.concat([pred_df["timestamp"], pred_df["timestamp"][::-1]]),
        y=pd.concat([pred_df["pm2_5"] * 1.15, pred_df["pm2_5"][::-1] * 0.85]),
        fill="toself", fillcolor="rgba(99,102,241,0.1)",
        line=dict(color="rgba(0,0,0,0)"), name="Confidence band", showlegend=False,
    ))

    # Main forecast line
    fig.add_trace(go.Scatter(
        x=pred_df["timestamp"], y=pred_df["pm2_5"],
        mode="lines", name="PM2.5 Forecast",
        line=dict(color="#818cf8", width=2.5),
    ))

    # Coloured markers
    fig.add_trace(go.Scatter(
        x=pred_df["timestamp"], y=pred_df["pm2_5"],
        mode="markers", name="Hourly reading",
        marker=dict(color=pred_df["color"], size=5, line=dict(width=0)),
        showlegend=False,
        hovertemplate="<b>%{x|%b %d %H:%M}</b><br>PM2.5: %{y:.1f} µg/m³<extra></extra>",
    ))

    # Day separators
    for day_n in [24, 48]:
        ts = pred_df["timestamp"].iloc[day_n - 1] if len(pred_df) > day_n else None
        if ts:
            fig.add_vline(x=ts, line=dict(color="#252840", dash="dash", width=1))

    # Day labels
    for day_n, label in [(12, "Day 1"), (36, "Day 2"), (60, "Day 3")]:
        if len(pred_df) > day_n:
            fig.add_annotation(
                x=pred_df["timestamp"].iloc[day_n], y=pred_df["pm2_5"].max() * 1.1,
                text=label, showarrow=False,
                font=dict(size=11, color="#3d4466"), bgcolor="rgba(0,0,0,0)",
            )

    fig.update_layout(
        **CHART_LAYOUT,
        height=340,
        xaxis_title="",
        yaxis_title="PM2.5 (µg/m³)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── AQI legend ─────────────────────────────────────────────────────────
    cols = st.columns(6)
    legend = [("Good","#00e676","0–12"),("Moderate","#ffee58","12–35"),
              ("Unhealthy*","#ffa726","35–55"),("Unhealthy","#ef5350","55–150"),
              ("Very Unhealthy","#ab47bc","150–250"),("Hazardous","#b71c1c","250+")]
    for col, (lbl, clr, rng) in zip(cols, legend):
        with col:
            st.markdown(f"""
            <div style='text-align:center;padding:6px;background:{clr}18;border:1px solid {clr}33;border-radius:8px'>
              <div style='font-size:11px;font-weight:600;color:{clr}'>{lbl}</div>
              <div style='font-size:10px;color:#6b7299'>{rng} µg/m³</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Day-by-day tabs ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Day-by-Day Breakdown</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📅 Day 1", "📅 Day 2", "📅 Day 3"])

    for tab, day_n in zip([tab1, tab2, tab3], [1, 2, 3]):
        with tab:
            day_preds = pred_df[pred_df["day"] == day_n]
            if day_preds.empty:
                st.info("No data for this day.")
                continue

            day_avg  = round(day_preds["pm2_5"].mean(), 1)
            day_peak = round(day_preds["pm2_5"].max(),  1)
            day_min  = round(day_preds["pm2_5"].min(),  1)
            day_lbl, day_col = aqi_category(day_avg)

            c1, c2, c3, c4 = st.columns(4)
            for col, title, val, extra in [
                (c1, "Average PM2.5", f"{day_avg}", f"<span style='color:{day_col}'>{day_lbl}</span>"),
                (c2, "Peak PM2.5",    f"{day_peak}", "worst hour"),
                (c3, "Min PM2.5",     f"{day_min}",  "best hour"),
                (c4, "Date",          day_preds.iloc[0]["date"], f"{len(day_preds)} hours"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="day-card">
                      <div class="day-card-title">{title}</div>
                      <div class="day-card-val">{val}</div>
                      <div style='font-size:12px;color:#6b7299;margin-top:4px'>{extra}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Hourly bar chart
            fig_day = go.Figure()
            fig_day.add_trace(go.Bar(
                x=day_preds["time"],
                y=day_preds["pm2_5"],
                marker=dict(
                    color=day_preds["color"],
                    line=dict(width=0),
                ),
                hovertemplate="<b>%{x}</b><br>PM2.5: %{y:.1f} µg/m³<extra></extra>",
                name="PM2.5",
            ))
            fig_day.add_hline(y=35.4, line=dict(color="#ffa726", dash="dot", width=1),
                              annotation_text="Unhealthy threshold (35.4)", annotation_position="top left",
                              annotation_font=dict(size=10, color="#ffa726"))
            fig_day.update_layout(
                **CHART_LAYOUT, height=260,
                xaxis_title="Hour", yaxis_title="PM2.5 (µg/m³)",
                bargap=0.15, showlegend=False,
            )
            st.plotly_chart(fig_day, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Raw forecast table ──────────────────────────────────────────────────
    with st.expander("📋 Full Forecast Data Table"):
        display_df = pred_df[["timestamp","day","time","pm2_5","label"]].copy()
        display_df.columns = ["Timestamp", "Day", "Time", "PM2.5 (µg/m³)", "Category"]
        st.dataframe(
            display_df.style.applymap(
                lambda v: f"color: {AQI_COLORS.get(v, '#9aa0b8')}",
                subset=["Category"]
            ),
            use_container_width=True,
            height=320,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: EDA & HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 EDA & History":

    st.markdown("""
    <div style='font-size:24px;font-weight:700;color:#e8eaf6;margin-bottom:4px'>
      Exploratory Data Analysis
    </div>
    <div style='font-size:14px;color:#6b7299;margin-bottom:24px'>Historical trends, distributions and pollutant correlations</div>
    """, unsafe_allow_html=True)

    if df.empty:
        st.warning("No historical data found.")
        st.stop()

    # ── PM2.5 historical trend ─────────────────────────────────────────────
    st.markdown('<div class="section-header">PM2.5 Historical Trend</div>', unsafe_allow_html=True)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=df["datetime"], y=df["pm2_5"],
        mode="lines+markers",
        line=dict(color="#818cf8", width=2),
        marker=dict(size=4, color="#818cf8"),
        name="PM2.5",
        hovertemplate="<b>%{x}</b><br>PM2.5: %{y:.2f} µg/m³<extra></extra>",
    ))
    if len(df) >= 3:
        rolling = df["pm2_5"].rolling(3, min_periods=1).mean()
        fig_trend.add_trace(go.Scatter(
            x=df["datetime"], y=rolling,
            mode="lines", line=dict(color="#f472b6", width=1.5, dash="dot"),
            name="3-pt Moving Avg",
        ))
    fig_trend.update_layout(**CHART_LAYOUT, height=300,
                            yaxis_title="PM2.5 (µg/m³)", xaxis_title="")
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── All pollutants multi-line ──────────────────────────────────────────
    pollutants_available = [c for c in ["pm2_5","pm10","no2","o3","so2","co"] if c in df.columns]
    if pollutants_available:
        st.markdown('<div class="section-header">All Pollutants Over Time</div>', unsafe_allow_html=True)
        colors_p = ["#818cf8","#34d399","#fbbf24","#f472b6","#60a5fa","#a78bfa"]
        fig_poll = go.Figure()
        for i, p in enumerate(pollutants_available):
            fig_poll.add_trace(go.Scatter(
                x=df["datetime"], y=df[p],
                mode="lines", name=p.upper().replace("_",""),
                line=dict(color=colors_p[i % len(colors_p)], width=1.8),
            ))
        fig_poll.update_layout(**CHART_LAYOUT, height=300, xaxis_title="", yaxis_title="Concentration")
        st.plotly_chart(fig_poll, use_container_width=True)

    col_l, col_r = st.columns(2)

    # ── Hourly distribution ────────────────────────────────────────────────
    with col_l:
        st.markdown('<div class="section-header">PM2.5 Distribution</div>', unsafe_allow_html=True)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df["pm2_5"], nbinsx=20,
            marker=dict(color="#818cf8", line=dict(color="#252840", width=1)),
            name="PM2.5",
        ))
        fig_hist.update_layout(**CHART_LAYOUT, height=260,
                               xaxis_title="PM2.5 (µg/m³)", yaxis_title="Count", showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── PM2.5 by hour of day ───────────────────────────────────────────────
    with col_r:
        st.markdown('<div class="section-header">Average PM2.5 by Hour</div>', unsafe_allow_html=True)
        if "hour" in df.columns:
            hourly = df.groupby("hour")["pm2_5"].mean().reset_index()
            fig_hr = go.Figure()
            fig_hr.add_trace(go.Bar(
                x=hourly["hour"], y=hourly["pm2_5"],
                marker=dict(color="#34d399", line=dict(width=0)),
                name="Avg PM2.5",
            ))
            fig_hr.update_layout(**CHART_LAYOUT, height=260,
                                 xaxis_title="Hour of Day", yaxis_title="Avg PM2.5", showlegend=False)
            st.plotly_chart(fig_hr, use_container_width=True)

    # ── Correlation heatmap ────────────────────────────────────────────────
    numeric_cols = [c for c in pollutants_available + ["hour","day_of_week","is_weekend"]
                    if c in df.columns]
    if len(numeric_cols) >= 2:
        st.markdown('<div class="section-header">Correlation Matrix</div>', unsafe_allow_html=True)
        corr = df[numeric_cols].corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale=[[0,"#3730a3"],[0.5,"#12141f"],[1,"#818cf8"]],
            zmid=0, text=np.round(corr.values,2),
            texttemplate="%{text}", textfont=dict(size=10),
            colorbar=dict(tickfont=dict(color="#9aa0b8")),
        ))
        fig_corr.update_layout(**CHART_LAYOUT, height=340)
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Summary statistics ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Summary Statistics</div>', unsafe_allow_html=True)
    numeric_df = df[pollutants_available].describe().round(3) if pollutants_available else pd.DataFrame()
    if not numeric_df.empty:
        st.dataframe(numeric_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":

    st.markdown("""
    <div style='font-size:24px;font-weight:700;color:#e8eaf6;margin-bottom:4px'>Model Performance</div>
    <div style='font-size:14px;color:#6b7299;margin-bottom:24px'>Training metrics, feature importance (SHAP), and model comparison</div>
    """, unsafe_allow_html=True)

    metrics = load_metrics()

    if metrics:
        best = metrics.get("best_model", "N/A")
        st.markdown(f"""
        <div style='background:#1a1d2e;border:1px solid #34d39933;border-left:4px solid #34d399;
                    border-radius:12px;padding:14px 20px;margin-bottom:24px'>
          <span style='font-size:13px;color:#34d399;font-weight:600'>✓ Best model selected: </span>
          <span style='font-size:13px;color:#c5cae9'>{best.replace("_"," ").title()}</span>
        </div>
        """, unsafe_allow_html=True)

        # Comparison table
        st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
        model_data = []
        for name, m in metrics.get("models", {}).items():
            model_data.append({
                "Model":  name.replace("_", " ").title(),
                "RMSE":   m.get("rmse", "—"),
                "MAE":    m.get("mae",  "—"),
                "R²":     m.get("r2",   "—"),
                "Best":   "✅" if name == best else "",
            })

        if model_data:
            comp_df = pd.DataFrame(model_data)

            # Bar chart comparison
            fig_comp = go.Figure()
            model_names = [d["Model"] for d in model_data]
            colors_m    = ["#818cf8", "#34d399", "#fbbf24"]

            for i, metric_name in enumerate(["RMSE", "MAE"]):
                vals = [d[metric_name] for d in model_data]
                fig_comp.add_trace(go.Bar(
                    name=metric_name, x=model_names,
                    y=[v if isinstance(v, (int, float)) else 0 for v in vals],
                    marker_color=colors_m[i],
                    text=[f"{v:.4f}" if isinstance(v, (int, float)) else "—" for v in vals],
                    textposition="outside",
                    textfont=dict(size=11),
                ))
            fig_comp.update_layout(
                **CHART_LAYOUT, height=300, barmode="group",
                yaxis_title="Error (µg/m³)", xaxis_title="",
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            st.dataframe(comp_df, use_container_width=True, hide_index=True)
    else:
        st.info("No metrics found. Run `python train_model.py` to train and evaluate models.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SHAP Feature Importance ────────────────────────────────────────────
    st.markdown('<div class="section-header">SHAP Feature Importance</div>', unsafe_allow_html=True)

    fi = load_feature_importance()
    if fi:
        # Top 15 features bar chart
        top_fi = dict(list(fi.items())[:15])
        fi_df  = pd.DataFrame({"Feature": list(top_fi.keys()), "Importance": list(top_fi.values())})
        fi_df  = fi_df.sort_values("Importance")

        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"],
            orientation="h",
            marker=dict(
                color=fi_df["Importance"],
                colorscale=[[0,"#252840"],[0.5,"#818cf8"],[1,"#f472b6"]],
                line=dict(width=0),
            ),
            text=[f"{v:.4f}" for v in fi_df["Importance"]],
            textposition="outside",
            textfont=dict(size=10, color="#9aa0b8"),
        ))
        layout = CHART_LAYOUT.copy()
        layout["margin"] = dict(l=160, r=60, t=20, b=20)
        fig_fi.update_layout( **layout, height=420, xaxis_title="Mean |SHAP value|", showlegend=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    elif os.path.exists(SHAP_PATH):
        st.image(SHAP_PATH, caption="SHAP Summary Plot", use_container_width=True)
    else:
        st.info("SHAP values not yet computed. Run `python train_model.py` with enough data.")

    # ── Prediction vs Actual scatter ───────────────────────────────────────
    if model and not df.empty:
        st.markdown('<div class="section-header">Predicted vs Actual</div>', unsafe_allow_html=True)
        DROP_COLS    = ["datetime", "target"]
        feature_cols = [c for c in df.columns if c not in DROP_COLS]
        X_all        = df[feature_cols].fillna(0)
        y_actual     = df["target"] if "target" in df.columns else df["pm2_5"]

        try:
            y_pred = model.predict(X_all)
            fig_pva = go.Figure()
            fig_pva.add_trace(go.Scatter(
                x=y_actual, y=y_pred, mode="markers",
                marker=dict(color="#818cf8", size=7, opacity=0.7,
                            line=dict(color="#252840", width=0.5)),
                name="Predictions",
                hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
            ))
            mn = min(y_actual.min(), y_pred.min())
            mx = max(y_actual.max(), y_pred.max())
            fig_pva.add_trace(go.Scatter(
                x=[mn, mx], y=[mn, mx],
                mode="lines", line=dict(color="#34d399", dash="dash", width=1.5),
                name="Perfect prediction",
            ))
            fig_pva.update_layout(**CHART_LAYOUT, height=320,
                                  xaxis_title="Actual PM2.5", yaxis_title="Predicted PM2.5")
            st.plotly_chart(fig_pva, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate prediction scatter: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":

    st.markdown("""
    <div style='font-size:24px;font-weight:700;color:#e8eaf6;margin-bottom:4px'>About This Project</div>
    <div style='font-size:14px;color:#6b7299;margin-bottom:24px'>Architecture, data sources, and how to use</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#1a1d2e;border:1px solid #252840;border-radius:16px;padding:24px 28px;margin-bottom:20px'>
      <div style='font-size:17px;font-weight:600;color:#c5cae9;margin-bottom:16px'>Pipeline Architecture</div>
      <div style='font-size:14px;color:#9aa0b8;line-height:2'>
        <b style='color:#818cf8'>1. Data Collection</b> → OpenWeatherMap Air Pollution API (PM2.5, PM10, NO₂, O₃, SO₂, CO)<br>
        <b style='color:#34d399'>2. Feature Engineering</b> → Time features, cyclical encodings, lag features, rolling statistics<br>
        <b style='color:#fbbf24'>3. Model Training</b> → Random Forest, Gradient Boosting, Ridge with TimeSeriesCV<br>
        <b style='color:#f472b6'>4. SHAP Explainability</b> → Feature importance for best model<br>
        <b style='color:#60a5fa'>5. 72-Hour Forecast</b> → Rolling prediction using lag features<br>
        <b style='color:#a78bfa'>6. Alerts Engine</b> → Flags periods exceeding WHO/EPA thresholds<br>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
        <div style='background:#1a1d2e;border:1px solid #252840;border-radius:12px;padding:20px 24px'>
          <div style='font-size:15px;font-weight:600;color:#c5cae9;margin-bottom:12px'>How to Run</div>
          <div style='font-size:13px;color:#9aa0b8;line-height:2.2;font-family:JetBrains Mono,monospace'>
            python fetch_data.py<br>
            python build_features.py<br>
            python train_model.py<br>
            uvicorn main:app --reload<br>
            streamlit run app.py
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col_r:
        st.markdown("""
        <div style='background:#1a1d2e;border:1px solid #252840;border-radius:12px;padding:20px 24px'>
          <div style='font-size:15px;font-weight:600;color:#c5cae9;margin-bottom:12px'>PM2.5 Thresholds</div>
          <div style='font-size:13px;line-height:2.2'>
            <span style='color:#00e676'>●</span> <span style='color:#9aa0b8'>Good: 0–12 µg/m³</span><br>
            <span style='color:#ffee58'>●</span> <span style='color:#9aa0b8'>Moderate: 12–35 µg/m³</span><br>
            <span style='color:#ffa726'>●</span> <span style='color:#9aa0b8'>Unhealthy*: 35–55 µg/m³</span><br>
            <span style='color:#ef5350'>●</span> <span style='color:#9aa0b8'>Unhealthy: 55–150 µg/m³</span><br>
            <span style='color:#ab47bc'>●</span> <span style='color:#9aa0b8'>Very Unhealthy: 150–250 µg/m³</span><br>
            <span style='color:#b71c1c'>●</span> <span style='color:#9aa0b8'>Hazardous: 250+ µg/m³</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#1a1d2e;border:1px solid #252840;border-radius:12px;padding:20px 24px'>
      <div style='font-size:15px;font-weight:600;color:#c5cae9;margin-bottom:12px'>CI/CD Automation</div>
      <div style='font-size:13px;color:#9aa0b8;line-height:1.8'>
        GitHub Actions workflows automate the pipeline:
        <br>• <b style='color:#818cf8'>.github/workflows/feature_pipeline.yml</b> — runs every hour (fetches + builds features)
        <br>• <b style='color:#34d399'>.github/workflows/training_pipeline.yml</b> — runs daily at 2 AM UTC (retrains model)
        <br><br>
      </div>
    </div>
    """, unsafe_allow_html=True)
