"""
app_v2.py
Phase 5B: Redesigned Streamlit Dashboard -- Command Centre Edition
Run: streamlit run phase5_dashboard/app_v2.py
"""

import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import h3
import folium
from folium.plugins import HeatMap
from datetime import datetime

# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Gridlock AI · Parking Intelligence",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------ #
# Cloud data bootstrap
# Downloads large files from Google Drive if they are missing.
# On local dev: files already exist → this block is a no-op.
# On Streamlit Cloud: uses st.secrets["drive"]["predictions_id"].
# ------------------------------------------------------------------ #
def _download_from_drive(file_id: str, dest: str) -> None:
    """Download a file from Google Drive using gdown."""
    try:
        import gdown
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, dest, quiet=False)
    except Exception as e:
        st.warning(f"Could not download {dest} from Drive: {e}")

def _bootstrap_data() -> None:
    """Ensure all required data files are present."""
    files = {
        "outputs/predictions.json": ("drive", "predictions_id"),
    }
    for local_path, (section, key) in files.items():
        if os.path.exists(local_path):
            continue  # already present (local dev or cached)
        # Running on Streamlit Cloud — try secrets
        try:
            file_id = st.secrets[section][key]
            if file_id and file_id != "PASTE_GOOGLE_DRIVE_FILE_ID_HERE":
                with st.spinner(f"⬇️ Downloading {local_path} from Google Drive…"):
                    _download_from_drive(file_id, local_path)
        except (KeyError, FileNotFoundError):
            pass  # secret not set — app will show a graceful error later

_bootstrap_data()


# ------------------------------------------------------------------ #
# Paths
# ------------------------------------------------------------------ #
P = {
    "heatmap":     "outputs/heatmap_updated.html",
    "predictions": "outputs/predictions.json",
    "deploy_plan": "outputs/deployment_plan.json",
    "deploy_map":  "outputs/deployment_map.html",
    "detections":  "outputs/detection_log.json",
    "video":       "outputs/cv_annotated_output.mp4",
    "fi_chart":    "outputs/feature_importance.png",
    "chronic":     "outputs/chronic_hotspots.json",
    "recovery":    "outputs/capacity_recovery.json",
    "hex_agg":     "data/processed/hex_aggregated.csv",
}

# ------------------------------------------------------------------ #
# Global CSS -- Command Centre Theme
# ------------------------------------------------------------------ #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

* { font-family: 'Inter', sans-serif !important; }
code, .mono { font-family: 'JetBrains Mono', monospace !important; }

/* Root */
.stApp {
    background: #050a15;
    color: #cdd9e5;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #080e1a;
    border-right: 1px solid #1a2540;
}

/* -- HEADER BANNER -- */
.cmd-header {
    background: linear-gradient(135deg, #080e1a 0%, #0d1a2e 50%, #080e1a 100%);
    border: 1px solid #1a2540;
    border-radius: 12px;
    padding: 18px 28px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
}
.cmd-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00d4ff, #0066ff, #6600ff, #00d4ff);
    animation: shimmer 3s linear infinite;
    background-size: 200% 100%;
}
@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position:  200% 0; }
}
.cmd-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: #e6edf3;
    letter-spacing: -0.02em;
}
.cmd-subtitle {
    font-size: 0.8rem;
    color: #5a7a9a;
    margin-top: 3px;
    font-weight: 400;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.cmd-badge {
    background: #0a1f15;
    border: 1px solid #00ff88;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 11px;
    color: #00ff88;
    font-weight: 700;
    letter-spacing: 0.08em;
    animation: pulse-green 2s infinite;
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 0 rgba(0,255,136,0.3); }
    50%       { box-shadow: 0 0 0 6px rgba(0,255,136,0); }
}
.cmd-time {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    color: #00d4ff;
    font-weight: 600;
}
.cmd-date {
    font-size: 0.72rem;
    color: #5a7a9a;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* -- KPI CARDS -- */
.kpi-card {
    background: linear-gradient(135deg, #080e1a 0%, #0d1627 100%);
    border: 1px solid #1a2540;
    border-radius: 10px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
    height: 100%;
}
.kpi-card:hover { border-color: #00d4ff55; }
.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
}
.kpi-card.blue::after   { background: #00d4ff; }
.kpi-card.red::after    { background: #ff4136; }
.kpi-card.orange::after { background: #ff851b; }
.kpi-card.cyan::after   { background: #00ffff; }
.kpi-card.green::after  { background: #00ff88; }
.kpi-icon  { font-size: 1.3rem; margin-bottom: 8px; display: block; }
.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 4px;
    font-variant-numeric: tabular-nums;
}
.kpi-label {
    font-size: 0.7rem;
    color: #5a7a9a;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 600;
}
.kpi-delta { font-size: 0.7rem; margin-top: 4px; font-weight: 600; color: #3a5a7a; }

/* -- TABS -- */
.stTabs [data-baseweb="tab-list"] {
    background: #080e1a;
    border: 1px solid #1a2540;
    border-radius: 10px;
    padding: 4px 6px;
    gap: 2px;
    margin-bottom: 16px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #5a7a9a;
    border-radius: 7px;
    font-weight: 600;
    font-size: 13px;
    padding: 8px 18px;
    border: none;
    transition: all 0.15s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0d1e38, #102040) !important;
    color: #00d4ff !important;
    border: 1px solid #1a3560 !important;
}

/* -- SECTION TITLE -- */
.sec-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: #00d4ff;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    padding: 0 0 10px 0;
    border-bottom: 1px solid #1a2540;
    margin-bottom: 16px;
}

/* -- GLASS PANEL -- */
.glass {
    background: rgba(13, 22, 39, 0.6);
    border: 1px solid #1a2540;
    border-radius: 10px;
    padding: 16px;
    backdrop-filter: blur(8px);
}

/* -- UNIT CARD -- */
.unit-card {
    background: #080e1a;
    border-radius: 8px;
    padding: 11px 14px;
    margin-bottom: 7px;
    border-left: 3px solid;
    transition: background 0.15s;
}
.unit-card:hover { background: #0d1627; }

/* -- RISK BADGES -- */
.risk-critical { background:#1f0505; color:#ff4136; padding:2px 10px; border-radius:12px; font-size:11px; font-weight:700; border:1px solid #ff413655; }
.risk-high     { background:#1f1005; color:#ff851b; padding:2px 10px; border-radius:12px; font-size:11px; font-weight:700; border:1px solid #ff851b55; }
.risk-medium   { background:#1f1b05; color:#ffdc00; padding:2px 10px; border-radius:12px; font-size:11px; font-weight:700; border:1px solid #ffdc0055; }
.risk-low      { background:#051f0d; color:#00ff88; padding:2px 10px; border-radius:12px; font-size:11px; font-weight:700; border:1px solid #00ff8855; }

/* -- CHRONIC BADGES -- */
.chronic-chronic    { background:#2d0a0a; color:#ff4136; padding:2px 10px; border-radius:12px; font-size:11px; font-weight:700; border:1px solid #ff4136; }
.chronic-persistent { background:#2d1a05; color:#ff851b; padding:2px 10px; border-radius:12px; font-size:11px; font-weight:700; border:1px solid #ff851b; }
.chronic-episodic   { background:#05152d; color:#00d4ff; padding:2px 10px; border-radius:12px; font-size:11px; font-weight:700; border:1px solid #00d4ff; }

/* -- RECOVERY BAR -- */
.recovery-bar-bg   { background:#1a2540; border-radius:4px; height:6px; margin-top:4px; }
.recovery-bar-fill { background:linear-gradient(90deg,#00d4ff,#00ff88); border-radius:4px; height:6px; }

/* -- DATAFRAME -- */
[data-testid="stDataFrame"] { border:1px solid #1a2540 !important; border-radius:8px !important; }
[data-testid="stDataFrame"] th { background:#080e1a !important; color:#5a7a9a !important; font-size:11px !important; text-transform:uppercase; letter-spacing:0.06em; }

/* -- PLOTLY OVERRIDE -- */
.js-plotly-plot { border-radius: 8px; overflow: hidden; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #050a15; }
::-webkit-scrollbar-thumb { background: #1a2540; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
@st.cache_data
def load_json(path):
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

def read_html(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def plotly_dark():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cdd9e5", family="Inter"),
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis=dict(gridcolor="#1a2540", zerolinecolor="#1a2540"),
        yaxis=dict(gridcolor="#1a2540", zerolinecolor="#1a2540"),
    )

@st.cache_data
def generate_deployment(predictions_list, num_units, hours_ahead):
    df = pd.DataFrame(predictions_list)
    unique_dts = sorted(df["target_datetime"].unique())
    target_dts = unique_dts[:hours_ahead]
    df_f = df[df["target_datetime"].isin(target_dts)]
    if df_f.empty:
        df_f = df
    candidates = (
        df_f.sort_values("predicted_violations", ascending=False)
           .drop_duplicates(subset="hex_id")
           .reset_index(drop=True)
    )
    covered, deployment, unit_id = set(), [], 1
    for _, c in candidates.iterrows():
        if unit_id > num_units:
            break
        hid = c["hex_id"]
        if hid in covered:
            continue
        ring = set(h3.grid_disk(hid, 1))
        new  = ring - covered
        covered.update(ring)
        deployment.append({
            "unit_id": f"UNIT-{unit_id:02d}",
            "hex_id": hid,
            "lat": round(float(c["lat"]), 6),
            "lon": round(float(c["lon"]), 6),
            "predicted_violations": round(float(c["predicted_violations"]), 1),
            "risk_level": c["risk_level"],
            "is_peak_hour": int(c["is_peak_hour"]),
            "deploy_for_datetime": c["target_datetime"],
            "hexes_covered": list(new),
            "total_hexes_covered": len(ring),
        })
        unit_id += 1
    return deployment

def build_deploy_map(deployment, predictions_list):
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=12, tiles="CartoDB dark_matter")
    df = pd.DataFrame(predictions_list)
    HeatMap(df[["lat","lon","predicted_violations"]].values.tolist(),
            min_opacity=0.2, radius=14, blur=18).add_to(m)
    COLOURS = {"CRITICAL":"#FF4136","HIGH":"#FF851B","MEDIUM":"#FFDC00","LOW":"#2ECC40"}
    for i, u in enumerate(deployment):
        c = COLOURS.get(u["risk_level"], "#AAAAAA")
        folium.Circle([u["lat"],u["lon"]], radius=750, color=c,
                      fill=True, fill_opacity=0.15, weight=2).add_to(m)
        folium.Marker([u["lat"],u["lon"]],
            icon=folium.DivIcon(
                html=f"<div style='background:{c};color:black;font-weight:bold;font-size:14px;"
                     f"border-radius:50%;width:30px;height:30px;display:flex;"
                     f"align-items:center;justify-content:center;border:2px solid white;"
                     f"box-shadow:0 2px 6px rgba(0,0,0,0.4);'>{i+1}</div>",
                icon_size=(30,30), icon_anchor=(15,15))).add_to(m)
    return m._repr_html_()


# ------------------------------------------------------------------ #
# Load data
# ------------------------------------------------------------------ #
predictions  = load_json(P["predictions"])
detections   = load_json(P["detections"])  if os.path.exists(P["detections"])  else []
chronic      = load_json(P["chronic"])     if os.path.exists(P["chronic"])     else []
recovery     = load_json(P["recovery"])    if os.path.exists(P["recovery"])    else []
hex_agg      = load_csv(P["hex_agg"])

pred_df     = pd.DataFrame(predictions)
chronic_df  = pd.DataFrame(chronic)  if chronic  else pd.DataFrame()
recovery_df = pd.DataFrame(recovery) if recovery else pd.DataFrame()


# ------------------------------------------------------------------ #
# SIDEBAR — Configuration
# ------------------------------------------------------------------ #
with st.sidebar:
    st.markdown("## 🚔 Gridlock AI")
    st.markdown("**Command Centre**")
    st.markdown("*Gridlock Hackathon 2.0 — Theme 1*")
    st.divider()
    st.markdown("### System Status")
    for phase in ["Phase 1 — Data Pipeline", "Phase 2 — LightGBM Predictor",
                  "Phase 3 — Patrol Optimizer", "Phase 4 — CV Detection Loop",
                  "Phase 5 — Dashboard v2"]:
        st.markdown(f"✅ **{phase}**")
    st.divider()
    st.markdown("### Model Stats")
    st.markdown("""
| Metric | Value |
|--------|-------|
| R² | **0.924** |
| MAE | **0.881** |
| RMSE | **10.002** |
| Training rows | **22,720** |
| H3 hexes | **770** |
    """)
    st.divider()
    st.markdown("### Configuration")
    num_units   = st.slider("Patrol Units",          4, 12, 8, 1)
    hours_ahead = st.slider("Forecast Window (hrs)", 1, 24, 6, 1)

# ------------------------------------------------------------------ #
# Dynamic deployment (responds to sidebar sliders)
# ------------------------------------------------------------------ #
deployment  = generate_deployment(predictions, num_units, hours_ahead)
deploy_df   = pd.DataFrame(deployment)

# ------------------------------------------------------------------ #
# KPI VALUES
# ------------------------------------------------------------------ #
total_violations   = int(hex_agg["violation_count"].sum())
critical_count     = int(pred_df[pred_df["risk_level"] == "CRITICAL"].shape[0])
units_deployed     = len(deployment)
cv_detections      = len(detections)
total_pcu_restored = int(sum(r.get("capacity_restored_pcu_hr", 0) for r in recovery))


# ================================================================== #
# HEADER
# ================================================================== #
now = datetime.now()
st.markdown(f"""
<div class="cmd-header">
    <div>
        <div class="cmd-title">🚔 Gridlock AI · Parking Intelligence</div>
        <div class="cmd-subtitle">
            Bengaluru Traffic Police &nbsp;·&nbsp;
            Theme 1: Parking-Induced Congestion &nbsp;·&nbsp;
            AI Enforcement System
        </div>
    </div>
    <div style="display:flex; align-items:center; gap:20px;">
        <span class="cmd-badge">● SYSTEM LIVE</span>
        <div style="text-align:right;">
            <div class="cmd-time">{now.strftime('%H:%M:%S')}</div>
            <div class="cmd-date">{now.strftime('%A, %d %B %Y')}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ================================================================== #
# KPI ROW
# ================================================================== #
k1, k2, k3, k4, k5 = st.columns(5)

kpi_data = [
    (k1, "blue",   "📊", f"{total_violations:,}", "Historical Violations",  "298,436 records · Jan–May"),
    (k2, "red",    "🔴", str(critical_count),      "Critical Zones (24h)",   "LightGBM R²=0.924"),
    (k3, "orange", "🚔", str(units_deployed),      "Patrol Units Deployed",  "Greedy max-coverage"),
    (k4, "cyan",   "🎥", str(cv_detections),       "CV Live Detections",     "YOLOv11 + ByteTrack"),
    (k5, "green",  "⚡", f"{total_pcu_restored:,}", "PCU/hr Restored",       "IndoHCM capacity model"),
]

colour_map = {"blue":"#00d4ff","red":"#ff4136","orange":"#ff851b","cyan":"#00ffff","green":"#00ff88"}

for col, cls, icon, value, label, sub in kpi_data:
    with col:
        st.markdown(f"""
        <div class="kpi-card {cls}">
            <span class="kpi-icon">{icon}</span>
            <div class="kpi-value" style="color:{colour_map[cls]};">{value}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-delta">{sub}</div>
        </div>
        """, unsafe_allow_html=True)


# ================================================================== #
# TABS
# ================================================================== #
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️  Heatmap",
    "📈  Predictions",
    "🚔  Deployment",
    "🔥  Chronic Hotspots",
    "🎥  CV Feed",
])


# ════════════════════════════════════════════════════════════════════ #
# TAB 1 — HEATMAP
# ════════════════════════════════════════════════════════════════════ #
with tab1:
    st.markdown('<div class="sec-title">Historical Violation Density + CV Detections · H3 Resolution 8</div>',
                unsafe_allow_html=True)
    left, right = st.columns([3, 1])

    with left:
        components.html(read_html(P["heatmap"]), height=560)

    with right:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("**Top Priority Zones**")
        top10 = (
            hex_agg.sort_values("priority_score", ascending=False)
            .head(10)[["hex_id", "violation_count", "priority_score"]]
            .reset_index(drop=True)
        )
        top10.index += 1
        top10.columns = ["Hex", "Violations", "Score"]
        top10["Score"] = top10["Score"].round(1)
        st.dataframe(top10, use_container_width=True, height=220)

        st.markdown("**Vehicle Type Distribution**")
        vtype_data = pd.Series({
            "Wrong Parking": 180422,
            "No Parking":    98642,
            "Main Road":     19372,
        })
        fig_donut = go.Figure(go.Pie(
            values=vtype_data.values,
            labels=vtype_data.index,
            hole=0.55,
            marker=dict(colors=["#ff4136","#ff851b","#ffdc00"]),
            textfont=dict(color="#cdd9e5", size=11),
        ))
        fig_donut.update_layout(**plotly_dark(), height=180,
                                showlegend=True,
                                legend=dict(font=dict(size=10), x=0, y=-0.2, orientation="h"))
        st.plotly_chart(fig_donut, use_container_width=True)

        if cv_detections:
            st.markdown(f"""
            <div style="background:#001a1a; border:1px solid #00ffff44;
                 border-radius:8px; padding:10px; font-size:12px; color:#00ffff; margin-top:8px;">
                🎥 <b>{cv_detections}</b> live detections injected<br>
                <span style="color:#3a5a7a;">Visible as cyan markers above</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════ #
# TAB 2 — PREDICTIONS
# ════════════════════════════════════════════════════════════════════ #
with tab2:
    st.markdown('<div class="sec-title">Next 24h Violation Forecast · LightGBM Regressor · R²=0.924</div>',
                unsafe_allow_html=True)

    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        sel_risk = st.multiselect("Risk Level", ["CRITICAL","HIGH","MEDIUM","LOW"],
                                   default=["CRITICAL","HIGH"])
    with f2:
        hr = st.slider("Hour Window", 0, 23, (0, 23))
    with f3:
        top_n = st.number_input("Top N Hexes", 5, 100, 25, 5)

    filtered = pred_df[
        pred_df["risk_level"].isin(sel_risk) &
        pred_df["hour"].between(hr[0], hr[1])
    ].sort_values("predicted_violations", ascending=False).head(int(top_n))

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Predicted Violations by Hour**")
        hourly = filtered.groupby("hour")["predicted_violations"].sum().reset_index()
        fig_bar = px.bar(hourly, x="hour", y="predicted_violations",
                         color="predicted_violations",
                         color_continuous_scale=["#003d20","#ffdc00","#ff851b","#ff4136"])
        fig_bar.add_vrect(x0=7.5, x1=10.5, fillcolor="#ff4136", opacity=0.07,
                          line_width=0, annotation_text="AM Peak",
                          annotation_font_color="#ff4136", annotation_font_size=10)
        fig_bar.add_vrect(x0=16.5, x1=20.5, fillcolor="#ff4136", opacity=0.07,
                          line_width=0, annotation_text="PM Peak",
                          annotation_font_color="#ff4136", annotation_font_size=10)
        fig_bar.update_layout(**plotly_dark(), height=260,
                              coloraxis_showscale=False,
                              xaxis_title="Hour of Day",
                              yaxis_title="Predicted Violations")
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.markdown("**Risk Distribution (filtered)**")
        rc = filtered["risk_level"].value_counts().reset_index()
        rc.columns = ["Risk", "Count"]
        fig_risk = px.bar(rc, x="Risk", y="Count", color="Risk",
                          color_discrete_map={"CRITICAL":"#ff4136","HIGH":"#ff851b",
                                              "MEDIUM":"#ffdc00","LOW":"#00ff88"})
        fig_risk.update_layout(**plotly_dark(), height=260, showlegend=False,
                               xaxis_title="", yaxis_title="Hex-Hour Slots")
        st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown(f"**Top {int(top_n)} Highest-Risk Forecasts**")
    disp = filtered[["hex_id","target_datetime","hour",
                      "predicted_violations","risk_level","is_peak_hour"]].copy()
    disp["predicted_violations"] = disp["predicted_violations"].round(1)
    disp["risk_level"] = disp["risk_level"].map(
        {"CRITICAL":"🔴 CRITICAL","HIGH":"🟠 HIGH",
         "MEDIUM":"🟡 MEDIUM","LOW":"🟢 LOW"}
    )
    disp["is_peak_hour"] = disp["is_peak_hour"].map({1:"⚡ Yes", 0:"No"})
    disp.columns = ["Hex ID","Forecast Time","Hour","Pred. Violations","Risk","Peak?"]
    st.dataframe(disp, use_container_width=True, height=300)

    if os.path.exists(P["fi_chart"]):
        with st.expander("📊 Feature Importance — What Drives Violations?"):
            st.image(P["fi_chart"], use_column_width=True)


# ════════════════════════════════════════════════════════════════════ #
# TAB 3 — DEPLOYMENT
# ════════════════════════════════════════════════════════════════════ #
with tab3:
    st.markdown('<div class="sec-title">Patrol Deployment Command · Greedy Spatial Coverage · Capacity Recovery</div>',
                unsafe_allow_html=True)

    map_col, ctrl_col = st.columns([3, 2])

    with map_col:
        deploy_map_html = build_deploy_map(deployment, predictions)
        components.html(deploy_map_html, height=500)

    with ctrl_col:
        # Recovery summary banner
        if not recovery_df.empty:
            total_pcu = recovery_df["capacity_restored_pcu_hr"].sum()
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#001a0d,#002a15);
                 border:1px solid #00ff8844; border-radius:10px; padding:14px;
                 margin-bottom:14px;">
                <div style="font-size:0.7rem; color:#00ff88; text-transform:uppercase;
                     letter-spacing:0.1em; font-weight:700; margin-bottom:6px;">
                    ⚡ ENFORCEMENT ROI
                </div>
                <div style="font-size:1.8rem; font-weight:800; color:#00ff88;">
                    {int(total_pcu):,} PCU/hr
                </div>
                <div style="font-size:0.75rem; color:#5a7a9a; margin-top:3px;">
                    Road capacity restored across all units
                </div>
            </div>""", unsafe_allow_html=True)

        # Deployment summary stats
        total_hexes = sum(u["total_hexes_covered"] for u in deployment)
        total_pred  = sum(u["predicted_violations"] for u in deployment)
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Units Out", units_deployed)
        with s2:
            st.metric("Hexes Covered", total_hexes)
        with s3:
            st.metric("Pred. Violations", f"{total_pred:.0f}")

        st.markdown("**Unit Assignments**")
        colour_map_risk = {"CRITICAL":"#ff4136","HIGH":"#ff851b",
                           "MEDIUM":"#ffdc00","LOW":"#00ff88"}

        for unit in deployment:
            c   = colour_map_risk.get(unit["risk_level"], "#5a7a9a")
            rec = next((r for r in recovery if r["unit_id"] == unit["unit_id"]), {})
            pcu_r = rec.get("capacity_restored_pcu_hr", 0)
            pct   = rec.get("congestion_reduction_pct", 0)
            bar_w = min(100, int(pct * 2))

            st.markdown(f"""
            <div class="unit-card" style="border-left-color:{c};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:700; color:{c}; font-size:13px;">{unit['unit_id']}</span>
                    <span class="risk-{unit['risk_level'].lower()}">{unit['risk_level']}</span>
                </div>
                <div style="font-size:11px; color:#5a7a9a; margin-top:4px; font-family:monospace;">
                    {unit['hex_id'][:14]}...
                    &nbsp;|&nbsp; {unit['total_hexes_covered']} hexes
                    &nbsp;|&nbsp; {unit['predicted_violations']:.0f} pred. violations
                </div>
                <div style="font-size:11px; color:#00ff88; margin-top:5px;">
                    ⚡ +{pcu_r:.0f} PCU/hr restored &nbsp;·&nbsp; -{pct:.1f}% congestion
                </div>
                <div class="recovery-bar-bg">
                    <div class="recovery-bar-fill" style="width:{bar_w}%;"></div>
                </div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════ #
# TAB 4 — CHRONIC HOTSPOTS
# ════════════════════════════════════════════════════════════════════ #
with tab4:
    st.markdown('<div class="sec-title">Chronic Hotspot Intelligence · Persistent vs Episodic Risk Classification</div>',
                unsafe_allow_html=True)

    if chronic_df.empty:
        st.warning("Run `python phase5b_enhancements/chronic_hotspot.py` first.")
    else:
        n_chronic    = int((chronic_df["persistence_class"] == "CHRONIC").sum())
        n_persistent = int((chronic_df["persistence_class"] == "PERSISTENT").sum())
        n_episodic   = int((chronic_df["persistence_class"] == "EPISODIC").sum())

        e1, e2, e3 = st.columns(3)
        with e1:
            st.markdown(f"""
            <div class="glass" style="border-color:#ff413655; text-align:center;">
                <div style="font-size:2rem; font-weight:800; color:#ff4136;">{n_chronic}</div>
                <div style="font-size:0.7rem; color:#ff4136; text-transform:uppercase; letter-spacing:0.1em; font-weight:700;">🔴 CHRONIC</div>
                <div style="font-size:11px; color:#5a7a9a; margin-top:6px;">Always in top 25%<br>Need permanent infrastructure</div>
            </div>""", unsafe_allow_html=True)
        with e2:
            st.markdown(f"""
            <div class="glass" style="border-color:#ff851b55; text-align:center;">
                <div style="font-size:2rem; font-weight:800; color:#ff851b;">{n_persistent}</div>
                <div style="font-size:0.7rem; color:#ff851b; text-transform:uppercase; letter-spacing:0.1em; font-weight:700;">🟠 PERSISTENT</div>
                <div style="font-size:11px; color:#5a7a9a; margin-top:6px;">Recurring high-risk<br>Need scheduled patrol</div>
            </div>""", unsafe_allow_html=True)
        with e3:
            st.markdown(f"""
            <div class="glass" style="border-color:#00d4ff55; text-align:center;">
                <div style="font-size:2rem; font-weight:800; color:#00d4ff;">{n_episodic}</div>
                <div style="font-size:0.7rem; color:#00d4ff; text-transform:uppercase; letter-spacing:0.1em; font-weight:700;">🔵 EPISODIC</div>
                <div style="font-size:11px; color:#5a7a9a; margin-top:6px;">Event-driven spikes<br>Monitor via CV</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns([2, 1])

        with c1:
            st.markdown("**Chronic Score Distribution by Hex**")
            fig_cs = px.histogram(
                chronic_df, x="chronic_score",
                color="persistence_class",
                nbins=30,
                color_discrete_map={
                    "CHRONIC":    "#ff4136",
                    "PERSISTENT": "#ff851b",
                    "EPISODIC":   "#00d4ff",
                },
                barmode="overlay",
            )
            fig_cs.add_vline(x=60, line_dash="dash", line_color="#ff4136",
                             annotation_text="Chronic threshold",
                             annotation_font_color="#ff4136", annotation_font_size=10)
            fig_cs.add_vline(x=30, line_dash="dash", line_color="#ff851b",
                             annotation_text="Persistent threshold",
                             annotation_font_color="#ff851b", annotation_font_size=10)
            fig_cs.update_layout(**plotly_dark(), height=280,
                                 xaxis_title="Chronic Score (%)",
                                 yaxis_title="Number of Hexes")
            st.plotly_chart(fig_cs, use_container_width=True)

        with c2:
            st.markdown("**Infrastructure Priority List**")
            top_chronic = (
                chronic_df[chronic_df["persistence_class"] == "CHRONIC"]
                .head(8)[["hex_id","chronic_score","recommended_action"]]
            )
            for _, row in top_chronic.iterrows():
                st.markdown(f"""
                <div class="glass" style="margin-bottom:7px; padding:10px 12px; border-left:3px solid #ff4136;">
                    <div style="font-family:monospace; font-size:11px; color:#ff4136;">{row['hex_id'][:16]}...</div>
                    <div style="font-size:11px; font-weight:700; color:#e6edf3; margin-top:2px;">Score: {row['chronic_score']:.1f}%</div>
                    <div style="font-size:10px; color:#5a7a9a; margin-top:3px;">{row['recommended_action']}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>**Full Chronic Hotspot Table**", unsafe_allow_html=True)
        disp_c = chronic_df[[
            "hex_id","chronic_score","persistence_class",
            "avg_violation_count","avg_impact","recommended_action"
        ]].copy()
        disp_c["chronic_score"]       = disp_c["chronic_score"].round(1)
        disp_c["avg_violation_count"] = disp_c["avg_violation_count"].round(1)
        disp_c["avg_impact"]          = disp_c["avg_impact"].round(1)
        disp_c.columns = ["Hex ID","Chronic Score %","Class",
                          "Avg Violations","Avg Impact","Recommended Action"]
        st.dataframe(disp_c, use_container_width=True, height=300)


# ════════════════════════════════════════════════════════════════════ #
# TAB 5 — CV FEED
# ════════════════════════════════════════════════════════════════════ #
with tab5:
    st.markdown('<div class="sec-title">CV Detection Loop · YOLOv11n + ByteTrack · Curbside ROI</div>',
                unsafe_allow_html=True)

    v_col, d_col = st.columns([3, 2])

    with v_col:
        if os.path.exists(P["video"]):
            with open(P["video"], "rb") as vf:
                st.video(vf.read())
            st.markdown("""
            <div style="font-size:11px; color:#5a7a9a; margin-top:6px; line-height:1.8;">
                🔴 <b style="color:#ff4136;">Red</b> = illegal curbside parking flagged &nbsp;·&nbsp;
                🟢 <b style="color:#00ff88;">Green</b> = moving / compliant vehicle<br>
                Curbside ROI = leftmost &amp; rightmost 15% of frame
            </div>""", unsafe_allow_html=True)
        else:
            st.info("cv_annotated_output.mp4 not found.")

    with d_col:
        st.markdown("""
        <div class="glass">
            <div style="font-size:0.7rem; color:#00d4ff; text-transform:uppercase;
                 letter-spacing:0.1em; font-weight:700; margin-bottom:12px;">
                The Feedback Loop
            </div>
            <div style="font-size:13px; line-height:2;">
                1️⃣ &nbsp;<b>Detect</b> — YOLOv11 spots vehicles in frame<br>
                2️⃣ &nbsp;<b>Track</b> — ByteTrack assigns persistent IDs<br>
                3️⃣ &nbsp;<b>Flag</b> — Stationary + curbside = violation<br>
                4️⃣ &nbsp;<b>Score</b> — IndoHCM capacity reduction calculated<br>
                5️⃣ &nbsp;<b>Inject</b> — Coord → H3 hex → heatmap updated<br>
                6️⃣ &nbsp;<b>Loop</b> — System learns from live detections
            </div>
        </div>""", unsafe_allow_html=True)

        if detections:
            st.markdown("<br>**Detection Log**", unsafe_allow_html=True)
            det_df = pd.DataFrame(detections)[
                ["vehicle_class","impact_score_norm","latitude","longitude"]
            ].copy()
            det_df.columns = ["Vehicle","Impact %","Lat","Lon"]
            det_df["Impact %"] = det_df["Impact %"].round(1)
            det_df["Lat"]      = det_df["Lat"].round(4)
            det_df["Lon"]      = det_df["Lon"].round(4)
            st.dataframe(det_df, use_container_width=True, height=160)

            total_loss = det_df["Impact %"].sum()
            st.markdown(f"""
            <div style="background:#001a0d; border:1px solid #00ff8844;
                 border-radius:8px; padding:10px; font-size:12px; margin-top:8px;">
                ✅ <b>{len(detections)}</b> violations detected &nbsp;·&nbsp;
                📉 <b>{total_loss:.1f}%</b> total capacity reduction<br>
                <span style="color:#3a5a7a; font-size:11px;">
                    Injected as cyan markers in Heatmap tab
                </span>
            </div>""", unsafe_allow_html=True)
