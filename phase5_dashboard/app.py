"""
app.py
Phase 5: Streamlit Dashboard -- Gridlock Hackathon 2.0
Run: streamlit run phase5_dashboard/app.py
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
from datetime import datetime, timedelta

# ------------------------------------------------------------------ #
# Page config -- must be first Streamlit call
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Gridlock AI -- Parking Intelligence",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
# Paths
# ------------------------------------------------------------------ #
OUTPUTS        = "outputs"
DATA_PROCESSED = "data/processed"

HEATMAP_PATH     = os.path.join(OUTPUTS, "heatmap_updated.html")
PREDICTIONS_PATH = os.path.join(OUTPUTS, "predictions.json")
DEPLOY_PLAN_PATH = os.path.join(OUTPUTS, "deployment_plan.json")
DEPLOY_MAP_PATH  = os.path.join(OUTPUTS, "deployment_map.html")
DETECTION_PATH   = os.path.join(OUTPUTS, "detection_log.json")
VIDEO_PATH       = os.path.join(OUTPUTS, "cv_annotated_output.mp4")
FI_CHART_PATH    = os.path.join(OUTPUTS, "feature_importance.png")
HEX_AGG_PATH     = os.path.join(DATA_PROCESSED, "hex_aggregated.csv")

# ------------------------------------------------------------------ #
# Custom CSS -- dark professional theme
# ------------------------------------------------------------------ #
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* Main background */
    .stApp { background-color: #0d1117; color: #e6edf3; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #161b22;
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #8b949e;
        border-radius: 6px;
        font-weight: 600;
        font-size: 14px;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        color: #e6edf3 !important;
    }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: border-color 0.2s;
    }
    .kpi-card:hover { border-color: #58a6ff; }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 6px;
    }
    .kpi-label {
        font-size: 0.78rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e6edf3;
        padding: 12px 0 8px 0;
        border-bottom: 1px solid #30363d;
        margin-bottom: 16px;
    }

    /* Unit cards */
    .unit-card {
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }

    /* Divider */
    hr { border-color: #30363d; }

    /* Dataframe */
    [data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
# Data loaders (cached)
# ------------------------------------------------------------------ #
@st.cache_data
def load_predictions():
    with open(PREDICTIONS_PATH) as f:
        return json.load(f)

@st.cache_data
def generate_dynamic_deployment(predictions_list, num_units, hours_ahead):
    df_pred = pd.DataFrame(predictions_list)
    unique_dts = sorted(df_pred['target_datetime'].unique())
    target_dts = unique_dts[:hours_ahead]
    
    df_filtered = df_pred[df_pred['target_datetime'].isin(target_dts)]
    if df_filtered.empty:
        df_filtered = df_pred
        
    candidates = (
        df_filtered.sort_values("predicted_violations", ascending=False)
          .drop_duplicates(subset="hex_id")
          .reset_index(drop=True)
    )
    
    ranked = candidates.sort_values("predicted_violations", ascending=False).reset_index(drop=True)
    covered_hexes = set()
    deployment = []
    unit_id = 1
    
    for _, candidate in ranked.iterrows():
        if unit_id > num_units: break
        hex_id = candidate["hex_id"]
        if hex_id in covered_hexes: continue
        
        covered_by_unit = set(h3.grid_disk(hex_id, 1))
        new_coverage = covered_by_unit - covered_hexes
        covered_hexes.update(covered_by_unit)
        
        deployment.append({
            "unit_id": f"UNIT-{unit_id:02d}",
            "hex_id": hex_id,
            "lat": round(float(candidate["lat"]), 6),
            "lon": round(float(candidate["lon"]), 6),
            "predicted_violations": round(float(candidate["predicted_violations"]), 1),
            "risk_level": candidate["risk_level"],
            "is_peak_hour": int(candidate["is_peak_hour"]),
            "deploy_for_datetime": candidate["target_datetime"],
            "hexes_covered": list(new_coverage),
            "total_hexes_covered": len(covered_by_unit),
        })
        unit_id += 1
        
    return deployment

def generate_dynamic_map(deployment, predictions_list):
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=12, tiles="CartoDB dark_matter")
    df_pred = pd.DataFrame(predictions_list)
    heat_data = df_pred[["lat", "lon", "predicted_violations"]].values.tolist()
    HeatMap(heat_data, name="Predicted", min_opacity=0.2, radius=14, blur=18).add_to(m)
    
    RISK_COLOURS = {"CRITICAL": "#FF4136", "HIGH": "#FF851B", "MEDIUM": "#FFDC00", "LOW": "#2ECC40"}
    
    for i, unit in enumerate(deployment):
        colour = RISK_COLOURS.get(unit["risk_level"], "#AAAAAA")
        folium.Circle(location=[unit["lat"], unit["lon"]], radius=750, color=colour, fill=True, fill_opacity=0.15, weight=2).add_to(m)
        
        icon_label = str(i+1)
        folium.Marker(
            location=[unit["lat"], unit["lon"]],
            icon=folium.DivIcon(
                html=f"<div style='background:{colour};color:black;font-weight:bold;font-size:14px;border-radius:50%;width:30px;height:30px;display:flex;align-items:center;justify-content:center;border:2px solid white;box-shadow:0 2px 6px rgba(0,0,0,0.4);'>{icon_label}</div>",
                icon_size=(30, 30), icon_anchor=(15, 15),
            )
        ).add_to(m)
        
    return m._repr_html_()

@st.cache_data
def load_detections():
    if not os.path.exists(DETECTION_PATH):
        return []
    with open(DETECTION_PATH) as f:
        return json.load(f)

@st.cache_data
def load_hex_agg():
    return pd.read_csv(HEX_AGG_PATH)

def read_html(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ------------------------------------------------------------------ #
# SIDEBAR
# ------------------------------------------------------------------ #
with st.sidebar:
    st.markdown("## 🚔 Gridlock AI")
    st.markdown("**Parking Intelligence System**")
    st.markdown("*Gridlock Hackathon 2.0 — Theme 1*")
    st.divider()

    st.markdown("### ✅ System Status")
    st.markdown("""
- ✅ **Phase 1** — Data Pipeline
- ✅ **Phase 2** — LightGBM Predictor
- ✅ **Phase 3** — Patrol Optimizer
- ✅ **Phase 4** — CV Detection Loop
- ✅ **Phase 5** — Dashboard
    """)
    st.divider()

    st.markdown("### 📊 Model Stats")
    st.markdown("""
| Metric | Value |
|--------|-------|
| R² Score | **0.924** |
| MAE | **0.881** |
| RMSE | **10.002** |
| Training rows | **22,720** |
| Hexes (H3 res 8) | **770** |
| Violations processed | **298,436** |
    """)
    st.divider()

    st.markdown("### ⚙️ Configuration")
    num_units   = st.slider("Patrol Units", min_value=4, max_value=12, value=8, step=1)
    hours_ahead = st.slider("Forecast Window (hours)", min_value=1, max_value=24, value=6, step=1)

    st.divider()
    st.markdown(
        "<div style='font-size:11px; color:#8b949e;'>"
        "IndoHCM 2017 capacity model<br>"
        "YOLOv11n + ByteTrack CV<br>"
        "H3 Resolution 8 hex grid"
        "</div>",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------ #
# Load data
# ------------------------------------------------------------------ #
predictions = load_predictions()
detections  = load_detections()
hex_agg     = load_hex_agg()
pred_df     = pd.DataFrame(predictions)

# DYNAMIC DEPLOYMENT
deployment  = generate_dynamic_deployment(predictions, num_units, hours_ahead)
deploy_df   = pd.DataFrame(deployment)


# ------------------------------------------------------------------ #
# KPI HEADER
# ------------------------------------------------------------------ #
st.markdown(
    "<h1 style='font-size:1.8rem; font-weight:800; margin-bottom:4px;'>"
    "🚔 Parking Intelligence Dashboard"
    "</h1>"
    "<p style='color:#8b949e; margin-top:0; font-size:0.9rem;'>"
    "Bengaluru Traffic Police · AI-Driven Enforcement System · Live"
    "</p>",
    unsafe_allow_html=True,
)

total_violations = int(hex_agg["violation_count"].sum())
critical_count   = len([p for p in predictions if p["risk_level"] == "CRITICAL"])
units_deployed   = len(deployment)
cv_detections    = len(detections)
top_impact       = round(float(hex_agg["avg_impact"].max()), 1)

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:#58a6ff;">{total_violations:,}</div>
        <div class="kpi-label">Historical Violations</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:#ff4136;">{critical_count}</div>
        <div class="kpi-label">Critical Zones (Next 24h)</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:#ff851b;">{units_deployed}</div>
        <div class="kpi-label">Patrol Units Deployed</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:#00ffff;">{cv_detections}</div>
        <div class="kpi-label">CV Live Detections</div>
    </div>""", unsafe_allow_html=True)

with k5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:#2ecc40;">{top_impact}</div>
        <div class="kpi-label">Peak Impact Score</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
# TABS
# ------------------------------------------------------------------ #
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Live Heatmap",
    "📈 Predictions",
    "🚔 Deployment Plan",
    "🎥 CV Demo",
])


# ================================================================== #
# TAB 1 -- LIVE HEATMAP
# ================================================================== #
with tab1:
    st.markdown('<div class="section-header">Historical Violation Heatmap + CV Detections</div>',
                unsafe_allow_html=True)

    left, right = st.columns([3, 1])

    with left:
        heatmap_html = read_html(HEATMAP_PATH)
        components.html(heatmap_html, height=560, scrolling=False)

    with right:
        st.markdown("**Top 10 Enforcement Zones**")
        top10 = (
            hex_agg.sort_values("priority_score", ascending=False)
            .head(10)[["hex_id", "violation_count", "priority_score"]]
            .reset_index(drop=True)
        )
        top10.index += 1
        top10.columns = ["Hex ID", "Violations", "Priority"]
        top10["Priority"] = top10["Priority"].round(1)
        st.dataframe(top10, width='stretch', height=300)

        st.markdown("**Violation Type Breakdown**")
        vtype_counts = pd.Series({
            "WRONG PARKING": 180422,
            "NO PARKING": 98642,
            "PARKING IN MAIN ROAD": 19372,
        })

        fig_pie = px.pie(
            values=vtype_counts.values,
            names=vtype_counts.index,
            color_discrete_sequence=["#ff4136", "#ff851b", "#ffdc00"],
            hole=0.45,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e6edf3",
            showlegend=True,
            legend=dict(font=dict(size=10)),
            margin=dict(t=10, b=10, l=10, r=10),
            height=220,
        )
        fig_pie.update_traces(textfont_color="#e6edf3")
        st.plotly_chart(fig_pie, width='stretch')

        if cv_detections > 0:
            st.markdown(
                f"<div style='background:#003d3d; border:1px solid #00ffff; "
                f"border-radius:8px; padding:10px; font-size:12px; color:#00ffff;'>"
                f"🎥 <b>{cv_detections} CV detections</b> injected<br>"
                f"<span style='color:#8b949e;'>Shown as cyan markers on map</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ================================================================== #
# TAB 2 -- PREDICTIONS
# ================================================================== #
with tab2:
    st.markdown(
        '<div class="section-header">Next 24-Hour Violation Forecast — LightGBM Model (R² = 0.924)</div>',
        unsafe_allow_html=True,
    )

    # Filters
    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        selected_risks = st.multiselect(
            "Filter by Risk Level",
            options=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            default=["CRITICAL", "HIGH"],
        )
    with f2:
        hour_range = st.slider("Hour Range", 0, 23, (0, 23))
    with f3:
        top_n = st.number_input("Show Top N Hexes", min_value=5, max_value=100, value=20, step=5)

    # Filter predictions
    filtered = pred_df[
        (pred_df["risk_level"].isin(selected_risks)) &
        (pred_df["hour"].between(hour_range[0], hour_range[1]))
    ].sort_values("predicted_violations", ascending=False).head(int(top_n))

    # Charts
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"**Predicted Violations by Hour ({len(filtered)} records)**")
        hourly = filtered.groupby("hour")["predicted_violations"].sum().reset_index()
        fig_bar = px.bar(
            hourly, x="hour", y="predicted_violations",
            color="predicted_violations",
            color_continuous_scale=["#2ecc40", "#ffdc00", "#ff851b", "#ff4136"],
            labels={"hour": "Hour of Day", "predicted_violations": "Predicted Violations"},
        )
        fig_bar.add_vrect(x0=7.5, x1=10.5, fillcolor="#ff4136",
                          opacity=0.1, line_width=0, annotation_text="AM Peak")
        fig_bar.add_vrect(x0=16.5, x1=20.5, fillcolor="#ff4136",
                          opacity=0.1, line_width=0, annotation_text="PM Peak")
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e6edf3", coloraxis_showscale=False,
            margin=dict(t=10, b=10, l=10, r=10), height=280,
            xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig_bar, width='stretch')

    with c2:
        st.markdown(f"**Risk Level Distribution (filtered)**")
        risk_counts = filtered["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        fig_risk = px.bar(
            risk_counts, x="Risk Level", y="Count",
            color="Risk Level",
            color_discrete_map={
                "CRITICAL": "#ff4136", "HIGH": "#ff851b",
                "MEDIUM": "#ffdc00",  "LOW":  "#2ecc40",
            },
        )
        fig_risk.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e6edf3", showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10), height=280,
            xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig_risk, width='stretch')

    # Prediction table
    st.markdown(f"**Top {int(top_n)} Highest-Risk Hex Forecasts**")
    display_df = filtered[[
        "hex_id", "target_datetime", "hour",
        "predicted_violations", "risk_level", "is_peak_hour", "lat", "lon"
    ]].copy()
    display_df["predicted_violations"] = display_df["predicted_violations"].round(1)
    display_df["risk_level"] = display_df["risk_level"].apply(
        lambda x: {"CRITICAL": "🔴 CRITICAL", "HIGH": "🟠 HIGH",
                   "MEDIUM": "🟡 MEDIUM", "LOW": "🟢 LOW"}.get(x, x)
    )
    display_df["is_peak_hour"] = display_df["is_peak_hour"].apply(
        lambda x: "⚡ Yes" if x == 1 else "No"
    )
    display_df.columns = [
        "Hex ID", "Forecast Time", "Hour",
        "Predicted Violations", "Risk Level", "Peak Hour", "Lat", "Lon"
    ]
    st.dataframe(display_df, width='stretch', height=320)

    # Feature importance
    if os.path.exists(FI_CHART_PATH):
        with st.expander("📊 Model Feature Importance"):
            st.image(FI_CHART_PATH, width='stretch')


# ================================================================== #
# TAB 3 -- DEPLOYMENT PLAN
# ================================================================== #
with tab3:
    st.markdown(
        '<div class="section-header">Greedy Spatial Coverage — Optimal Patrol Deployment</div>',
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([3, 2])

    with left_col:
        deploy_map_html = generate_dynamic_map(deployment, predictions)
        components.html(deploy_map_html, height=520, scrolling=False)

    with right_col:
        st.markdown("**Deployment Summary**")

        total_covered   = len(set(h for u in deployment for h in u["hexes_covered"]))
        total_pred_viol = sum(u["predicted_violations"] for u in deployment)

        m1, m2, m3 = st.columns(3)
        m1.metric("Units Out", len(deployment))
        m2.metric("Hexes Covered", total_covered)
        m3.metric("Violations Addressed", f"{total_pred_viol:.0f}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Unit Assignments**")

        for unit in deployment:
            risk   = unit["risk_level"]
            colour = {"CRITICAL": "#ff4136", "HIGH": "#ff851b",
                      "MEDIUM": "#ffdc00", "LOW": "#2ecc40"}.get(risk, "#888")
            peak_tag = "⚡ Peak" if unit["is_peak_hour"] else ""

            st.markdown(
                f"<div style='background:#161b22; border-left:4px solid {colour}; "
                f"border-radius:6px; padding:10px 14px; margin-bottom:8px;'>"
                f"<b style='color:{colour};'>{unit['unit_id']}</b> "
                f"<span style='color:#8b949e; font-size:12px;'>→ "
                f"{unit['hex_id'][:12]}...</span> {peak_tag}<br>"
                f"<span style='font-size:13px;'>"
                f"Predicted: <b>{unit['predicted_violations']}</b> violations | "
                f"Covers <b>{unit['total_hexes_covered']}</b> hexes</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Algorithm: Greedy Max-Coverage**")
        st.markdown(
            "<div style='background:#0d1117; border:1px solid #30363d; "
            "border-radius:8px; padding:12px; font-size:12px; color:#8b949e;'>"
            "Each unit assigned to highest-predicted uncovered hex. "
            "Coverage radius = H3 grid_disk(1) ≈ 7 hexes per unit. "
            "O(N log N) time complexity. No hex overlap guaranteed."
            "</div>",
            unsafe_allow_html=True,
        )


# ================================================================== #
# TAB 4 -- CV DEMO
# ================================================================== #
with tab4:
    st.markdown(
        '<div class="section-header">Computer Vision Detection Loop — YOLOv11n + ByteTrack</div>',
        unsafe_allow_html=True,
    )

    v_col, d_col = st.columns([3, 2])

    with v_col:
        if os.path.exists(VIDEO_PATH):
            st.markdown("**Annotated Dashcam Feed**")
            with open(VIDEO_PATH, "rb") as vf:
                video_bytes = vf.read()
            st.video(video_bytes)
            st.markdown(
                "<div style='font-size:12px; color:#8b949e; margin-top:6px;'>"
                "🔴 Red box = illegal curbside parking detected<br>"
                "🟢 Green box = moving / non-violation vehicle<br>"
                "Curbside ROI = leftmost and rightmost 15% of frame"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("cv_annotated_output.mp4 not found in outputs/. "
                       "Complete Phase 4 and download the video from Colab.")

    with d_col:
        st.markdown("**How the Feedback Loop Works**")
        st.markdown(
            "<div style='background:#161b22; border:1px solid #30363d; "
            "border-radius:8px; padding:14px; font-size:13px;'>"
            "1️⃣ <b>Detect</b> — YOLOv11 spots vehicles in frame<br><br>"
            "2️⃣ <b>Track</b> — ByteTrack assigns persistent IDs<br><br>"
            "3️⃣ <b>Flag</b> — Stationary + curbside = violation<br><br>"
            "4️⃣ <b>Score</b> — IndoHCM capacity reduction calculated<br><br>"
            "5️⃣ <b>Inject</b> — Coordinate → H3 hex → heatmap updated<br><br>"
            "6️⃣ <b>Loop</b> — System learns from live detections"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>**CV Detection Log**", unsafe_allow_html=True)

        if detections:
            det_df = pd.DataFrame(detections)[[
                "vehicle_class", "impact_score_norm",
                "latitude", "longitude", "hex_id"
            ]].copy()
            det_df.columns = ["Vehicle", "Impact %", "Lat", "Lon", "Hex ID"]
            det_df["Impact %"] = det_df["Impact %"].round(1)
            det_df["Lat"]      = det_df["Lat"].round(4)
            det_df["Lon"]      = det_df["Lon"].round(4)
            st.dataframe(det_df, width='stretch', height=220)

            total_cap_loss = det_df["Impact %"].sum()
            st.markdown(
                f"<div style='background:#003d00; border:1px solid #2ecc40; "
                f"border-radius:8px; padding:10px; font-size:13px; margin-top:8px;'>"
                f"✅ <b>{len(detections)}</b> violations detected<br>"
                f"📉 Total capacity reduction: <b>{total_cap_loss:.1f}%</b><br>"
                f"🗺️ Injected into heatmap as cyan markers"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No CV detections in log. Run Phase 4 Colab notebook to populate this.")

        st.markdown("<br>**Model Used**", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:12px; color:#8b949e;'>"
            "YOLOv11n (nano) via ultralytics<br>"
            "Tracker: ByteTrack (built-in)<br>"
            "Classes: car, motorcycle, bus, truck<br>"
            "Stationary threshold: 20px / 15 frames<br>"
            "Trained on COCO (Indian road fine-tuning optional)"
            "</div>",
            unsafe_allow_html=True,
        )
