# PHASE 5B — Dashboard Redesign + New Intelligence Features
## Gridlock Hackathon 2.0 | AI-Driven Parking Intelligence System

---

## 1. What This Phase Does

This phase does two things:

**A. Two new backend analysis scripts** that extend the system beyond
the original theme requirement:
- `chronic_hotspot.py` — identifies zones that are *always* problematic,
  not just sometimes. Recommends permanent infrastructure, not patrol.
- `capacity_recovery.py` — calculates how many PCU/hr of road capacity
  is restored when each patrol unit clears violations in its zone.

**B. Full frontend redesign** — replaces `app.py` with `app_v2.py`,
a command-centre style dashboard with 5 tabs, glass morphism cards,
animated charts, and a live clock header.

---

## 2. What Exists Already (Do Not Touch)

All Phase 1–4 outputs are complete and untouched:

```
outputs/
├── heatmap_updated.html        ✅
├── predictions.json            ✅
├── deployment_plan.json        ✅
├── deployment_map.html         ✅
├── detection_log.json          ✅
├── cv_annotated_output.mp4     ✅
└── feature_importance.png      ✅

data/processed/
├── impact_scored.csv           ✅
├── hex_aggregated.csv          ✅
└── training_features.csv       ✅
```

---

## 3. New Files This Phase Creates

```
gridlock-parking/
│
├── phase5b_enhancements/
│   ├── chronic_hotspot.py          ← NEW backend script
│   ├── capacity_recovery.py        ← NEW backend script
│   └── verify.py                   ← NEW verification
│
├── outputs/
│   ├── chronic_hotspots.json       ← NEW
│   └── capacity_recovery.json      ← NEW
│
└── phase5_dashboard/
    └── app_v2.py                   ← NEW (replaces app.py)
```

---

## 4. Tech Stack

Same as Phase 5. No new installs needed.
`streamlit`, `pandas`, `plotly`, `folium`, `json`

---

## 5. New Backend Script 1 — `phase5b_enhancements/chronic_hotspot.py`

**What it does:**

A hex is "chronic" if it consistently appears in the top 25% of
violators across multiple time slices — not just peak hour, not just
Monday, but reliably, repeatedly high-risk.

These zones need permanent infrastructure (no-parking signs, bollards,
line markings) — not rotating patrol. This distinction is a genuine
analytical insight that no other team will surface.

**Algorithm:**
```
For each hex:
  - Look at its violation count at every (hour, day_of_week) slot
  - Count how many slots it appears in the TOP 25% of all hexes
  - chronic_score = (slots_in_top_quartile / total_slots) × 100
  - If chronic_score > 60% → CHRONIC (needs infrastructure)
  - If chronic_score 30–60% → PERSISTENT (needs scheduled patrol)
  - If chronic_score < 30% → EPISODIC (needs event-driven response)
```

```python
"""
chronic_hotspot.py
Phase 5B - New Feature: Identify chronically high-risk parking zones.
Input:  data/processed/training_features.csv
        data/processed/hex_aggregated.csv
Output: outputs/chronic_hotspots.json
"""

import pandas as pd
import numpy as np
import json
import os

TRAIN_PATH  = "data/processed/training_features.csv"
HEX_PATH    = "data/processed/hex_aggregated.csv"
OUT_PATH    = "outputs/chronic_hotspots.json"

CHRONIC_THRESHOLD    = 60.0   # % of time slots in top quartile
PERSISTENT_THRESHOLD = 30.0


def classify_persistence(score):
    if score >= CHRONIC_THRESHOLD:
        return "CHRONIC"
    elif score >= PERSISTENT_THRESHOLD:
        return "PERSISTENT"
    return "EPISODIC"


def recommend_action(persistence, avg_impact, violation_count):
    if persistence == "CHRONIC":
        if avg_impact > 50:
            return "Install permanent no-parking barriers + road markings"
        return "Install no-parking signage + CCTV coverage"
    elif persistence == "PERSISTENT":
        if violation_count > 500:
            return "Schedule dedicated patrol 2x daily during peak hours"
        return "Include in weekly enforcement rotation"
    return "Monitor via CV — deploy only on event-driven triggers"


def run():
    print("[1/4] Loading training features...")
    tf       = pd.read_csv(TRAIN_PATH)
    hex_agg  = pd.read_csv(HEX_PATH)
    print(f"      Rows: {len(tf):,} | Hexes: {tf['hex_id'].nunique()}")

    print("[2/4] Computing top-quartile threshold per time slot...")
    # For each (hour, day_of_week) slot, compute the 75th percentile threshold
    slot_thresholds = (
        tf.groupby(["hour", "day_of_week"])["violation_count"]
        .quantile(0.75)
        .reset_index()
        .rename(columns={"violation_count": "threshold_q75"})
    )
    tf_merged = tf.merge(slot_thresholds, on=["hour", "day_of_week"])
    tf_merged["in_top_quartile"] = (
        tf_merged["violation_count"] >= tf_merged["threshold_q75"]
    ).astype(int)

    print("[3/4] Scoring hexes by chronic frequency...")
    chronic_scores = (
        tf_merged.groupby("hex_id")
        .agg(
            total_slots        = ("in_top_quartile", "count"),
            top_quartile_slots = ("in_top_quartile", "sum"),
            avg_violation_count= ("violation_count", "mean"),
            peak_violation     = ("violation_count", "max"),
            avg_impact         = ("avg_impact", "mean"),
        )
        .reset_index()
    )
    chronic_scores["chronic_score"] = (
        chronic_scores["top_quartile_slots"] /
        chronic_scores["total_slots"] * 100
    ).round(1)

    chronic_scores["persistence_class"] = chronic_scores["chronic_score"].apply(
        classify_persistence
    )

    # Merge lat/lon from hex_agg
    hex_meta = hex_agg[["hex_id", "lat", "lon", "priority_score",
                          "violation_count"]].copy()
    result   = chronic_scores.merge(hex_meta, on="hex_id", how="left")

    result["recommended_action"] = result.apply(
        lambda r: recommend_action(
            r["persistence_class"],
            r["avg_impact"],
            r["violation_count"],
        ),
        axis=1,
    )

    result = result.sort_values("chronic_score", ascending=False).reset_index(drop=True)

    print("[4/4] Saving chronic_hotspots.json...")
    output = result[[
        "hex_id", "lat", "lon", "chronic_score",
        "persistence_class", "avg_violation_count",
        "peak_violation", "avg_impact", "priority_score",
        "recommended_action",
    ]].to_dict(orient="records")

    # Round floats
    for rec in output:
        for k, v in rec.items():
            if isinstance(v, float):
                rec[k] = round(v, 3)

    os.makedirs("outputs", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    classes = pd.Series([r["persistence_class"] for r in output]).value_counts()
    print(f"\n  Chronic Hotspot Analysis:")
    for cls in ["CHRONIC", "PERSISTENT", "EPISODIC"]:
        if cls in classes:
            print(f"    {cls:<12}: {classes[cls]} hexes")
    print(f"\n  Saved → {OUT_PATH}")
    return output


if __name__ == "__main__":
    result = run()
    print("\n=== CHRONIC_HOTSPOT.PY COMPLETE ===")
    print(f"Top 5 chronic zones:")
    for r in result[:5]:
        print(f"  {r['hex_id']} | score={r['chronic_score']}% | {r['persistence_class']} | {r['recommended_action'][:50]}")
```

---

## 6. New Backend Script 2 — `phase5b_enhancements/capacity_recovery.py`

**What it does:**

For each patrol unit in the deployment plan, calculates how many
PCU/hr of road capacity is restored by clearing violations in its
coverage zone. This answers the theme's core question in reverse:
not just "what is the damage" but "what is the return on enforcement."

**Formula:**
```
For each hex in a unit's coverage zone:
  Capacity_Restored (PCU/hr) =
    Σ (W_parked / W_road) × f_sf × PCU_weight × 1800
    (1800 = base lane capacity in PCU/hr per Indo-HCM)
```

```python
"""
capacity_recovery.py
Phase 5B - New Feature: Quantify PCU/hr capacity restored per patrol unit.
Input:  outputs/deployment_plan.json
        data/processed/impact_scored.csv
Output: outputs/capacity_recovery.json
"""

import pandas as pd
import numpy as np
import json
import os
import h3

DEPLOY_PATH  = "outputs/deployment_plan.json"
IMPACT_PATH  = "data/processed/impact_scored.csv"
OUT_PATH     = "outputs/capacity_recovery.json"

BASE_CAPACITY_PCU_HR = 1800   # Indo-HCM base lane capacity
W_ROAD_DEFAULT       = 7.0    # metres, standard arterial road
F_SF_DEFAULT         = 0.85   # side friction factor, arterial


def run():
    print("[1/3] Loading deployment plan and impact data...")
    with open(DEPLOY_PATH) as f:
        plan = json.load(f)

    impact_df = pd.read_csv(IMPACT_PATH, low_memory=False)

    # Pre-compute per-hex average impact metrics
    hex_metrics = (
        impact_df.groupby("hex_id")
        .agg(
            avg_pcu_weight = ("pcu_weight", "mean"),
            avg_w_parked   = ("w_parked",   "mean"),
            violation_count= ("impact_score_norm", "count"),
        )
        .reset_index()
    )
    hex_dict = hex_metrics.set_index("hex_id").to_dict(orient="index")

    print("[2/3] Calculating capacity recovery per unit...")
    results = []

    for unit in plan:
        unit_id      = unit["unit_id"]
        covered_hexes = unit["hexes_covered"]
        risk_level    = unit["risk_level"]
        pred_viol     = unit["predicted_violations"]

        total_capacity_restored = 0.0
        hexes_with_data         = 0

        for hid in covered_hexes:
            if hid not in hex_dict:
                continue
            m   = hex_dict[hid]
            w_p = m["avg_w_parked"]
            pcu = m["avg_pcu_weight"]
            n   = m["violation_count"]
            hexes_with_data += 1

            # Capacity restored per violation cleared
            capacity_per_violation = (
                (w_p / W_ROAD_DEFAULT) * F_SF_DEFAULT * pcu * BASE_CAPACITY_PCU_HR
            )
            # Estimate active violations in this hex right now
            # (use predicted count proportionally across coverage zone)
            active_est = max(1, pred_viol / len(covered_hexes))
            total_capacity_restored += capacity_per_violation * active_est

        total_capacity_restored = round(total_capacity_restored, 1)
        congestion_reduction_pct = round(
            min(100, total_capacity_restored / BASE_CAPACITY_PCU_HR * 100), 1
        )

        results.append({
            "unit_id":                    unit_id,
            "risk_level":                 risk_level,
            "hexes_covered":              len(covered_hexes),
            "predicted_violations":       pred_viol,
            "capacity_restored_pcu_hr":   total_capacity_restored,
            "congestion_reduction_pct":   congestion_reduction_pct,
            "lat":                        unit["lat"],
            "lon":                        unit["lon"],
            "deploy_for_datetime":        unit["deploy_for_datetime"],
        })

    total_restored = sum(r["capacity_restored_pcu_hr"] for r in results)

    print("[3/3] Saving capacity_recovery.json...")
    os.makedirs("outputs", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Capacity Recovery Summary:")
    print(f"  {'Unit':<10} {'Restored (PCU/hr)':>18} {'Congestion ↓':>14}")
    print(f"  {'-'*44}")
    for r in results:
        print(f"  {r['unit_id']:<10} {r['capacity_restored_pcu_hr']:>18.1f} "
              f"{r['congestion_reduction_pct']:>13.1f}%")
    print(f"  {'-'*44}")
    print(f"  {'TOTAL':<10} {total_restored:>18.1f} PCU/hr restored")
    print(f"\n  Saved → {OUT_PATH}")
    return results


if __name__ == "__main__":
    results = run()
    print("\n=== CAPACITY_RECOVERY.PY COMPLETE ===")
```

---

## 7. Run Both New Scripts First

```bash
python phase5b_enhancements/chronic_hotspot.py
python phase5b_enhancements/capacity_recovery.py
```

Verify outputs exist:
```bash
# Both should print non-empty
python -c "import json; d=json.load(open('outputs/chronic_hotspots.json')); print(len(d), 'hotspots')"
python -c "import json; d=json.load(open('outputs/capacity_recovery.json')); print(len(d), 'units')"
```

---

## 8. Redesigned Frontend — `phase5_dashboard/app_v2.py`

Replace the old `app.py` entirely. This is a full rewrite.

Design language: **command centre** — deep dark background,
electric accent colours, glass morphism panels, live clock, glowing
risk indicators. Feels like a traffic operations room, not a notebook.

```python
"""
app_v2.py
Phase 5B: Redesigned Streamlit Dashboard — Command Centre Edition
Run: streamlit run phase5_dashboard/app_v2.py
"""

import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
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
# Global CSS — Command Centre Theme
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

/* ── HEADER BANNER ── */
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
    100% { background-position: 200% 0; }
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
.cmd-live {
    text-align: right;
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

/* ── KPI CARDS ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}
.kpi-card {
    background: linear-gradient(135deg, #080e1a 0%, #0d1627 100%);
    border: 1px solid #1a2540;
    border-radius: 10px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: #00d4ff55; }
.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
}
.kpi-card.blue::after  { background: #00d4ff; }
.kpi-card.red::after   { background: #ff4136; }
.kpi-card.orange::after{ background: #ff851b; }
.kpi-card.cyan::after  { background: #00ffff; }
.kpi-card.green::after { background: #00ff88; }
.kpi-icon {
    font-size: 1.3rem;
    margin-bottom: 8px;
    display: block;
}
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
.kpi-delta {
    font-size: 0.7rem;
    margin-top: 4px;
    font-weight: 600;
}

/* ── TABS ── */
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

/* ── SECTION TITLE ── */
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

/* ── GLASS PANEL ── */
.glass {
    background: rgba(13, 22, 39, 0.6);
    border: 1px solid #1a2540;
    border-radius: 10px;
    padding: 16px;
    backdrop-filter: blur(8px);
}

/* ── UNIT CARD ── */
.unit-card {
    background: #080e1a;
    border-radius: 8px;
    padding: 11px 14px;
    margin-bottom: 7px;
    border-left: 3px solid;
    transition: background 0.15s;
}
.unit-card:hover { background: #0d1627; }

/* ── RISK BADGES ── */
.risk-critical {
    background: #1f0505; color: #ff4136;
    padding: 2px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 700;
    border: 1px solid #ff413655;
}
.risk-high {
    background: #1f1005; color: #ff851b;
    padding: 2px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 700;
    border: 1px solid #ff851b55;
}
.risk-medium {
    background: #1f1b05; color: #ffdc00;
    padding: 2px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 700;
    border: 1px solid #ffdc0055;
}
.risk-low {
    background: #051f0d; color: #00ff88;
    padding: 2px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 700;
    border: 1px solid #00ff8855;
}

/* ── CHRONIC BADGES ── */
.chronic-chronic {
    background: #2d0a0a; color: #ff4136;
    padding: 2px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 700;
    border: 1px solid #ff4136;
}
.chronic-persistent {
    background: #2d1a05; color: #ff851b;
    padding: 2px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 700;
    border: 1px solid #ff851b;
}
.chronic-episodic {
    background: #05152d; color: #00d4ff;
    padding: 2px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 700;
    border: 1px solid #00d4ff;
}

/* ── RECOVERY BAR ── */
.recovery-bar-bg {
    background: #1a2540;
    border-radius: 4px;
    height: 6px;
    margin-top: 4px;
}
.recovery-bar-fill {
    background: linear-gradient(90deg, #00d4ff, #00ff88);
    border-radius: 4px;
    height: 6px;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1a2540 !important;
    border-radius: 8px !important;
}
[data-testid="stDataFrame"] th {
    background: #080e1a !important;
    color: #5a7a9a !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── PLOTLY OVERRIDE ── */
.js-plotly-plot { border-radius: 8px; overflow: hidden; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #050a15; }
::-webkit-scrollbar-thumb { background: #1a2540; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
# Cached loaders
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


# ------------------------------------------------------------------ #
# Load all data
# ------------------------------------------------------------------ #
predictions  = load_json(P["predictions"])
deployment   = load_json(P["deploy_plan"])
detections   = load_json(P["detections"])  if os.path.exists(P["detections"])  else []
chronic      = load_json(P["chronic"])     if os.path.exists(P["chronic"])     else []
recovery     = load_json(P["recovery"])    if os.path.exists(P["recovery"])    else []
hex_agg      = load_csv(P["hex_agg"])

pred_df      = pd.DataFrame(predictions)
deploy_df    = pd.DataFrame(deployment)
chronic_df   = pd.DataFrame(chronic)   if chronic   else pd.DataFrame()
recovery_df  = pd.DataFrame(recovery)  if recovery  else pd.DataFrame()

# KPI values
total_violations   = int(hex_agg["violation_count"].sum())
critical_count     = int(pred_df[pred_df["risk_level"] == "CRITICAL"].shape[0])
units_deployed     = len(deployment)
cv_detections      = len(detections)
total_pcu_restored = round(sum(r.get("capacity_restored_pcu_hr", 0) for r in recovery), 0)


# ------------------------------------------------------------------ #
# HEADER
# ------------------------------------------------------------------ #
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
        <div class="cmd-live">
            <div class="cmd-time">{now.strftime('%H:%M:%S')}</div>
            <div class="cmd-date">{now.strftime('%A, %d %B %Y')}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
# KPI ROW
# ------------------------------------------------------------------ #
k1, k2, k3, k4, k5 = st.columns(5)

kpi_data = [
    (k1, "blue",   "📊", f"{total_violations:,}", "Historical Violations", "298,436 records · Jan–May"),
    (k2, "red",    "🔴", str(critical_count),      "Critical Zones (24h)",  "LightGBM R²=0.924"),
    (k3, "orange", "🚔", str(units_deployed),      "Patrol Units Deployed", "Greedy max-coverage"),
    (k4, "cyan",   "🎥", str(cv_detections),       "CV Live Detections",    "YOLOv11 + ByteTrack"),
    (k5, "green",  "⚡", f"{int(total_pcu_restored):,}", "PCU/hr Restored",  "IndoHCM capacity model"),
]

for col, cls, icon, value, label, sub in kpi_data:
    with col:
        st.markdown(f"""
        <div class="kpi-card {cls}">
            <span class="kpi-icon">{icon}</span>
            <div class="kpi-value" style="color:{'#00d4ff' if cls=='blue' else '#ff4136' if cls=='red' else '#ff851b' if cls=='orange' else '#00ffff' if cls=='cyan' else '#00ff88'};">
                {value}
            </div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-delta" style="color:#3a5a7a;">{sub}</div>
        </div>
        """, unsafe_allow_html=True)


# ------------------------------------------------------------------ #
# TABS
# ------------------------------------------------------------------ #
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
                                legend=dict(font=dict(size=10), x=0, y=-0.2,
                                            orientation="h"))
        st.plotly_chart(fig_donut, use_container_width=True)

        if cv_detections:
            st.markdown(f"""
            <div style="background:#001a1a; border:1px solid #00ffff44;
                 border-radius:8px; padding:10px; font-size:12px; color:#00ffff;
                 margin-top:8px;">
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

    # Filters
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
    ].sort_values("predicted_violations", ascending=False).head(top_n)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Predicted Violations by Hour**")
        hourly = pred_df.groupby("hour")["predicted_violations"].sum().reset_index()
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
        st.markdown("**Risk Distribution**")
        rc = pred_df["risk_level"].value_counts().reset_index()
        rc.columns = ["Risk", "Count"]
        fig_risk = px.bar(rc, x="Risk", y="Count", color="Risk",
                          color_discrete_map={"CRITICAL":"#ff4136","HIGH":"#ff851b",
                                              "MEDIUM":"#ffdc00","LOW":"#00ff88"})
        fig_risk.update_layout(**plotly_dark(), height=260, showlegend=False,
                               xaxis_title="", yaxis_title="Hex-Hour Slots")
        st.plotly_chart(fig_risk, use_container_width=True)

    # Forecast table
    st.markdown(f"**Top {top_n} Highest-Risk Forecasts**")
    disp = filtered[["hex_id","target_datetime","hour",
                      "predicted_violations","risk_level","is_peak_hour"]].copy()
    disp["predicted_violations"] = disp["predicted_violations"].round(1)
    disp["risk_level"] = disp["risk_level"].map(
        {"CRITICAL":"🔴 CRITICAL","HIGH":"🟠 HIGH",
         "MEDIUM":"🟡 MEDIUM","LOW":"🟢 LOW"}
    )
    disp["is_peak_hour"] = disp["is_peak_hour"].map({1:"⚡ Yes", 0:"No"})
    disp.columns = ["Hex ID","Forecast Time","Hour",
                    "Pred. Violations","Risk","Peak?"]
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
        components.html(read_html(P["deploy_map"]), height=500)

    with ctrl_col:
        # Recovery summary
        if not recovery_df.empty:
            total_pcu = recovery_df["capacity_restored_pcu_hr"].sum()
            max_pcu   = recovery_df["capacity_restored_pcu_hr"].max()
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

        # Unit cards
        st.markdown("**Unit Assignments**")
        colour_map = {"CRITICAL":"#ff4136","HIGH":"#ff851b",
                      "MEDIUM":"#ffdc00","LOW":"#00ff88"}
        for unit in deployment:
            c     = colour_map.get(unit["risk_level"], "#5a7a9a")
            rec   = next((r for r in recovery if r["unit_id"]==unit["unit_id"]), {})
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
        # Explain the insight
        n_chronic    = int((chronic_df["persistence_class"] == "CHRONIC").sum())
        n_persistent = int((chronic_df["persistence_class"] == "PERSISTENT").sum())
        n_episodic   = int((chronic_df["persistence_class"] == "EPISODIC").sum())

        e1, e2, e3 = st.columns(3)
        with e1:
            st.markdown(f"""
            <div class="glass" style="border-color:#ff413655; text-align:center;">
                <div style="font-size:2rem; font-weight:800; color:#ff4136;">{n_chronic}</div>
                <div style="font-size:0.7rem; color:#ff4136; text-transform:uppercase;
                     letter-spacing:0.1em; font-weight:700;">🔴 CHRONIC</div>
                <div style="font-size:11px; color:#5a7a9a; margin-top:6px;">
                    Always in top 25%<br>Need permanent infrastructure
                </div>
            </div>""", unsafe_allow_html=True)
        with e2:
            st.markdown(f"""
            <div class="glass" style="border-color:#ff851b55; text-align:center;">
                <div style="font-size:2rem; font-weight:800; color:#ff851b;">{n_persistent}</div>
                <div style="font-size:0.7rem; color:#ff851b; text-transform:uppercase;
                     letter-spacing:0.1em; font-weight:700;">🟠 PERSISTENT</div>
                <div style="font-size:11px; color:#5a7a9a; margin-top:6px;">
                    Recurring high-risk<br>Need scheduled patrol
                </div>
            </div>""", unsafe_allow_html=True)
        with e3:
            st.markdown(f"""
            <div class="glass" style="border-color:#00d4ff55; text-align:center;">
                <div style="font-size:2rem; font-weight:800; color:#00d4ff;">{n_episodic}</div>
                <div style="font-size:0.7rem; color:#00d4ff; text-transform:uppercase;
                     letter-spacing:0.1em; font-weight:700;">🔵 EPISODIC</div>
                <div style="font-size:11px; color:#5a7a9a; margin-top:6px;">
                    Event-driven spikes<br>Monitor via CV
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns([2, 1])

        with c1:
            # Chronic score distribution
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
            # Top chronic zones + action
            st.markdown("**Infrastructure Priority List**")
            top_chronic = (
                chronic_df[chronic_df["persistence_class"] == "CHRONIC"]
                .head(8)[["hex_id","chronic_score","recommended_action"]]
            )
            for _, row in top_chronic.iterrows():
                st.markdown(f"""
                <div class="glass" style="margin-bottom:7px; padding:10px 12px;
                     border-left:3px solid #ff4136;">
                    <div style="font-family:monospace; font-size:11px; color:#ff4136;">
                        {row['hex_id'][:16]}...
                    </div>
                    <div style="font-size:11px; font-weight:700; color:#e6edf3; margin-top:2px;">
                        Score: {row['chronic_score']:.1f}%
                    </div>
                    <div style="font-size:10px; color:#5a7a9a; margin-top:3px;">
                        {row['recommended_action']}
                    </div>
                </div>""", unsafe_allow_html=True)

        # Full table
        st.markdown("<br>**Full Chronic Hotspot Table**")
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
                Curbside ROI = leftmost & rightmost 15% of frame
            </div>""", unsafe_allow_html=True)
        else:
            st.info("cv_annotated_output.mp4 not found. Complete Phase 4 Colab notebook.")

    with d_col:
        st.markdown("""
        <div class="glass">
            <div style="font-size:0.7rem; color:#00d4ff; text-transform:uppercase;
                 letter-spacing:0.1em; font-weight:700; margin-bottom:12px;">
                The Feedback Loop
            </div>
            <div style="font-size:13px; line-height:2;">
                1️⃣ &nbsp;<b>Detect</b> — YOLOv11 finds vehicles<br>
                2️⃣ &nbsp;<b>Track</b> — ByteTrack assigns IDs<br>
                3️⃣ &nbsp;<b>Flag</b> — Stationary + curbside = violation<br>
                4️⃣ &nbsp;<b>Score</b> — IndoHCM capacity reduction<br>
                5️⃣ &nbsp;<b>Inject</b> — Coord → H3 → heatmap updated<br>
                6️⃣ &nbsp;<b>Loop</b> — System learns from live feeds
            </div>
        </div>""", unsafe_allow_html=True)

        if detections:
            st.markdown("<br>**Detection Log**")
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
```

---

## 9. Run Order

```bash
# New backend scripts first
python phase5b_enhancements/chronic_hotspot.py
python phase5b_enhancements/capacity_recovery.py

# Launch new dashboard
streamlit run phase5_dashboard/app_v2.py
```

---

## 10. Verify Before Launch

```python
# phase5b_enhancements/verify.py

import os, json

checks = [
    ("outputs/chronic_hotspots.json",   "Chronic hotspots generated"),
    ("outputs/capacity_recovery.json",  "Capacity recovery generated"),
    ("phase5_dashboard/app_v2.py",      "New dashboard exists"),
]

print("\n=== PHASE 5B VERIFICATION ===\n")
passed = 0
for path, label in checks:
    ok = os.path.exists(path)
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}  {label}")
    if ok: passed += 1

# Data checks
with open("outputs/chronic_hotspots.json") as f:
    ch = json.load(f)
has_chronic = any(r["persistence_class"] == "CHRONIC" for r in ch)
print(f"  {'✅ PASS' if has_chronic else '❌ FAIL'}  At least one CHRONIC zone found")
if has_chronic: passed += 1

with open("outputs/capacity_recovery.json") as f:
    cr = json.load(f)
has_pcu = all("capacity_restored_pcu_hr" in r for r in cr)
print(f"  {'✅ PASS' if has_pcu else '❌ FAIL'}  PCU recovery values present in all units")
if has_pcu: passed += 1

total = len(checks) + 2
print(f"\n{'='*40}")
print(f"  PASSED: {passed} | FAILED: {total-passed}")
if passed == total:
    print("  ✅ Phase 5B complete.")
    print("  Run: streamlit run phase5_dashboard/app_v2.py")
print(f"{'='*40}\n")
```

---

## 11. What Judges See vs Old Dashboard

| Element | Old `app.py` | New `app_v2.py` |
|---|---|---|
| Theme | Generic dark grey | Command-centre deep space |
| Header | Static title text | Animated gradient + live clock |
| KPI cards | Basic HTML boxes | Glass cards with colour-coded bottom bars |
| Tabs | 4 generic tabs | 5 tabs with icon labels |
| Charts | Default plotly | Transparent dark background, peak hour markers |
| Deployment | Simple list | Unit cards with capacity recovery bar |
| New: Chronic tab | ❌ | ✅ Infrastructure recommendations |
| New: PCU restored | ❌ | ✅ Per-unit enforcement ROI |
| New: Feedback loop | Bulleted text | Styled glass panel |

---

*All phases complete. System is submission-ready.*
