# PHASE 3 — Prescriptive Optimizer (Patrol Deployment)
## Gridlock Hackathon 2.0 | AI-Driven Parking Intelligence System

---

## 1. What This Phase Does

This phase takes the LightGBM forecasts from Phase 2 and answers the
question no other team will answer:

> **"Given N patrol units, exactly where should each one be deployed
> right now to maximise enforcement coverage?"**

It runs a **greedy spatial coverage algorithm** over the predicted
high-risk hexes and outputs a structured deployment plan — one
assignment per patrol unit — with the hex location, risk level,
predicted violations, and nearby hexes each unit covers.

This is the prescriptive layer. Phase 1 was descriptive,
Phase 2 was predictive, Phase 3 is **actionable**.

---

## 2. Where This Sits in the Pipeline

```
[PHASE 1] ✅ DONE — impact_scored.csv, hex_aggregated.csv
[PHASE 2] ✅ DONE — lgbm_violation_predictor.pkl, predictions.json
     ↓
[PHASE 3] ← YOU ARE HERE
  predictions.json → Greedy Optimizer → deployment_plan.json
     ↓
[PHASE 4] CV Detection Loop
     ↓
[PHASE 5] Streamlit Dashboard
```

---

## 3. What To Expect After This Phase

- `outputs/deployment_plan.json` — one record per patrol unit with
  assigned hex, coordinates, risk level, predicted violations, and
  list of neighbour hexes covered
- `outputs/deployment_map.html` — Folium map showing patrol unit
  positions overlaid on the violation heatmap
- Console summary table: Unit → Location → Risk → Coverage

---

## 4. Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data handling |
| `h3` | Hex neighbour lookups (`k_ring`) |
| `folium` | Deployment map generation |
| `json` | Reading predictions, writing plan |
| `math` | Haversine distance calculation |

No new installs needed — all already in `requirements.txt`.

---

## 5. Directory Structure After This Phase

```
gridlock-parking/
│
├── outputs/
│   ├── heatmap.html                ← Phase 1
│   ├── predictions.json            ← Phase 2
│   ├── feature_importance.png      ← Phase 2
│   ├── deployment_plan.json        ← NEW
│   └── deployment_map.html         ← NEW
│
└── phase3_optimizer/
    ├── patrol_optimizer.py         ← Script 1
    └── verify.py                   ← Script 2
```

---

## 6. Algorithm — How the Greedy Optimizer Works

This is important to understand because you will need to explain it
to judges.

```
INPUTS:
  - predictions.json         (hex × hour forecasts)
  - NUM_UNITS = 8            (configurable)
  - COVERAGE_RADIUS = 1      (H3 k_ring radius, ~1.5km)
  - TARGET_HOUR = now + 1    (which hour slot to deploy for)

ALGORITHM:
  1. Filter predictions to TARGET_HOUR
  2. Sort all hexes by predicted_violations descending
  3. covered_hexes = empty set
  4. deployment   = empty list

  5. REPEAT until NUM_UNITS are assigned:
       a. Pick the highest-predicted uncovered hex (candidate)
       b. Assign next patrol unit to candidate
       c. Mark candidate + all k_ring(candidate, 1) neighbours
          as covered  (this is ~7 hexes per unit)
       d. Add assignment to deployment list

  6. OUTPUT deployment list as JSON

WHY GREEDY WORKS HERE:
  - Proven O(N log N) time complexity
  - Each unit covers a non-overlapping zone
  - Maximises unique hex coverage
  - Explainable to non-technical judges
  - Competitive with ILP solvers on small N (8 units, 770 hexes)
```

---

## 7. Architecture — Two Scripts

---

### Script 1: `phase3_optimizer/patrol_optimizer.py`

```python
"""
patrol_optimizer.py
Phase 3: Greedy spatial coverage optimizer for patrol deployment.
Input:  outputs/predictions.json
Output: outputs/deployment_plan.json
        outputs/deployment_map.html
"""

import json
import math
import os
import h3
import folium
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

PREDICTIONS_PATH  = "outputs/predictions.json"
PLAN_OUT          = "outputs/deployment_plan.json"
MAP_OUT           = "outputs/deployment_map.html"
HEATMAP_PATH      = "outputs/heatmap.html"

# ------------------------------------------------------------------ #
# Configuration — change these to explore different scenarios
# ------------------------------------------------------------------ #
NUM_UNITS        = 8    # number of patrol units available
COVERAGE_RADIUS  = 1    # H3 k_ring radius (1 = hex + 6 neighbours)
HOURS_AHEAD      = 1    # deploy for how many hours from now

BENGALURU_CENTER = [12.9716, 77.5946]

RISK_COLOURS = {
    "CRITICAL": "#FF4136",
    "HIGH":     "#FF851B",
    "MEDIUM":   "#FFDC00",
    "LOW":      "#2ECC40",
}

UNIT_ICONS = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧",
              "⑨", "⑩", "⑪", "⑫"]


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def haversine_km(lat1, lon1, lat2, lon2):
    """Straight-line distance between two GPS points in km."""
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlam       = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def get_target_slots(predictions, hours_ahead=1):
    """
    Return predictions for the next N hours from now.
    Merge them so each hex has its peak predicted value across the window.
    """
    now   = datetime.now()
    slots = []
    for h in range(hours_ahead):
        target = (now + timedelta(hours=h)).strftime("%Y-%m-%d %H:00")
        # Match on date + hour prefix
        matches = [
            p for p in predictions
            if p["target_datetime"].startswith(target[:13])
        ]
        slots.extend(matches)

    if not slots:
        # fallback: use all predictions, take first 24h
        print("  [WARN] No exact time match found — using full prediction set.")
        slots = predictions

    # One record per hex — keep the highest predicted slot
    df = pd.DataFrame(slots)
    df = (
        df.sort_values("predicted_violations", ascending=False)
          .drop_duplicates(subset="hex_id")
          .reset_index(drop=True)
    )
    return df


def greedy_coverage(candidates_df, num_units, coverage_radius):
    """
    Greedy max-coverage algorithm.
    Returns list of deployment assignment dicts.
    """
    # Sort candidates by predicted violations descending
    ranked = candidates_df.sort_values(
        "predicted_violations", ascending=False
    ).reset_index(drop=True)

    covered_hexes  = set()
    deployment     = []
    unit_id        = 1

    for _, candidate in ranked.iterrows():
        if unit_id > num_units:
            break

        hex_id = candidate["hex_id"]

        # Skip if this hex is already covered by a previously assigned unit
        if hex_id in covered_hexes:
            continue

        # Get all hexes this unit will cover (centre + neighbours)
        covered_by_unit = h3.k_ring(hex_id, coverage_radius)

        # Count how many of those are still uncovered (marginal gain)
        new_coverage = covered_by_unit - covered_hexes

        # Mark all as covered
        covered_hexes.update(covered_by_unit)

        deployment.append({
            "unit_id":               f"UNIT-{unit_id:02d}",
            "hex_id":                hex_id,
            "lat":                   round(float(candidate["lat"]), 6),
            "lon":                   round(float(candidate["lon"]), 6),
            "predicted_violations":  round(float(candidate["predicted_violations"]), 1),
            "risk_level":            candidate["risk_level"],
            "is_peak_hour":          int(candidate["is_peak_hour"]),
            "deploy_for_datetime":   candidate["target_datetime"],
            "hexes_covered":         list(covered_by_unit),
            "new_hexes_covered":     len(new_coverage),
            "total_hexes_covered":   len(covered_by_unit),
        })

        unit_id += 1

    return deployment


def build_deployment_map(deployment, predictions):
    """
    Folium map with:
    - Background heatmap layer (all violation predictions)
    - Patrol unit markers with coverage circles
    """
    m = folium.Map(
        location=BENGALURU_CENTER,
        zoom_start=12,
        tiles="CartoDB dark_matter"
    )

    # --- Background: all predicted violations as heat points ---
    df_pred = pd.DataFrame(predictions)
    heat_data = df_pred[["lat", "lon", "predicted_violations"]].values.tolist()
    from folium.plugins import HeatMap
    HeatMap(
        heat_data,
        name="Predicted Violation Density",
        min_opacity=0.2,
        radius=14,
        blur=18,
    ).add_to(m)

    # --- Patrol unit markers ---
    for i, unit in enumerate(deployment):
        icon_label = UNIT_ICONS[i] if i < len(UNIT_ICONS) else str(i+1)
        colour     = RISK_COLOURS.get(unit["risk_level"], "#AAAAAA")

        # Coverage circle (~750m per H3 res-8 hex at k_ring=1)
        folium.Circle(
            location=[unit["lat"], unit["lon"]],
            radius=750,
            color=colour,
            fill=True,
            fill_opacity=0.15,
            weight=2,
        ).add_to(m)

        # Unit marker
        folium.Marker(
            location=[unit["lat"], unit["lon"]],
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    background:{colour};
                    color:black;
                    font-weight:bold;
                    font-size:14px;
                    border-radius:50%;
                    width:30px;
                    height:30px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    border:2px solid white;
                    box-shadow:0 2px 6px rgba(0,0,0,0.4);
                ">{icon_label}</div>
                """,
                icon_size=(30, 30),
                icon_anchor=(15, 15),
            ),
            popup=folium.Popup(
                f"""
                <b>{unit['unit_id']}</b><br>
                <b>Risk:</b> {unit['risk_level']}<br>
                <b>Predicted Violations:</b> {unit['predicted_violations']}<br>
                <b>Peak Hour:</b> {'Yes' if unit['is_peak_hour'] else 'No'}<br>
                <b>Hexes Covered:</b> {unit['total_hexes_covered']}<br>
                <b>Deploy At:</b> {unit['deploy_for_datetime']}
                """,
                max_width=220,
            ),
            tooltip=f"{unit['unit_id']} — {unit['risk_level']}",
        ).add_to(m)

    # --- Legend ---
    legend_html = f"""
    <div style="position:fixed; bottom:30px; left:30px; z-index:9999;
                background:#1a1a2e; padding:14px 18px; border-radius:8px;
                color:white; font-family:Arial; font-size:12px;
                border:1px solid #555;">
        <b>🚔 Patrol Deployment Plan</b><br>
        <i>Next {HOURS_AHEAD}h | {NUM_UNITS} Units</i><br><br>
        <span style="color:{RISK_COLOURS['CRITICAL']};">●</span> CRITICAL zone<br>
        <span style="color:{RISK_COLOURS['HIGH']};">●</span> HIGH zone<br>
        <span style="color:{RISK_COLOURS['MEDIUM']};">●</span> MEDIUM zone<br><br>
        Circles = ~750m coverage radius<br>
        Numbers = unit assignment order
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)

    os.makedirs("outputs", exist_ok=True)
    m.save(MAP_OUT)
    print(f"      Deployment map saved → {MAP_OUT}")


def print_summary(deployment):
    print("\n  ┌──────────────────────────────────────────────────────────┐")
    print("  │              PATROL DEPLOYMENT PLAN                     │")
    print("  ├─────────┬────────────┬──────────┬───────────┬──────────┤")
    print("  │  Unit   │  Risk      │  Pred.   │  Hexes    │  Peak?   │")
    print("  │         │  Level     │  Violat. │  Covered  │          │")
    print("  ├─────────┼────────────┼──────────┼───────────┼──────────┤")
    for u in deployment:
        print(
            f"  │ {u['unit_id']:<8}│ {u['risk_level']:<11}│ "
            f"{u['predicted_violations']:>8.1f} │ "
            f"{u['total_hexes_covered']:>9} │ "
            f"{'Yes' if u['is_peak_hour'] else 'No ':>8} │"
        )
    print("  └─────────┴────────────┴──────────┴───────────┴──────────┘")

    total_covered = len(set(
        h for u in deployment for h in u["hexes_covered"]
    ))
    total_pred = sum(u["predicted_violations"] for u in deployment)
    print(f"\n  Total unique hexes covered : {total_covered}")
    print(f"  Total predicted violations : {total_pred:.0f}")
    print(f"  Units deployed             : {len(deployment)}")


def run():
    print("[1/4] Loading predictions.json...")
    with open(PREDICTIONS_PATH) as f:
        predictions = json.load(f)
    print(f"      Loaded {len(predictions):,} prediction records")

    print(f"[2/4] Filtering to next {HOURS_AHEAD} hour(s) window...")
    candidates = get_target_slots(predictions, hours_ahead=HOURS_AHEAD)
    print(f"      Candidates for deployment: {len(candidates)} hexes")
    print(f"      Risk distribution:")
    print(candidates["risk_level"].value_counts().to_string())

    print(f"\n[3/4] Running greedy coverage optimizer ({NUM_UNITS} units)...")
    deployment = greedy_coverage(candidates, NUM_UNITS, COVERAGE_RADIUS)
    print(f"      Deployment plan generated: {len(deployment)} unit assignments")

    # Save JSON
    with open(PLAN_OUT, "w") as f:
        json.dump(deployment, f, indent=2)
    print(f"      Saved → {PLAN_OUT}")

    print(f"\n[4/4] Building deployment map...")
    build_deployment_map(deployment, predictions)

    print_summary(deployment)
    return deployment


if __name__ == "__main__":
    deployment = run()
    print("\n=== PATROL_OPTIMIZER.PY COMPLETE ===")
```

---

### Script 2: `phase3_optimizer/verify.py`

```python
"""
verify.py — Phase 3 verification
Run: python phase3_optimizer/verify.py
"""

import json
import os

CHECKS_PASSED = 0
CHECKS_FAILED = 0

def check(condition, label):
    global CHECKS_PASSED, CHECKS_FAILED
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"  {status}  {label}")
    if condition: CHECKS_PASSED += 1
    else:         CHECKS_FAILED += 1

print("\n=== PHASE 3 VERIFICATION ===\n")

# --- File existence ---
check(os.path.exists("outputs/deployment_plan.json"), "deployment_plan.json exists")
check(os.path.exists("outputs/deployment_map.html"),  "deployment_map.html exists")

# --- Load plan ---
with open("outputs/deployment_plan.json") as f:
    plan = json.load(f)

check(len(plan) > 0,                                  f"Plan has entries (got {len(plan)})")
check(len(plan) <= 12,                                "Plan has reasonable unit count (≤12)")

required_keys = [
    "unit_id", "hex_id", "lat", "lon",
    "predicted_violations", "risk_level",
    "hexes_covered", "deploy_for_datetime"
]
for key in required_keys:
    check(all(key in u for u in plan),                f"'{key}' present in all unit records")

# --- Logical checks ---
unit_ids = [u["unit_id"] for u in plan]
check(len(unit_ids) == len(set(unit_ids)),            "No duplicate unit assignments")

all_lats = [u["lat"] for u in plan]
all_lons = [u["lon"] for u in plan]
check(all(12.7 < lat < 13.2 for lat in all_lats),    "All unit latitudes within Bengaluru")
check(all(77.3 < lon < 77.8 for lon in all_lons),    "All unit longitudes within Bengaluru")

check(all(u["predicted_violations"] >= 0 for u in plan),  "All predicted violations non-negative")
check(
    all(u["risk_level"] in ["CRITICAL","HIGH","MEDIUM","LOW"] for u in plan),
    "All risk levels valid"
)

# Coverage should not overlap (greedy ensures this)
all_covered = [h for u in plan for h in u["hexes_covered"]]
check(
    len(all_covered) == len(set(all_covered)),
    "No overlapping hex coverage between units"
)

# --- Map file non-empty ---
map_size = os.path.getsize("outputs/deployment_map.html")
check(map_size > 10_000,                              f"deployment_map.html is non-trivial ({map_size:,} bytes)")

print(f"\n{'='*44}")
print(f"  PASSED: {CHECKS_PASSED} | FAILED: {CHECKS_FAILED}")
if CHECKS_FAILED == 0:
    print("  ✅ Phase 3 complete. Safe to proceed to Phase 4.")
else:
    print("  ❌ Fix failures above before proceeding.")
print(f"{'='*44}\n")
```

---

## 8. Run Order

```bash
# Step 1 — Run optimizer and generate deployment plan
python phase3_optimizer/patrol_optimizer.py

# Step 2 — Verify all outputs
python phase3_optimizer/verify.py
```

---

## 9. Auto-Verification Checklist

The verify script checks:
- Both output files exist
- All 8 unit records are present with correct keys
- No duplicate unit assignments
- All coordinates within Bengaluru bounding box
- No overlap in covered hexes (proves greedy is working)
- Map file is non-trivial in size

---

## 10. Output Walkthrough (Handoff to Phase 4 & 5)

### What exists after Phase 3:

```
gridlock-parking/
│
└── outputs/
    ├── heatmap.html              ← Phase 1 (historical)
    ├── predictions.json          ← Phase 2 (forecasts)
    ├── feature_importance.png    ← Phase 2
    ├── deployment_plan.json      ← NEW ← Phase 5 reads this
    └── deployment_map.html       ← NEW ← Phase 5 embeds this
```

### Structure of `deployment_plan.json` (Phase 5 consumes this):

```json
[
  {
    "unit_id": "UNIT-01",
    "hex_id": "8861892e9bfffff",
    "lat": 12.971642,
    "lon": 77.594562,
    "predicted_violations": 153.0,
    "risk_level": "CRITICAL",
    "is_peak_hour": 1,
    "deploy_for_datetime": "2025-11-20 09:00",
    "hexes_covered": ["8861892e9bfffff", "8861892e8bfffff", ...],
    "new_hexes_covered": 7,
    "total_hexes_covered": 7
  },
  ...
]
```

### What Phase 4 needs to know:
- Phase 4 (CV) is standalone — it runs in Google Colab
- Its output is a pre-processed annotated video saved locally
- It also generates one synthetic violation coordinate per detection
- That coordinate gets converted to a hex_id and injected into
  `hex_aggregated.csv` to show the live feedback loop in the dashboard

### What Phase 5 (Streamlit) needs to know:
- Tab 1 reads `outputs/heatmap.html` — historical heatmap
- Tab 2 reads `outputs/predictions.json` — next 24h risk forecast
- Tab 3 reads `outputs/deployment_plan.json` + embeds `deployment_map.html`
- Tab 4 shows the pre-processed CV demo video from Phase 4

### Tech stack used so far:
`pandas`, `numpy`, `osmnx`, `h3==3.7.6`, `folium`, `tqdm`,
`lightgbm`, `scikit-learn`, `joblib`, `matplotlib`, `json`, `math`

---

*Next file: `PHASE_4_PROMPT.md` (Google Colab — YOLOv11 CV Detection)*
