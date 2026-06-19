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

UNIT_ICONS = ["1", "2", "3", "4", "5", "6", "7", "8",
              "9", "10", "11", "12"]


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
        print("  [WARN] No exact time match found -- using full prediction set.")
        slots = predictions

    # One record per hex -- keep the highest predicted slot
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
        covered_by_unit = set(h3.grid_disk(hex_id, coverage_radius))

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
            "hexes_covered":         list(new_coverage),   # only non-overlapping hexes
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
            tooltip=f"{unit['unit_id']} -- {unit['risk_level']}",
        ).add_to(m)

    # --- Legend ---
    legend_html = f"""
    <div style="position:fixed; bottom:30px; left:30px; z-index:9999;
                background:#1a1a2e; padding:14px 18px; border-radius:8px;
                color:white; font-family:Arial; font-size:12px;
                border:1px solid #555;">
        <b>Patrol Deployment Plan</b><br>
        <i>Next {HOURS_AHEAD}h | {NUM_UNITS} Units</i><br><br>
        <span style="color:{RISK_COLOURS['CRITICAL']};">&#9679;</span> CRITICAL zone<br>
        <span style="color:{RISK_COLOURS['HIGH']};">&#9679;</span> HIGH zone<br>
        <span style="color:{RISK_COLOURS['MEDIUM']};">&#9679;</span> MEDIUM zone<br><br>
        Circles = ~750m coverage radius<br>
        Numbers = unit assignment order
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)

    os.makedirs("outputs", exist_ok=True)
    m.save(MAP_OUT)
    print(f"      Deployment map saved -> {MAP_OUT}")


def print_summary(deployment):
    print("\n  +----------------------------------------------------------+")
    print("  |              PATROL DEPLOYMENT PLAN                     |")
    print("  +---------+------------+----------+-----------+----------+")
    print("  |  Unit   |  Risk      |  Pred.   |  Hexes    |  Peak?   |")
    print("  |         |  Level     |  Violat. |  Covered  |          |")
    print("  +---------+------------+----------+-----------+----------+")
    for u in deployment:
        print(
            f"  | {u['unit_id']:<8}| {u['risk_level']:<11}| "
            f"{u['predicted_violations']:>8.1f} | "
            f"{u['total_hexes_covered']:>9} | "
            f"{'Yes' if u['is_peak_hour'] else 'No ':<8} |"
        )
    print("  +---------+------------+----------+-----------+----------+")

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
    print(f"      Saved -> {PLAN_OUT}")

    print(f"\n[4/4] Building deployment map...")
    build_deployment_map(deployment, predictions)

    print_summary(deployment)
    return deployment


if __name__ == "__main__":
    deployment = run()
    print("\n=== PATROL_OPTIMIZER.PY COMPLETE ===")
