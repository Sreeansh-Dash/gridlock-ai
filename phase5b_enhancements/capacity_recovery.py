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
            avg_pcu_weight = ("pcu_weight",   "mean"),
            avg_w_parked   = ("w_parked",   "mean"),
            violation_count= ("impact_score_norm", "count"),
        )
        .reset_index()
    )
    hex_dict = hex_metrics.set_index("hex_id").to_dict(orient="index")

    print("[2/3] Calculating capacity recovery per unit...")
    results = []

    for unit in plan:
        unit_id       = unit["unit_id"]
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
    print(f"  {'Unit':<10} {'Restored (PCU/hr)':>18} {'Congestion Reduction':>20}")
    print(f"  {'-'*50}")
    for r in results:
        print(f"  {r['unit_id']:<10} {r['capacity_restored_pcu_hr']:>18.1f} "
              f"{r['congestion_reduction_pct']:>19.1f}%")
    print(f"  {'-'*50}")
    print(f"  {'TOTAL':<10} {total_restored:>18.1f} PCU/hr restored")
    print(f"\n  Saved -> {OUT_PATH}")
    return results


if __name__ == "__main__":
    results = run()
    print("\n=== CAPACITY_RECOVERY.PY COMPLETE ===")
