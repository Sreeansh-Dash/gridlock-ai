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
    return "Monitor via CV -- deploy only on event-driven triggers"


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
    print(f"\n  Saved -> {OUT_PATH}")
    return output


if __name__ == "__main__":
    result = run()
    print("\n=== CHRONIC_HOTSPOT.PY COMPLETE ===")
    print(f"Top 5 chronic zones:")
    for r in result[:5]:
        print(f"  {r['hex_id']} | score={r['chronic_score']}% | {r['persistence_class']} | {r['recommended_action'][:50]}")
