"""
feature_engineering.py
Phase 2 - Step 1: Build training dataset from impact_scored.csv
Input:  data/processed/impact_scored.csv
        data/processed/hex_aggregated.csv
Output: data/processed/training_features.csv
"""

import pandas as pd
import numpy as np
import os

IMPACT_PATH = "data/processed/impact_scored.csv"
HEX_PATH    = "data/processed/hex_aggregated.csv"
OUT_PATH    = "data/processed/training_features.csv"


def build_training_set():
    print("[1/3] Loading data...")
    df  = pd.read_csv(IMPACT_PATH, low_memory=False)
    agg = pd.read_csv(HEX_PATH)

    print(f"      impact_scored rows : {len(df):,}")
    print(f"      unique hexes        : {df['hex_id'].nunique()}")

    # ------------------------------------------------------------------ #
    # Step 1: Aggregate to (hex_id, hour, day_of_week) level
    # Target variable = number of violations in that slot historically
    # ------------------------------------------------------------------ #
    print("[2/3] Aggregating to (hex_id, hour, day_of_week)...")

    grouped = (
        df.groupby(["hex_id", "hour", "day_of_week"])
        .agg(
            violation_count = ("impact_score_norm", "count"),
            avg_impact      = ("impact_score_norm", "mean"),
            total_impact    = ("impact_score_norm", "sum"),
            peak_ratio      = ("is_peak_hour", "mean"),   # fraction of peak violations
            avg_pcu         = ("pcu_weight", "mean"),
        )
        .reset_index()
    )

    # ------------------------------------------------------------------ #
    # Step 2: Merge hex-level spatial features (lat, lon, priority_score)
    # ------------------------------------------------------------------ #
    hex_meta = agg[["hex_id", "lat", "lon", "priority_score"]].copy()
    grouped  = grouped.merge(hex_meta, on="hex_id", how="left")

    # ------------------------------------------------------------------ #
    # Step 3: Derived time features
    # ------------------------------------------------------------------ #
    grouped["is_weekend"]   = (grouped["day_of_week"] >= 5).astype(int)
    grouped["is_peak_hour"] = grouped["hour"].apply(
        lambda h: 1 if (8 <= h <= 10) or (17 <= h <= 20) else 0
    )
    grouped["hour_sin"]  = np.sin(2 * np.pi * grouped["hour"] / 24)
    grouped["hour_cos"]  = np.cos(2 * np.pi * grouped["hour"] / 24)
    grouped["dow_sin"]   = np.sin(2 * np.pi * grouped["day_of_week"] / 7)
    grouped["dow_cos"]   = np.cos(2 * np.pi * grouped["day_of_week"] / 7)

    # ------------------------------------------------------------------ #
    # Step 4: Lag features — average violations in the same hex at
    # neighbouring hours (proxy for temporal autocorrelation)
    # ------------------------------------------------------------------ #
    # Sort so shift works correctly
    grouped = grouped.sort_values(["hex_id", "day_of_week", "hour"]).reset_index(drop=True)

    grouped["violations_lag1h"] = (
        grouped.groupby(["hex_id", "day_of_week"])["violation_count"]
        .shift(1)
        .fillna(0)
    )
    grouped["violations_lag2h"] = (
        grouped.groupby(["hex_id", "day_of_week"])["violation_count"]
        .shift(2)
        .fillna(0)
    )

    print(f"      Training rows generated: {len(grouped):,}")
    print(f"      Features: {list(grouped.columns)}")

    print("[3/3] Saving training features...")
    os.makedirs("data/processed", exist_ok=True)
    grouped.to_csv(OUT_PATH, index=False)
    print(f"      Saved → {OUT_PATH}\n")
    return grouped


if __name__ == "__main__":
    df = build_training_set()
    print("=== FEATURE_ENGINEERING.PY COMPLETE ===")
    print(df.describe())
