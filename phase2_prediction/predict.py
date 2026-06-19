"""
predict.py
Phase 2 - Step 3: Generate next-24h violation forecasts per hex.
Input:  models/lgbm_violation_predictor.pkl
        models/hex_label_encoder.pkl
        data/processed/hex_aggregated.csv
        data/processed/training_features.csv
Output: outputs/predictions.json
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime, timedelta

MODEL_PATH    = "models/lgbm_violation_predictor.pkl"
ENCODER_PATH  = "models/hex_label_encoder.pkl"
HEX_PATH      = "data/processed/hex_aggregated.csv"
TRAIN_PATH    = "data/processed/training_features.csv"
OUT_PATH      = "outputs/predictions.json"

FEATURE_COLS = [
    "hex_id_encoded",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_peak_hour",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "avg_impact",
    "total_impact",
    "peak_ratio",
    "avg_pcu",
    "lat",
    "lon",
    "priority_score",
    "violations_lag1h",
    "violations_lag2h",
]

RISK_THRESHOLDS = {
    "CRITICAL": 0.75,
    "HIGH":     0.50,
    "MEDIUM":   0.25,
    "LOW":      0.0,
}


def get_risk_level(score, max_score):
    ratio = score / max_score if max_score > 0 else 0
    if ratio >= RISK_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    elif ratio >= RISK_THRESHOLDS["HIGH"]:
        return "HIGH"
    elif ratio >= RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


def build_prediction_rows(hex_agg, train_df, le, forecast_hours=24):
    """
    For each hex × each of the next 24 hours, build one feature row.
    Temporal lag features are approximated from historical averages.
    """
    now = datetime.now()
    rows = []

    # Pre-compute hex-level historical averages for lag approximation
    lag_lookup = (
        train_df.groupby("hex_id")["violation_count"]
        .mean()
        .to_dict()
    )
    
    # Hex static features
    hex_feats = hex_agg.set_index("hex_id")[
        ["lat", "lon", "priority_score"]
    ].to_dict(orient="index")

    # Training aggregates per hex for avg_impact etc.
    hex_train_agg = (
        train_df.groupby("hex_id")
        .agg(
            avg_impact   = ("avg_impact", "mean"),
            total_impact = ("total_impact", "mean"),
            peak_ratio   = ("peak_ratio", "mean"),
            avg_pcu      = ("avg_pcu", "mean"),
        )
        .to_dict(orient="index")
    )

    known_hexes = set(le.classes_)

    for h in range(forecast_hours):
        target_time   = now + timedelta(hours=h)
        hour          = target_time.hour
        day_of_week   = target_time.weekday()
        is_weekend    = int(day_of_week >= 5)
        is_peak       = int((8 <= hour <= 10) or (17 <= hour <= 20))
        hour_sin      = np.sin(2 * np.pi * hour / 24)
        hour_cos      = np.cos(2 * np.pi * hour / 24)
        dow_sin       = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos       = np.cos(2 * np.pi * day_of_week / 7)

        for _, hex_row in hex_agg.iterrows():
            hid = hex_row["hex_id"]
            if hid not in known_hexes:
                continue

            ta  = hex_train_agg.get(hid, {})
            hf  = hex_feats.get(hid, {})
            lag = lag_lookup.get(hid, 0)

            rows.append({
                "hex_id":          hid,
                "target_datetime": target_time.strftime("%Y-%m-%d %H:%M"),
                "hour":            hour,
                "day_of_week":     day_of_week,
                "is_weekend":      is_weekend,
                "is_peak_hour":    is_peak,
                "hour_sin":        hour_sin,
                "hour_cos":        hour_cos,
                "dow_sin":         dow_sin,
                "dow_cos":         dow_cos,
                "avg_impact":      ta.get("avg_impact", 50.0),
                "total_impact":    ta.get("total_impact", 100.0),
                "peak_ratio":      ta.get("peak_ratio", 0.3),
                "avg_pcu":         ta.get("avg_pcu", 1.0),
                "lat":             hf.get("lat", hex_row["lat"]),
                "lon":             hf.get("lon", hex_row["lon"]),
                "priority_score":  hf.get("priority_score", hex_row["priority_score"]),
                "violations_lag1h": lag,
                "violations_lag2h": lag,
                "hex_id_encoded":  le.transform([hid])[0],
            })

    return pd.DataFrame(rows)


def predict():
    print("[1/4] Loading model and encoder...")
    model = joblib.load(MODEL_PATH)
    le    = joblib.load(ENCODER_PATH)
    print(f"      Model loaded: {MODEL_PATH}")

    print("[2/4] Loading hex and training data...")
    hex_agg  = pd.read_csv(HEX_PATH)
    train_df = pd.read_csv(TRAIN_PATH)

    print("[3/4] Building prediction feature matrix (770 hexes × 24 hours)...")
    pred_df = build_prediction_rows(hex_agg, train_df, le, forecast_hours=24)
    print(f"      Feature matrix shape: {pred_df.shape}")

    X_pred = pred_df[FEATURE_COLS]
    raw_preds = model.predict(X_pred)
    pred_df["predicted_violations"] = np.clip(raw_preds, 0, None).round(1)

    # Risk levels based on predicted count relative to max
    max_pred = pred_df["predicted_violations"].max()
    pred_df["risk_level"] = pred_df["predicted_violations"].apply(
        lambda x: get_risk_level(x, max_pred)
    )

    print("[4/4] Saving predictions.json...")
    output_records = pred_df[[
        "hex_id",
        "target_datetime",
        "hour",
        "day_of_week",
        "lat",
        "lon",
        "predicted_violations",
        "risk_level",
        "is_peak_hour",
        "priority_score",
    ]].to_dict(orient="records")

    os.makedirs("outputs", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output_records, f, indent=2)

    print(f"      Saved {len(output_records):,} prediction records → {OUT_PATH}")

    # Summary
    print("\n  Predictions summary (next 24h):")
    summary = pred_df.groupby("risk_level")["hex_id"].count()
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        if level in summary:
            print(f"    {level:10s} : {summary[level]} hex-hour slots")

    top5 = (
        pred_df.sort_values("predicted_violations", ascending=False)
        .head(5)[["hex_id", "target_datetime", "predicted_violations", "risk_level"]]
    )
    print(f"\n  Top 5 highest-risk forecasts:")
    print(top5.to_string(index=False))
    return pred_df


if __name__ == "__main__":
    df = predict()
    print("\n=== PREDICT.PY COMPLETE ===")
