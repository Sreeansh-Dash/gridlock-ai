# PHASE 2 — Predictive Analytics (LightGBM Violation Forecasting)
## Gridlock Hackathon 2.0 | AI-Driven Parking Intelligence System

---

## 1. What This Phase Does

This phase trains a **LightGBM model** that learns temporal and spatial patterns
from 5 months of historical violation data and forecasts violation counts per
H3 hex cell for any future time slot.

This is the core differentiator of the system.
Other teams show what happened. This predicts what **will** happen.

Specifically it:
1. Engineers a training dataset: `hex_id + hour + day_of_week → violation_count`
2. Trains and validates a LightGBM regression model
3. Generates predictions for the **next 24 hours** across all 770 hexes
4. Saves the model artifact and a structured `predictions.json`

---

## 2. Where This Sits in the Pipeline

```
[PHASE 1] ✅ DONE
  impact_scored.csv + hex_aggregated.csv
     ↓
[PHASE 2] ← YOU ARE HERE
  Feature Engineering → LightGBM Train → Predict Next 24h
     ↓
  lgbm_violation_predictor.pkl
  predictions.json
     ↓
[PHASE 3] Patrol Optimizer consumes predictions.json
     ↓
[PHASE 4] CV Detection Loop
     ↓
[PHASE 5] Streamlit Dashboard
```

---

## 3. What To Expect After This Phase

- A trained LightGBM model saved to `models/lgbm_violation_predictor.pkl`
- A label encoder saved to `models/hex_label_encoder.pkl`
- `outputs/predictions.json` — predicted violation counts per hex per hour
  for the next 24 hours, with risk levels and coordinates
- Printed model evaluation metrics: RMSE, MAE, R²
- A feature importance chart saved to `outputs/feature_importance.png`

---

## 4. Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `lightgbm` | Gradient boosted prediction model |
| `scikit-learn` | Label encoding, train/test split, metrics |
| `joblib` | Saving model artifacts |
| `matplotlib` | Feature importance chart |
| `json` | Saving predictions output |

No new installs needed — all in `requirements.txt` from Phase 1.

---

## 5. Directory Structure After This Phase

```
gridlock-parking/
│
├── data/
│   └── processed/
│       ├── cleaned_violations.csv      ← Phase 1 (untouched)
│       ├── impact_scored.csv           ← Phase 1 (untouched)
│       ├── hex_aggregated.csv          ← Phase 1 (untouched)
│       └── training_features.csv       ← NEW: engineered training set
│
├── models/
│   ├── lgbm_violation_predictor.pkl    ← NEW: trained model
│   └── hex_label_encoder.pkl           ← NEW: hex_id encoder
│
├── outputs/
│   ├── heatmap.html                    ← Phase 1 (untouched)
│   ├── predictions.json                ← NEW: next 24h forecast
│   └── feature_importance.png          ← NEW: model explainability chart
│
└── phase2_prediction/
    ├── feature_engineering.py          ← Script 1
    ├── train.py                        ← Script 2
    ├── predict.py                      ← Script 3
    └── verify.py                       ← Script 4
```

---

## 6. Architecture — Three Scripts

---

### Script 1: `phase2_prediction/feature_engineering.py`

**Job:** Aggregate `impact_scored.csv` into one row per
`(hex_id, hour, day_of_week)` combination — this is the training set.
Each row answers: "On Mondays at 9AM in hex X, how many violations occurred
historically?"

```python
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
```

---

### Script 2: `phase2_prediction/train.py`

**Job:** Train a LightGBM model on the engineered features.
Evaluate it. Save the model and encoder.

```python
"""
train.py
Phase 2 - Step 2: Train LightGBM violation count predictor.
Input:  data/processed/training_features.csv
Output: models/lgbm_violation_predictor.pkl
        models/hex_label_encoder.pkl
        outputs/feature_importance.png
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

TRAIN_PATH    = "data/processed/training_features.csv"
MODEL_OUT     = "models/lgbm_violation_predictor.pkl"
ENCODER_OUT   = "models/hex_label_encoder.pkl"
FI_CHART_OUT  = "outputs/feature_importance.png"

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

TARGET_COL = "violation_count"


def train():
    print("[1/5] Loading training features...")
    df = pd.read_csv(TRAIN_PATH)
    print(f"      Rows: {len(df):,}")

    # ------------------------------------------------------------------ #
    # Encode hex_id (string → integer for LightGBM)
    # ------------------------------------------------------------------ #
    print("[2/5] Encoding hex_id...")
    le = LabelEncoder()
    df["hex_id_encoded"] = le.fit_transform(df["hex_id"])
    os.makedirs("models", exist_ok=True)
    joblib.dump(le, ENCODER_OUT)
    print(f"      Encoder saved → {ENCODER_OUT}")
    print(f"      Unique hexes encoded: {len(le.classes_)}")

    # ------------------------------------------------------------------ #
    # Train / test split — use last 20% of data as test
    # (time-aware: sort by day_of_week then hour before splitting)
    # ------------------------------------------------------------------ #
    df_sorted = df.sort_values(["day_of_week", "hour"]).reset_index(drop=True)
    X = df_sorted[FEATURE_COLS]
    y = df_sorted[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    print(f"      Train size: {len(X_train):,} | Test size: {len(X_test):,}")

    # ------------------------------------------------------------------ #
    # LightGBM — tuned for tabular violation count data
    # ------------------------------------------------------------------ #
    print("[3/5] Training LightGBM...")
    params = {
        "objective":        "regression",
        "metric":           "rmse",
        "boosting_type":    "gbdt",
        "num_leaves":       63,
        "learning_rate":    0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "min_child_samples": 20,
        "n_estimators":     500,
        "random_state":     42,
        "verbose":          -1,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    # ------------------------------------------------------------------ #
    # Evaluate
    # ------------------------------------------------------------------ #
    print("[4/5] Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)   # no negative counts

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"\n  ┌─────────────────────────────┐")
    print(f"  │  LGBM Model Evaluation      │")
    print(f"  │  RMSE : {rmse:>8.3f}             │")
    print(f"  │  MAE  : {mae:>8.3f}             │")
    print(f"  │  R²   : {r2:>8.3f}             │")
    print(f"  └─────────────────────────────┘\n")

    # ------------------------------------------------------------------ #
    # Feature importance chart
    # ------------------------------------------------------------------ #
    os.makedirs("outputs", exist_ok=True)
    fi = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    fi = fi.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    fi.plot(kind="barh", ax=ax, color="#FF4136")
    ax.set_title("LightGBM Feature Importance — Violation Count Predictor",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.set_facecolor("#f9f9f9")
    fig.tight_layout()
    fig.savefig(FI_CHART_OUT, dpi=150)
    plt.close()
    print(f"      Feature importance chart → {FI_CHART_OUT}")

    # ------------------------------------------------------------------ #
    # Save model
    # ------------------------------------------------------------------ #
    print("[5/5] Saving model...")
    joblib.dump(model, MODEL_OUT)
    print(f"      Model saved → {MODEL_OUT}")
    return model, le


if __name__ == "__main__":
    model, le = train()
    print("\n=== TRAIN.PY COMPLETE ===")
```

---

### Script 3: `phase2_prediction/predict.py`

**Job:** Load the trained model and generate predictions for every hex
for the **next 24 hours**. Save as `predictions.json`.

```python
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
```

---

## 7. Run Order

Run from the project root in this exact order:

```bash
# Step 1 — Build training set
python phase2_prediction/feature_engineering.py

# Step 2 — Train model (takes 1-3 mins on CPU)
python phase2_prediction/train.py

# Step 3 — Generate 24h forecasts
python phase2_prediction/predict.py
```

---

## 8. Auto-Verification Checklist

Save as `phase2_prediction/verify.py` and run after all three scripts.

```python
"""
verify.py — Phase 2 verification
Run: python phase2_prediction/verify.py
"""

import pandas as pd
import numpy as np
import joblib
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

print("\n=== PHASE 2 VERIFICATION ===\n")

# --- File existence ---
check(os.path.exists("data/processed/training_features.csv"),  "training_features.csv exists")
check(os.path.exists("models/lgbm_violation_predictor.pkl"),   "LightGBM model saved")
check(os.path.exists("models/hex_label_encoder.pkl"),          "Label encoder saved")
check(os.path.exists("outputs/predictions.json"),              "predictions.json saved")
check(os.path.exists("outputs/feature_importance.png"),        "feature_importance.png saved")

# --- Training features checks ---
tf = pd.read_csv("data/processed/training_features.csv")
check(len(tf) > 100,                                           f"Training rows > 100 (got {len(tf):,})")
check("violation_count" in tf.columns,                         "violation_count target column present")
check("hex_id" in tf.columns,                                  "hex_id present")
check("hour_sin" in tf.columns,                                "Cyclical hour encoding present")
check("violations_lag1h" in tf.columns,                        "Lag features present")
check(tf["violation_count"].min() >= 0,                        "No negative violation counts")

# --- Model load check ---
try:
    model = joblib.load("models/lgbm_violation_predictor.pkl")
    le    = joblib.load("models/hex_label_encoder.pkl")
    check(True, "Model and encoder load without errors")
    check(len(le.classes_) > 50,                               f"Encoder has > 50 hex classes (got {len(le.classes_)})")
except Exception as e:
    check(False, f"Model load failed: {e}")
    check(False, "Encoder load failed")

# --- Predictions.json checks ---
with open("outputs/predictions.json") as f:
    preds = json.load(f)

check(len(preds) > 100,                                        f"Predictions count > 100 (got {len(preds):,})")
check(all("hex_id"                in p for p in preds[:10]),   "hex_id field present in predictions")
check(all("predicted_violations"  in p for p in preds[:10]),   "predicted_violations field present")
check(all("risk_level"            in p for p in preds[:10]),   "risk_level field present")
check(all("target_datetime"       in p for p in preds[:10]),   "target_datetime field present")
check(all("lat"                   in p for p in preds[:10]),   "lat field present")
check(all("lon"                   in p for p in preds[:10]),   "lon field present")
check(all(p["predicted_violations"] >= 0 for p in preds),      "All predictions non-negative")
check(
    all(p["risk_level"] in ["CRITICAL","HIGH","MEDIUM","LOW"] for p in preds),
    "Risk levels are valid strings"
)

print(f"\n{'='*42}")
print(f"  PASSED: {CHECKS_PASSED} | FAILED: {CHECKS_FAILED}")
if CHECKS_FAILED == 0:
    print("  ✅ Phase 2 complete. Safe to proceed to Phase 3.")
else:
    print("  ❌ Fix failures above before proceeding.")
print(f"{'='*42}\n")
```

Run:
```bash
python phase2_prediction/verify.py
```

---

## 9. Output Walkthrough (Handoff to Phase 3)

### What exists after Phase 2:

```
gridlock-parking/
│
├── data/
│   └── processed/
│       └── training_features.csv       ← NEW
│
├── models/
│   ├── lgbm_violation_predictor.pkl    ← NEW
│   └── hex_label_encoder.pkl           ← NEW
│
└── outputs/
    ├── heatmap.html                    ← Phase 1
    ├── predictions.json                ← NEW ← Phase 3 reads this
    └── feature_importance.png          ← NEW
```

### Structure of `predictions.json` (Phase 3 consumes this):

```json
[
  {
    "hex_id": "8854580dfffffff",
    "target_datetime": "2025-11-20 09:00",
    "hour": 9,
    "day_of_week": 2,
    "lat": 12.9716,
    "lon": 77.5946,
    "predicted_violations": 42.3,
    "risk_level": "CRITICAL",
    "is_peak_hour": 1,
    "priority_score": 87.4
  },
  ...
]
```

### What Phase 3 needs to know:
- Read `outputs/predictions.json`
- Filter to the **next time slot** (or next N hours)
- Sort by `predicted_violations` descending
- Take top-K hexes as candidate deployment zones
- Run greedy coverage optimizer across those hexes
- Use `lat` + `lon` for spatial distance calculations
- Use `risk_level` for patrol urgency labelling

### Tech stack used so far:
`pandas`, `numpy`, `osmnx`, `h3==3.7.6`, `folium`, `tqdm`,
`lightgbm`, `scikit-learn`, `joblib`, `matplotlib`

---

*Next file: `PHASE_3_PROMPT.md`*
