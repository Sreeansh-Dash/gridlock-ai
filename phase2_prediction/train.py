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
