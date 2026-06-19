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
