"""
verify.py -- Phase 5 pre-launch verification
Run: python phase5_dashboard/verify.py
"""

import os
import json
import pandas as pd

CHECKS_PASSED = 0
CHECKS_FAILED = 0

def check(condition, label):
    global CHECKS_PASSED, CHECKS_FAILED
    icon   = "[OK]" if condition else "[X] "
    print(f"  {icon}  {label}")
    if condition:
        CHECKS_PASSED += 1
    else:
        CHECKS_FAILED += 1

print("\n=== PHASE 5 PRE-LAUNCH VERIFICATION ===\n")

# --- Script existence ---
check(os.path.exists("phase5_dashboard/app.py"),           "app.py exists")

# --- Required output files ---
check(os.path.exists("outputs/heatmap_updated.html"),      "heatmap_updated.html present")
check(os.path.exists("outputs/predictions.json"),          "predictions.json present")
check(os.path.exists("outputs/deployment_plan.json"),      "deployment_plan.json present")
check(os.path.exists("outputs/deployment_map.html"),       "deployment_map.html present")
check(os.path.exists("outputs/detection_log.json"),        "detection_log.json present")
check(os.path.exists("outputs/cv_annotated_output.mp4"),   "cv_annotated_output.mp4 present")
check(os.path.exists("outputs/feature_importance.png"),    "feature_importance.png present")
check(os.path.exists("data/processed/hex_aggregated.csv"), "hex_aggregated.csv present")

# --- Data integrity ---
with open("outputs/predictions.json") as f:
    preds = json.load(f)
check(len(preds) > 100,    f"Predictions loaded ({len(preds):,} records)")
check("risk_level" in preds[0], "risk_level field present in predictions")

with open("outputs/deployment_plan.json") as f:
    plan = json.load(f)
check(len(plan) > 0,       f"Deployment plan loaded ({len(plan)} units)")
check("hexes_covered" in plan[0], "hexes_covered field present in deployment plan")

with open("outputs/detection_log.json") as f:
    dets = json.load(f)
check(isinstance(dets, list), "Detection log is valid list")

hex_agg = pd.read_csv("data/processed/hex_aggregated.csv")
check(len(hex_agg) >= 770, f"Hex aggregated loaded ({len(hex_agg)} hexes)")

# --- File size checks ---
check(
    os.path.getsize("outputs/heatmap_updated.html") > 20_000,
    f"Heatmap HTML non-trivial ({os.path.getsize('outputs/heatmap_updated.html')//1000}KB)",
)
check(
    os.path.getsize("outputs/deployment_map.html") > 50_000,
    f"Deployment map HTML non-trivial ({os.path.getsize('outputs/deployment_map.html')//1000}KB)",
)
check(
    os.path.getsize("outputs/cv_annotated_output.mp4") > 100_000,
    f"Annotated video non-trivial ({os.path.getsize('outputs/cv_annotated_output.mp4')//1000}KB)",
)

# --- Import check ---
try:
    import streamlit
    import plotly
    import pandas
    check(True, f"streamlit {streamlit.__version__}, plotly {plotly.__version__} importable")
except ImportError as e:
    check(False, f"Import failed: {e}")

print(f"\n{'='*46}")
print(f"  PASSED: {CHECKS_PASSED} | FAILED: {CHECKS_FAILED}")
if CHECKS_FAILED == 0:
    print("  All checks passed.")
    print("  Run: streamlit run phase5_dashboard/app.py")
else:
    print("  Fix failures above before launching.")
print(f"{'='*46}\n")
