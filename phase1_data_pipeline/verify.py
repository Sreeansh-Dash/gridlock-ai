"""
Run this as a standalone check: python phase1_data_pipeline/verify.py
"""
import pandas as pd
import os

CHECKS_PASSED = 0
CHECKS_FAILED = 0

def check(condition, label):
    global CHECKS_PASSED, CHECKS_FAILED
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"  {status}  {label}")
    if condition:
        CHECKS_PASSED += 1
    else:
        CHECKS_FAILED += 1

print("\n=== PHASE 1 VERIFICATION ===\n")

# File existence
check(os.path.exists("data/processed/cleaned_violations.csv"),   "cleaned_violations.csv exists")
check(os.path.exists("data/processed/impact_scored.csv"),        "impact_scored.csv exists")
check(os.path.exists("data/processed/hex_aggregated.csv"),       "hex_aggregated.csv exists")
check(os.path.exists("outputs/heatmap.html"),                    "heatmap.html exists")

# Cleaned data checks
clean = pd.read_csv("data/processed/cleaned_violations.csv")
check(len(clean) > 1000,                                         f"Cleaned rows > 1000 (got {len(clean):,})")
check("latitude" in clean.columns,                               "latitude column present")
check("longitude" in clean.columns,                              "longitude column present")
check("hour" in clean.columns,                                   "hour feature present")
check("pcu_weight" in clean.columns,                             "pcu_weight present")
check(clean["latitude"].between(12.7, 13.25).all(),               "All latitudes in Bengaluru range")
check(clean["pcu_weight"].notna().all(),                         "No null PCU weights")

# Impact scored checks
impact = pd.read_csv("data/processed/impact_scored.csv")
check("impact_score_norm" in impact.columns,                     "impact_score_norm column present")
check(impact["impact_score_norm"].between(0, 100).all(),         "All impact scores in 0-100 range")
check("hex_id" in impact.columns,                                "hex_id column present")
check(impact["hex_id"].notna().all(),                            "No null hex IDs")

# Aggregated checks
agg = pd.read_csv("data/processed/hex_aggregated.csv")
check(len(agg) > 50,                                             f"Aggregated hexes > 50 (got {len(agg)})")
check("priority_score" in agg.columns,                           "priority_score column present")
check(agg["priority_score"].max() <= 100,                        "Priority scores bounded correctly")

print(f"\n{'='*40}")
print(f"  PASSED: {CHECKS_PASSED} | FAILED: {CHECKS_FAILED}")
if CHECKS_FAILED == 0:
    print("  ✅ Phase 1 complete. Safe to proceed to Phase 2.")
else:
    print("  ❌ Fix failures above before proceeding.")
print(f"{'='*40}\n")
