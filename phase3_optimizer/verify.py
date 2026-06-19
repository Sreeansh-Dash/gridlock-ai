"""
verify.py -- Phase 3 verification
Run: python phase3_optimizer/verify.py
"""

import json
import os

CHECKS_PASSED = 0
CHECKS_FAILED = 0

def check(condition, label):
    global CHECKS_PASSED, CHECKS_FAILED
    status = "PASS" if condition else "FAIL"
    icon   = "[OK]" if condition else "[X] "
    print(f"  {icon}  {label}")
    if condition: CHECKS_PASSED += 1
    else:         CHECKS_FAILED += 1

print("\n=== PHASE 3 VERIFICATION ===\n")

# --- File existence ---
check(os.path.exists("outputs/deployment_plan.json"), "deployment_plan.json exists")
check(os.path.exists("outputs/deployment_map.html"),  "deployment_map.html exists")

# --- Load plan ---
with open("outputs/deployment_plan.json") as f:
    plan = json.load(f)

check(len(plan) > 0,                                  f"Plan has entries (got {len(plan)})")
check(len(plan) <= 12,                                "Plan has reasonable unit count (<=12)")

required_keys = [
    "unit_id", "hex_id", "lat", "lon",
    "predicted_violations", "risk_level",
    "hexes_covered", "deploy_for_datetime"
]
for key in required_keys:
    check(all(key in u for u in plan),                f"'{key}' present in all unit records")

# --- Logical checks ---
unit_ids = [u["unit_id"] for u in plan]
check(len(unit_ids) == len(set(unit_ids)),            "No duplicate unit assignments")

all_lats = [u["lat"] for u in plan]
all_lons = [u["lon"] for u in plan]
check(all(12.7 < lat < 13.2 for lat in all_lats),    "All unit latitudes within Bengaluru")
check(all(77.3 < lon < 77.8 for lon in all_lons),    "All unit longitudes within Bengaluru")

check(all(u["predicted_violations"] >= 0 for u in plan),  "All predicted violations non-negative")
check(
    all(u["risk_level"] in ["CRITICAL","HIGH","MEDIUM","LOW"] for u in plan),
    "All risk levels valid"
)

# Coverage should not overlap (greedy ensures this)
all_covered = [h for u in plan for h in u["hexes_covered"]]
check(
    len(all_covered) == len(set(all_covered)),
    "No overlapping hex coverage between units"
)

# --- Map file non-empty ---
map_size = os.path.getsize("outputs/deployment_map.html")
check(map_size > 10_000,                              f"deployment_map.html is non-trivial ({map_size:,} bytes)")

print(f"\n{'='*44}")
print(f"  PASSED: {CHECKS_PASSED} | FAILED: {CHECKS_FAILED}")
if CHECKS_FAILED == 0:
    print("  Phase 3 complete. Safe to proceed to Phase 4.")
else:
    print("  Fix failures above before proceeding.")
print(f"{'='*44}\n")
