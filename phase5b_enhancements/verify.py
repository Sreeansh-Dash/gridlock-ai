"""
verify.py
Phase 5B verification — checks all outputs exist and have valid data.
"""

import os, json

checks = [
    ("outputs/chronic_hotspots.json",  "Chronic hotspots generated"),
    ("outputs/capacity_recovery.json", "Capacity recovery generated"),
    ("phase5_dashboard/app_v2.py",     "New dashboard exists"),
]

print("\n=== PHASE 5B VERIFICATION ===\n")
passed = 0
for path, label in checks:
    ok = os.path.exists(path)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if ok:
        passed += 1

# Data checks
if os.path.exists("outputs/chronic_hotspots.json"):
    with open("outputs/chronic_hotspots.json") as f:
        ch = json.load(f)
    has_chronic = any(r["persistence_class"] == "CHRONIC" for r in ch)
    print(f"  {'PASS' if has_chronic else 'FAIL'}  At least one CHRONIC zone found ({len(ch)} total hexes)")
    if has_chronic:
        passed += 1
else:
    print("  FAIL  Cannot check CHRONIC zones - file missing")

if os.path.exists("outputs/capacity_recovery.json"):
    with open("outputs/capacity_recovery.json") as f:
        cr = json.load(f)
    has_pcu = all("capacity_restored_pcu_hr" in r for r in cr)
    total_pcu = sum(r.get("capacity_restored_pcu_hr", 0) for r in cr)
    print(f"  {'PASS' if has_pcu else 'FAIL'}  PCU recovery in all {len(cr)} units (total={total_pcu:.0f} PCU/hr)")
    if has_pcu:
        passed += 1
else:
    print("  FAIL  Cannot check PCU values - file missing")

total = len(checks) + 2
print(f"\n{'='*40}")
print(f"  PASSED: {passed} | FAILED: {total - passed}")
if passed == total:
    print("  Phase 5B complete.")
    print("  Run: streamlit run phase5_dashboard/app_v2.py")
print(f"{'='*40}\n")
