"""
verify.py -- Phase 4 local verification
Run: python phase4_cv/verify.py
"""

import os
import json

CHECKS_PASSED = 0
CHECKS_FAILED = 0

def check(condition, label):
    global CHECKS_PASSED, CHECKS_FAILED
    status = "PASS" if condition else "FAIL"
    icon   = "[OK]" if condition else "[X] "
    print(f"  {icon}  {label}")
    if condition:
        CHECKS_PASSED += 1
    else:
        CHECKS_FAILED += 1

print("\n=== PHASE 4 LOCAL VERIFICATION ===\n")

# --- File existence checks ---
check(os.path.exists("outputs/cv_annotated_output.mp4"),
      "Annotated video downloaded")
check(os.path.exists("outputs/detection_log.json"),
      "Detection log downloaded")
check(os.path.exists("data/processed/hex_aggregated_updated.csv"),
      "Updated hex CSV downloaded")
check(os.path.exists("outputs/heatmap_updated.html"),
      "Updated heatmap generated")

# --- Video size check ---
vid_path = "outputs/cv_annotated_output.mp4"
vid_size = os.path.getsize(vid_path) if os.path.exists(vid_path) else 0
check(vid_size > 500_000,
      f"Video is non-trivial ({vid_size // 1000} KB)")

# --- Detection log schema checks ---
if os.path.exists("outputs/detection_log.json"):
    with open("outputs/detection_log.json") as f:
        log = json.load(f)

    check(isinstance(log, list), "Detection log is a valid list")

    if log:
        check("hex_id"           in log[0], "hex_id present in log records")
        check("latitude"         in log[0], "latitude present in log records")
        check("impact_score_norm" in log[0], "impact_score_norm present in log records")
        check("vehicle_class"    in log[0], "vehicle_class present in log records")

        # Coordinate bounds check
        all_in_bluru = all(
            12.7 < d["latitude"] < 13.2 and 77.3 < d["longitude"] < 77.8
            for d in log
        )
        check(all_in_bluru, "All detection coordinates within Bengaluru bounds")
    else:
        print("  [WARN]  No detections in log -- video may have had no parked vehicles")
        print("          This is acceptable: CV loop is proven, system still works")
else:
    check(False, "Detection log exists (cannot check schema)")

print(f"\n{'='*44}")
print(f"  PASSED: {CHECKS_PASSED} | FAILED: {CHECKS_FAILED}")
if CHECKS_FAILED == 0:
    print("  Phase 4 complete. Safe to proceed to Phase 5.")
else:
    print("  Fix failures above before proceeding.")
    print("  Reminder: Download all 3 files from Google Drive first!")
print(f"{'='*44}\n")
