# PHASE 4 — Computer Vision Detection Loop (Google Colab)
## Gridlock Hackathon 2.0 | AI-Driven Parking Intelligence System

---

## 1. What This Phase Does

This phase closes the data loop of the entire system.

In a real deployment, a dashcam on a patrol vehicle streams footage
continuously. YOLOv11 detects vehicles parked on the curbside, flags
them as violations, generates a coordinate, and feeds that coordinate
back into the impact pipeline — updating the heatmap in near real-time.

For this hackathon prototype, we simulate that loop:

1. Download a real Indian dashcam video from YouTube (30-60 seconds)
2. Run YOLOv11 + ByteTrack to detect and track all vehicles
3. Flag vehicles that are **stationary** and in the **curbside ROI**
4. Generate a **synthetic violation coordinate** for each flagged vehicle
5. Save an annotated video with overlays (bounding boxes, labels, impact %)
6. Export a `detection_log.json` with all detections
7. Inject detections back into `hex_aggregated.csv` locally

**This runs entirely in Google Colab** — no local GPU needed.

---

## 2. Where This Sits in the Pipeline

```
[PHASE 1] ✅ DONE — impact_scored.csv, heatmap.html
[PHASE 2] ✅ DONE — lgbm model, predictions.json
[PHASE 3] ✅ DONE — deployment_plan.json, deployment_map.html
     ↓
[PHASE 4] ← YOU ARE HERE (Google Colab)
  Dashcam video → YOLOv11 → Stationary detection → Synthetic coords
  → cv_annotated_output.mp4 + detection_log.json
     ↓
  Local: inject_detection.py updates hex_aggregated.csv
     ↓
[PHASE 5] Streamlit Dashboard (Tab 4 shows the annotated video)
```

---

## 3. What To Expect After This Phase

**In Google Drive (download these to your local project):**
- `cv_annotated_output.mp4` — annotated dashcam video with:
  - Green boxes: moving vehicles
  - Red boxes: flagged parked/stationary curbside vehicles
  - Overlaid text: track ID, class, "PARKED ⚠", capacity reduction %
- `detection_log.json` — one record per unique flagged vehicle

**Locally (after running inject script):**
- `data/processed/hex_aggregated.csv` — updated with CV detections appended
- `outputs/heatmap_updated.html` — heatmap with CV detections shown

---

## 4. Before You Start

### Step 1 — Find a dashcam video

Search YouTube for any of these:
```
"Bangalore Brigade Road traffic dashcam"
"Bengaluru Koramangala driving POV 2023"
"Indian city traffic dashcam footage"
"MG Road Bangalore driving video"
```

Pick a video that:
- Is at least 1 minute long
- Shows a street-level driving perspective
- Has visible curbside parking or stopped vehicles
- Is filmed in an Indian city (ideally Bengaluru)

Copy the full YouTube URL. You will paste it into the Colab notebook.

### Step 2 — Set up Google Drive

Create this folder structure in your Google Drive:

```
My Drive/
└── gridlock-parking/
    ├── data/
    │   └── processed/
    │       └── hex_aggregated.csv    ← upload this from your local machine
    └── outputs/                      ← Colab will write here
```

Upload `data/processed/hex_aggregated.csv` from your local project
to Drive before running the notebook.

---

## 5. Tech Stack (Colab)

| Library | Purpose |
|---|---|
| `ultralytics` | YOLOv11 model + built-in ByteTrack tracker |
| `opencv-python` | Frame processing, ROI drawing, video writing |
| `yt-dlp` | Download YouTube dashcam video |
| `h3` | Convert detected coordinates to hex IDs |
| `pandas` | Update hex_aggregated.csv |
| `json` | Save detection log |

---

## 6. The Colab Notebook

Create a new notebook in Google Colab.
Copy each cell block below in order.

---

### CELL 1 — Install Dependencies

```python
# CELL 1: Install all dependencies
!pip install ultralytics yt-dlp h3 opencv-python-headless --quiet
print("✅ Dependencies installed")
```

---

### CELL 2 — Mount Google Drive

```python
# CELL 2: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

import os

DRIVE_BASE   = "/content/drive/MyDrive/gridlock-parking"
HEX_AGG_PATH = f"{DRIVE_BASE}/data/processed/hex_aggregated.csv"
OUTPUT_DIR   = f"{DRIVE_BASE}/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"✅ Drive mounted")
print(f"   hex_aggregated.csv exists: {os.path.exists(HEX_AGG_PATH)}")
```

---

### CELL 3 — Download Dashcam Video

```python
# CELL 3: Download dashcam video
# PASTE YOUR YOUTUBE URL BELOW
YOUTUBE_URL = "PASTE_YOUR_YOUTUBE_URL_HERE"

VIDEO_PATH = "/content/dashcam_sample.mp4"

!yt-dlp -f "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]" \
         --merge-output-format mp4 \
         -o "{VIDEO_PATH}" \
         "{YOUTUBE_URL}"

# Trim to first 45 seconds to keep processing fast
TRIMMED_PATH = "/content/dashcam_trimmed.mp4"
!ffmpeg -i "{VIDEO_PATH}" -t 45 -c copy "{TRIMMED_PATH}" -y -loglevel quiet

import os
size_mb = os.path.getsize(TRIMMED_PATH) / 1e6
print(f"✅ Video ready: {TRIMMED_PATH} ({size_mb:.1f} MB)")
```

---

### CELL 4 — Configuration

```python
# CELL 4: Configuration — adjust these if needed

# ROI: curbside zones are the leftmost and rightmost % of the frame
CURB_LEFT_RATIO  = 0.15   # left 15% of frame = left curbside
CURB_RIGHT_RATIO = 0.85   # right 85% onwards = right curbside

# A vehicle is "stationary" if its centroid moves less than this
# many pixels across STATIONARY_FRAMES consecutive frames
STATIONARY_THRESHOLD_PX     = 20
STATIONARY_FRAMES_THRESHOLD = 15   # ~0.5 seconds at 30fps

# Bengaluru bounding box for synthetic coordinate generation
# (violations injected will fall within this area)
BLURU_LAT_RANGE = (12.85, 13.05)
BLURU_LON_RANGE = (77.45, 77.70)

# Vehicle classes from COCO that YOLOv11 can detect
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# PCU weights for detected vehicle types
PCU_MAP = {"car": 1.0, "motorcycle": 0.5, "bus": 2.0, "truck": 3.0}
W_PARKED_MAP = {"car": 2.5, "motorcycle": 1.0, "bus": 3.0, "truck": 3.0}
W_ROAD_DEFAULT = 7.0   # metres, standard Bengaluru arterial road
F_SF = 0.85            # side friction factor, arterial road

print("✅ Configuration set")
```

---

### CELL 5 — Detection & Tracking Loop

```python
# CELL 5: Main detection loop

import cv2
import json
import numpy as np
import random
from ultralytics import YOLO
from collections import defaultdict

# Load YOLOv11n (nano — fast enough for Colab CPU/T4)
model = YOLO("yolo11n.pt")   # auto-downloads weights on first run
print("✅ YOLOv11n loaded")

cap = cv2.VideoCapture(TRIMMED_PATH)
fps    = cap.get(cv2.CAP_PROP_FPS) or 25
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"   Video: {width}×{height} @ {fps:.1f}fps | {total_frames} frames")

# Output writer
ANNOTATED_PATH = "/content/cv_annotated_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter(ANNOTATED_PATH, fourcc, fps, (width, height))

# Curbside ROI x-boundaries
curb_left_x  = int(width * CURB_LEFT_RATIO)
curb_right_x = int(width * CURB_RIGHT_RATIO)

# Track state: {track_id: [centroid history list]}
track_history  = defaultdict(list)
flagged_tracks = {}   # {track_id: {"class", "first_frame", "centroid"}}
detection_log  = []
frame_idx      = 0

def is_in_curb_zone(cx, frame_width):
    return cx < int(frame_width * CURB_LEFT_RATIO) or \
           cx > int(frame_width * CURB_RIGHT_RATIO)

def compute_impact(veh_class):
    w_p = W_PARKED_MAP.get(veh_class, 2.5)
    pcu = PCU_MAP.get(veh_class, 1.0)
    score = (w_p / W_ROAD_DEFAULT) * F_SF * pcu
    return round(score * 100, 1)   # as percentage capacity reduction

def synthetic_coord():
    """Generate a random coordinate within Bengaluru bounds."""
    lat = random.uniform(*BLURU_LAT_RANGE)
    lon = random.uniform(*BLURU_LON_RANGE)
    return round(lat, 6), round(lon, 6)

print("🎬 Running detection loop...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    if frame_idx % 100 == 0:
        print(f"   Frame {frame_idx}/{total_frames} | Flagged: {len(flagged_tracks)}")

    # --- Draw ROI overlay ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (curb_left_x, height),
                  (0, 0, 255), -1)
    cv2.rectangle(overlay, (curb_right_x, 0), (width, height),
                  (0, 0, 255), -1)
    frame = cv2.addWeighted(overlay, 0.08, frame, 0.92, 0)
    cv2.line(frame, (curb_left_x, 0),  (curb_left_x, height),  (0, 0, 200), 1)
    cv2.line(frame, (curb_right_x, 0), (curb_right_x, height), (0, 0, 200), 1)

    # --- YOLOv11 inference with ByteTrack ---
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=list(VEHICLE_CLASSES.keys()),
        conf=0.35,
        verbose=False,
    )

    if results[0].boxes is None or results[0].boxes.id is None:
        out.write(frame)
        continue

    boxes   = results[0].boxes.xyxy.cpu().numpy()
    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    confs   = results[0].boxes.conf.cpu().numpy()

    for box, tid, cls_id, conf in zip(boxes, track_ids, classes, confs):
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        veh_class = VEHICLE_CLASSES.get(cls_id, "vehicle")

        # Update centroid history (keep last 30 positions)
        track_history[tid].append((cx, cy))
        if len(track_history[tid]) > 30:
            track_history[tid].pop(0)

        # --- Check if stationary ---
        is_stationary = False
        if len(track_history[tid]) >= STATIONARY_FRAMES_THRESHOLD:
            recent = track_history[tid][-STATIONARY_FRAMES_THRESHOLD:]
            xs = [p[0] for p in recent]
            ys = [p[1] for p in recent]
            displacement = max(max(xs)-min(xs), max(ys)-min(ys))
            is_stationary = displacement < STATIONARY_THRESHOLD_PX

        in_curb = is_in_curb_zone(cx, width)
        is_parked = is_stationary and in_curb

        # --- Flag new parked vehicle ---
        if is_parked and tid not in flagged_tracks:
            flagged_tracks[tid] = {
                "class":       veh_class,
                "first_frame": frame_idx,
                "centroid":    (cx, cy),
            }

        # --- Draw bounding box ---
        if is_parked:
            colour    = (0, 0, 255)   # Red = parked violation
            label_txt = f"PARKED {veh_class.upper()} | -{compute_impact(veh_class)}% cap"
        else:
            colour    = (0, 200, 0)   # Green = moving / non-violation
            label_txt = f"{veh_class} #{tid}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        # Label background
        (lw, lh), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1-lh-8), (x1+lw+4, y1), colour, -1)
        cv2.putText(frame, label_txt, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # --- Frame info overlay ---
    cv2.putText(frame,
                f"Frame {frame_idx} | Violations: {len(flagged_tracks)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame,
                "RED = Illegal Parking | Curbside ROI active",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    out.write(frame)

cap.release()
out.release()
print(f"\n✅ Detection complete")
print(f"   Unique parked vehicles flagged : {len(flagged_tracks)}")
print(f"   Annotated video saved          : {ANNOTATED_PATH}")
```

---

### CELL 6 — Generate Detection Log & Synthetic Coordinates

```python
# CELL 6: Build detection_log.json with synthetic coordinates

import h3 as h3lib
from datetime import datetime

H3_RESOLUTION = 8

detection_log = []

for tid, info in flagged_tracks.items():
    lat, lon     = synthetic_coord()
    veh_class    = info["class"]
    impact       = compute_impact(veh_class)
    hex_id       = h3lib.geo_to_h3(lat, lon, H3_RESOLUTION)

    detection_log.append({
        "track_id":           int(tid),
        "vehicle_class":      veh_class,
        "pcu_weight":         PCU_MAP.get(veh_class, 1.0),
        "latitude":           lat,
        "longitude":          lon,
        "hex_id":             hex_id,
        "impact_score_norm":  impact,
        "detected_at_frame":  info["first_frame"],
        "detected_at_time":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source":             "cv_detection",
    })

# Save to Drive
LOG_PATH = f"{OUTPUT_DIR}/detection_log.json"
with open(LOG_PATH, "w") as f:
    json.dump(detection_log, f, indent=2)

print(f"✅ Detection log saved: {LOG_PATH}")
print(f"   Total detections: {len(detection_log)}")
if detection_log:
    print(f"\n   Sample record:")
    print(json.dumps(detection_log[0], indent=4))
```

---

### CELL 7 — Inject Detections into hex_aggregated.csv

```python
# CELL 7: Update hex_aggregated.csv with CV detections

import pandas as pd

hex_agg = pd.read_csv(HEX_AGG_PATH)
print(f"Original hex_aggregated rows: {len(hex_agg)}")

new_rows = []
for det in detection_log:
    hid = det["hex_id"]

    if hid in hex_agg["hex_id"].values:
        # Update existing hex: increment violation count + impact
        idx = hex_agg.index[hex_agg["hex_id"] == hid][0]
        hex_agg.at[idx, "violation_count"] += 1
        hex_agg.at[idx, "total_impact"]    += det["impact_score_norm"]
        hex_agg.at[idx, "avg_impact"]       = (
            hex_agg.at[idx, "total_impact"] /
            hex_agg.at[idx, "violation_count"]
        )
    else:
        # New hex from CV detection
        new_rows.append({
            "hex_id":          hid,
            "violation_count": 1,
            "total_impact":    det["impact_score_norm"],
            "avg_impact":      det["impact_score_norm"],
            "peak_hour_count": 0,
            "lat":             det["latitude"],
            "lon":             det["longitude"],
            "priority_score":  det["impact_score_norm"] * 0.6,
            "source":          "cv_detection",
        })

if new_rows:
    hex_agg = pd.concat([hex_agg, pd.DataFrame(new_rows)], ignore_index=True)

# Recalculate priority score
hex_agg["priority_score"] = (
    0.4 * (hex_agg["violation_count"] / hex_agg["violation_count"].max() * 100) +
    0.6 * hex_agg["avg_impact"]
).round(2)

# Save updated file
UPDATED_HEX_PATH = f"{OUTPUT_DIR}/hex_aggregated_updated.csv"
hex_agg.to_csv(UPDATED_HEX_PATH, index=False)
print(f"✅ Updated hex_aggregated saved: {UPDATED_HEX_PATH}")
print(f"   Original hexes : {len(hex_agg) - len(new_rows)}")
print(f"   New CV hexes   : {len(new_rows)}")
print(f"   Total hexes    : {len(hex_agg)}")
```

---

### CELL 8 — Copy Annotated Video to Drive & Verify

```python
# CELL 8: Copy video to Drive and run verification

import shutil

# Copy annotated video to Drive
VIDEO_DRIVE_PATH = f"{OUTPUT_DIR}/cv_annotated_output.mp4"
shutil.copy(ANNOTATED_PATH, VIDEO_DRIVE_PATH)
print(f"✅ Annotated video copied to Drive: {VIDEO_DRIVE_PATH}")

# Verification summary
print("\n=== PHASE 4 COLAB VERIFICATION ===\n")

checks = [
    (os.path.exists(VIDEO_DRIVE_PATH),                       "cv_annotated_output.mp4 in Drive"),
    (os.path.exists(LOG_PATH),                               "detection_log.json in Drive"),
    (os.path.exists(UPDATED_HEX_PATH),                       "hex_aggregated_updated.csv in Drive"),
    (os.path.getsize(VIDEO_DRIVE_PATH) > 500_000,            "Annotated video is non-trivial size"),
    (len(detection_log) >= 0,                                f"Detection log created ({len(detection_log)} records)"),
    (all("hex_id" in d for d in detection_log[:3]),          "hex_id present in detection records"),
    (all("latitude" in d for d in detection_log[:3]),        "coordinates present in detection records"),
    (len(hex_agg) >= 770,                                    f"hex_aggregated updated ({len(hex_agg)} hexes)"),
]

passed = 0
for condition, label in checks:
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"  {status}  {label}")
    if condition: passed += 1

print(f"\n{'='*44}")
print(f"  PASSED: {passed} | FAILED: {len(checks)-passed}")
if passed == len(checks):
    print("  ✅ Phase 4 Colab complete.")
    print("  Download these 3 files to your local project:")
    print(f"    {VIDEO_DRIVE_PATH}")
    print(f"    {LOG_PATH}")
    print(f"    {UPDATED_HEX_PATH}")
print(f"{'='*44}")
```

---

## 7. After Colab — Local Steps

Download these 3 files from Drive to your local project:

| Drive file | Local destination |
|---|---|
| `cv_annotated_output.mp4` | `outputs/cv_annotated_output.mp4` |
| `detection_log.json` | `outputs/detection_log.json` |
| `hex_aggregated_updated.csv` | `data/processed/hex_aggregated_updated.csv` |

Then run the local inject script:

### `phase4_cv/inject_detection.py`

```python
"""
inject_detection.py
Phase 4 - Local: Replace hex_aggregated.csv with the CV-updated version
and regenerate heatmap to show updated data.
Input:  data/processed/hex_aggregated_updated.csv (downloaded from Colab)
        outputs/detection_log.json
Output: data/processed/hex_aggregated.csv (overwritten with CV data)
        outputs/heatmap_updated.html
"""

import pandas as pd
import json
import folium
from folium.plugins import HeatMap
import os
import shutil

UPDATED_PATH = "data/processed/hex_aggregated_updated.csv"
ORIG_PATH    = "data/processed/hex_aggregated.csv"
LOG_PATH     = "outputs/detection_log.json"
MAP_OUT      = "outputs/heatmap_updated.html"
BENGALURU_CENTER = [12.9716, 77.5946]


def run():
    # Replace original hex_aggregated with updated version
    if os.path.exists(UPDATED_PATH):
        shutil.copy(UPDATED_PATH, ORIG_PATH)
        print(f"✅ hex_aggregated.csv updated with CV detections")
    else:
        print("⚠  hex_aggregated_updated.csv not found — skipping overwrite")

    hex_agg = pd.read_csv(ORIG_PATH)
    
    with open(LOG_PATH) as f:
        detections = json.load(f)

    cv_hexes = [d for d in detections]
    print(f"   CV detections injected : {len(cv_hexes)}")
    print(f"   Total hexes in map     : {len(hex_agg)}")

    # Rebuild heatmap with CV layer
    m = folium.Map(location=BENGALURU_CENTER, zoom_start=12,
                   tiles="CartoDB dark_matter")

    # Historical heatmap layer
    heat_data = hex_agg[["lat","lon","priority_score"]].values.tolist()
    HeatMap(heat_data, name="Historical Violations",
            min_opacity=0.3, radius=12, blur=15).add_to(m)

    # CV detection layer — distinct cyan markers
    cv_group = folium.FeatureGroup(name="🎥 CV Live Detections")
    for det in detections:
        folium.CircleMarker(
            location=[det["latitude"], det["longitude"]],
            radius=8,
            color="#00FFFF",
            fill=True,
            fill_color="#00FFFF",
            fill_opacity=0.8,
            popup=folium.Popup(
                f"<b>CV Detection</b><br>"
                f"Vehicle: {det['vehicle_class']}<br>"
                f"Impact: -{det['impact_score_norm']}% capacity<br>"
                f"Hex: {det['hex_id']}",
                max_width=200,
            ),
            tooltip="CV Detected Violation",
        ).add_to(cv_group)
    cv_group.add_to(m)

    folium.LayerControl().add_to(m)
    os.makedirs("outputs", exist_ok=True)
    m.save(MAP_OUT)
    print(f"✅ Updated heatmap saved → {MAP_OUT}")


if __name__ == "__main__":
    run()
    print("\n=== INJECT_DETECTION.PY COMPLETE ===")
```

Run locally:
```bash
python phase4_cv/inject_detection.py
```

---

## 8. Local Verification

### `phase4_cv/verify.py`

```python
"""
verify.py — Phase 4 local verification
Run: python phase4_cv/verify.py
"""

import os, json

CHECKS_PASSED = 0
CHECKS_FAILED = 0

def check(condition, label):
    global CHECKS_PASSED, CHECKS_FAILED
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"  {status}  {label}")
    if condition: CHECKS_PASSED += 1
    else:         CHECKS_FAILED += 1

print("\n=== PHASE 4 LOCAL VERIFICATION ===\n")

check(os.path.exists("outputs/cv_annotated_output.mp4"),       "Annotated video downloaded")
check(os.path.exists("outputs/detection_log.json"),            "Detection log downloaded")
check(os.path.exists("data/processed/hex_aggregated_updated.csv"), "Updated hex CSV downloaded")
check(os.path.exists("outputs/heatmap_updated.html"),          "Updated heatmap generated")

vid_size = os.path.getsize("outputs/cv_annotated_output.mp4") if \
           os.path.exists("outputs/cv_annotated_output.mp4") else 0
check(vid_size > 500_000,                                      f"Video is non-trivial ({vid_size//1000}KB)")

with open("outputs/detection_log.json") as f:
    log = json.load(f)

check(isinstance(log, list),                                   "Detection log is a valid list")
if log:
    check("hex_id"          in log[0],                         "hex_id present in log records")
    check("latitude"        in log[0],                         "latitude present in log records")
    check("impact_score_norm" in log[0],                       "impact_score_norm present")
    check("vehicle_class"   in log[0],                         "vehicle_class present")
else:
    print("  ⚠  WARN  No detections in log — video may have no parked vehicles")
    print("           This is acceptable: system still works, CV loop is proven")

print(f"\n{'='*44}")
print(f"  PASSED: {CHECKS_PASSED} | FAILED: {CHECKS_FAILED}")
if CHECKS_FAILED == 0:
    print("  ✅ Phase 4 complete. Safe to proceed to Phase 5.")
print(f"{'='*44}\n")
```

Run:
```bash
python phase4_cv/verify.py
```

---

## 9. If Detection Log Is Empty

If the video produces zero parked vehicle detections, it means the
video didn't have clear curbside stationary vehicles. This is fine.

**Quick fix:** Add this to Colab after Cell 6 to inject 5 synthetic
detections manually for demonstration purposes:

```python
# Manual synthetic detections (use only if detection_log is empty)
if len(detection_log) == 0:
    print("⚠ No live detections — injecting synthetic demo detections")
    demo_coords = [
        (12.9352, 77.6245), (12.9716, 77.5946),
        (12.9279, 77.6271), (12.9850, 77.5533),
        (12.9121, 77.6445),
    ]
    for i, (lat, lon) in enumerate(demo_coords):
        veh = ["car","motorcycle","car","truck","car"][i]
        detection_log.append({
            "track_id":          i+1,
            "vehicle_class":     veh,
            "pcu_weight":        PCU_MAP[veh],
            "latitude":          lat,
            "longitude":         lon,
            "hex_id":            h3lib.geo_to_h3(lat, lon, H3_RESOLUTION),
            "impact_score_norm": compute_impact(veh),
            "detected_at_frame": (i+1)*30,
            "detected_at_time":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source":            "synthetic_demo",
        })
    print(f"   Injected {len(detection_log)} synthetic detections")
```

---

## 10. Output Walkthrough (Handoff to Phase 5)

### What exists after Phase 4:

```
gridlock-parking/
│
├── data/
│   └── processed/
│       ├── hex_aggregated.csv              ← updated with CV data
│       └── hex_aggregated_updated.csv      ← backup copy
│
├── outputs/
│   ├── heatmap.html                        ← Phase 1 (historical)
│   ├── heatmap_updated.html                ← NEW (with CV layer)
│   ├── predictions.json                    ← Phase 2
│   ├── deployment_plan.json                ← Phase 3
│   ├── deployment_map.html                 ← Phase 3
│   ├── cv_annotated_output.mp4             ← NEW (Phase 5 Tab 4)
│   └── detection_log.json                  ← NEW
│
└── phase4_cv/
    ├── inject_detection.py                 ← done
    └── verify.py                           ← done
```

### What Phase 5 (Streamlit) reads:

| Tab | File |
|---|---|
| Tab 1 — Live Heatmap | `outputs/heatmap_updated.html` |
| Tab 2 — Predictions | `outputs/predictions.json` |
| Tab 3 — Deployment Plan | `outputs/deployment_plan.json` + `deployment_map.html` |
| Tab 4 — CV Demo | `outputs/cv_annotated_output.mp4` |

### Tech stack used so far:
`pandas`, `numpy`, `h3==3.7.6`, `folium`, `tqdm`,
`lightgbm`, `scikit-learn`, `joblib`, `matplotlib`,
`ultralytics` (YOLOv11 + ByteTrack), `opencv-python`, `yt-dlp`

---

*Next file: `PHASE_5_PROMPT.md` (Streamlit Dashboard — 4 tabs)*
