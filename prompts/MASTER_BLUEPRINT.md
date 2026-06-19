# Gridlock Hackathon 2.0 — Master Blueprint
## Theme 1: AI-Driven Parking Intelligence System
**Team:** Solo | **Stack:** Python, Streamlit, YOLOv11, LightGBM, H3, osmnx

---

## What This System Does

Most teams will build a parking violation heatmap — showing *where violations happened.*

This system does something fundamentally different:

> **It predicts where violations will happen next, quantifies their traffic impact, and prescribes an optimal patrol deployment plan — all fed by a live CV detection loop.**

This shifts the submission from a **analytics tool** to an **intelligent enforcement system.**

---

## The 4-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1 — Data Intelligence                                    │
│  CSV → Clean → H3 Hex Grid → IndoHCM Impact Score              │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 2 — Predictive Analytics                                 │
│  LightGBM → hour + day + hex_id → violation count forecast      │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 3 — Prescriptive Optimizer                               │
│  N patrol units → greedy coverage → optimal deployment zones    │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 4 — Computer Vision Feedback Loop                        │
│  YOLOv11 dashcam → new detection → feeds back into Layer 1      │
└─────────────────────────────────────────────────────────────────┘
                         ↓
              STREAMLIT DASHBOARD
         (Heatmap | Predictions | Deployment)
```

---

## Full Pipeline Flow

```
violations.csv
     │
     ▼
[Phase 1] Data Pipeline
  - Parse & clean CSV
  - Filter parking violations only
  - Assign PCU weights per vehicle type
  - osmnx → fetch road width per coordinate
  - IndoHCM impact score per violation
  - H3 hex indexing (resolution 8)
  - Aggregate: violations + impact per hex
     │
     ▼
[Phase 2] Predictive Model
  - Feature engineering: hour, day_of_week, hex_id, vehicle_type
  - Train LightGBM on 5 months of data
  - Output: predicted violation count per hex per future time slot
  - Save model artifact
     │
     ▼
[Phase 3] Patrol Optimizer
  - Input: top-N high-risk hexes from Phase 2 prediction
  - Input: number of patrol units (configurable, default 8)
  - Greedy spatial coverage algorithm
  - Output: deployment plan JSON (unit → hex → time)
     │
     ▼
[Phase 4] CV Detection Loop
  - YOLOv11 on sample Indian dashcam video
  - OC-SORT tracking → flag stationary curbside vehicles
  - Each detection → synthetic violation coordinate generated
  - Coordinate injected back into Phase 1 impact pipeline
  - Heatmap updates with new detection
     │
     ▼
[Phase 5] Streamlit Dashboard
  - Tab 1: Live Heatmap (Folium/Kepler.gl)
  - Tab 2: Predictions (next 24h by hex)
  - Tab 3: Deployment Plan (map + table)
  - Tab 4: CV Demo (video feed with detections)
```

---

## Directory Structure

```
gridlock-parking/
│
├── data/
│   ├── raw/
│   │   └── violations.csv              ← hackathon dataset
│   ├── processed/
│   │   ├── cleaned_violations.csv
│   │   ├── impact_scored.csv
│   │   └── hex_aggregated.csv
│   └── sample_video/
│       └── dashcam_sample.mp4          ← Indian dashcam footage
│
├── models/
│   ├── lgbm_violation_predictor.pkl    ← trained LightGBM
│   └── yolo11n.pt                      ← YOLOv11 weights
│
├── outputs/
│   ├── heatmap.html                    ← Folium interactive map
│   ├── predictions.json                ← next 24h forecast
│   └── deployment_plan.json            ← patrol unit assignments
│
├── phase1_data_pipeline/
│   ├── clean.py
│   ├── impact_score.py
│   └── spatial_index.py
│
├── phase2_prediction/
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
│
├── phase3_optimizer/
│   └── patrol_optimizer.py
│
├── phase4_cv/
│   ├── detect_parking.py
│   └── inject_detection.py
│
├── phase5_dashboard/
│   └── app.py
│
├── requirements.txt
├── MASTER_BLUEPRINT.md                 ← this file
└── README.md
```

---

## Full Tech Stack

| Layer | Library / Tool | Purpose |
|---|---|---|
| Data | `pandas`, `numpy` | CSV processing, feature engineering |
| Geospatial | `h3-pandas`, `osmnx`, `shapely` | Hex indexing, road data, geometry |
| ML | `lightgbm`, `scikit-learn` | Violation prediction model |
| CV | `ultralytics` (YOLOv11), `opencv-python` | Parking detection |
| Tracking | `ocsort` | Multi-object tracking in dashcam video |
| Visualization | `folium`, `branca` | Interactive heatmap |
| Dashboard | `streamlit`, `streamlit-folium` | Final UI |
| Optimization | Pure Python (greedy) | Patrol deployment |
| Colab (if needed) | Google Colab + Drive | YOLOv11 fine-tuning on IDD subset |

---

## Phase Summary

| Phase | What It Builds | Key Output |
|---|---|---|
| Phase 1 | Data pipeline + impact scoring | `impact_scored.csv`, `hex_aggregated.csv` |
| Phase 2 | LightGBM prediction model | `lgbm_violation_predictor.pkl`, `predictions.json` |
| Phase 3 | Patrol deployment optimizer | `deployment_plan.json` |
| Phase 4 | YOLOv11 CV detection loop | Annotated video + live hex injection |
| Phase 5 | Streamlit dashboard | `app.py` with 4 tabs |

---

## Key Differentiators vs Other Teams

| What Most Teams Build | What This Builds |
|---|---|
| Historical heatmap | Historical + **predictive** heatmap |
| Show violations | Show violations + **congestion impact score** |
| No deployment logic | **Optimal patrol unit assignment** |
| Static notebook | **Live Streamlit dashboard** |
| No CV | **CV detection closes the data loop** |
| Arbitrary scoring | **IndoHCM-grounded mathematical model** |

---

## IndoHCM Impact Score Formula

```
Impact_Score = (W_parked / W_road) × f_sf × PCU_vehicle × T_multiplier
```

Where:
- `W_parked` = width of parked vehicle (2.5m car, 1.0m two-wheeler)
- `W_road` = road width from osmnx (OpenStreetMap)
- `f_sf` = side friction factor (0.85 for arterial, 0.70 for local road)
- `PCU_vehicle` = car=1.0, two-wheeler=0.5, LMV=1.5, tanker=3.0
- `T_multiplier` = 1.5 if 8-10AM or 5-8PM, else 1.0

---

## PCU Weights (from Indo-HCM)

| vehicle_type_description | PCU Weight |
|---|---|
| CAR / MAXI-CAB | 1.0 |
| SCOOTER / MOTOR CYCLE | 0.5 |
| LMV | 1.5 |
| TANKER / GOODS AUTO | 3.0 |
| MOPED | 0.5 |
| PASSENGER (bus/van) | 2.0 |

---

## Notes on Google Colab Usage

Phase 4 (YOLOv11) may need Colab if local GPU is unavailable.

When using Colab:
- Mount Google Drive, place `data/` folder there
- Train/run inference in Colab
- Download model weights and annotated video output back to local
- Local Streamlit dashboard loads the saved outputs

The Phase 4 prompt will include a Colab-specific setup block.

---

## How Phase Prompts Are Used

Each phase has its own `PHASE_X_PROMPT.md` file.  
Feed that file directly to your coding agent (Google Antigravity IDE with Claude/Gemini).  
Each prompt is self-contained — it tells the agent exactly what to build, what files exist already, and what to produce.

**Build order is strictly sequential:**
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5

Do not skip phases. Each phase's output is another phase's input.

---

*Next file: `PHASE_1_PROMPT.md`*
