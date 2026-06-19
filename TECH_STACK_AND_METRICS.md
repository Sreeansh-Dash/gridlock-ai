# Technical Specifications & Metrics — Gridlock AI

## 1. Technology Stack
- **Core Language**: Python 3.11+
- **Frontend / Dashboard**: Streamlit
- **Backend / Data Pipeline**: Pandas, NumPy
- **Geospatial Indexing**: Uber H3 (Hexagonal Hierarchical Spatial Index)
- **Mapping & Visualization**: Folium, Plotly Express
- **Machine Learning**: LightGBM (Gradient Boosting Framework)
- **Computer Vision**: Ultralytics YOLOv11, ByteTrack
- **Video Processing**: OpenCV, ImageIO, FFmpeg (libx264)
- **Model Persistence**: Joblib

## 2. System Architecture
The system operates as a continuous, 5-phase closed-loop architecture:

### Phase 1: Data Pipeline (Spatial Transformation)
- Cleans and formats raw Bengaluru traffic police violation data.
- Translates unstructured textual locations into geospatial coordinates via geocoding lookups.
- Projects GPS points into an **Uber H3 (Resolution 8)** hexagonal grid.
- Calculates an `impact_score` utilizing IndoHCM (Indian Highway Capacity Manual) formulas based on vehicle PCU (Passenger Car Unit) sizes.

### Phase 2: Predictive Modeling
- Synthesizes an hourly time-series dataset featuring temporal features (sin/cos encoding of hour/day), spatial features (lat/lon/base priority), and lag features.
- Trains a **LightGBM Regressor** to forecast violation counts per hour for the next 24 hours.

### Phase 3: Patrol Deployment Optimizer
- Implements a **Greedy Spatial Coverage Algorithm**.
- Ranks hexes by predicted violations.
- Allocates patrol units iteratively, ensuring each unit covers an H3 k-ring of 1 (a central hex + 6 immediate neighbors, approx 750m radius).
- Dynamically calculates the optimal distribution of units to maximize geographic coverage of high-risk zones without overlap.

### Phase 4: CV Detection Loop
- Processes real-time dashcam footage using **YOLOv11** for vehicle detection and **ByteTrack** for persistent cross-frame tracking.
- Identifies stationary vehicles occupying the curbside ROI (Region of Interest) as illegal parking violations.
- Injects detected violations backward into the `hex_aggregated.csv` historical data, enabling the system to "learn" from live feeds and immediately recalculate priority heatmaps.

### Phase 5: Intelligence Dashboard
- A unified **Streamlit** web application consolidating historical heatmaps, predictive hour-by-hour forecast charts, live interactive patrol deployment mapping, and video playback of CV operations.

## 3. Model Metrics & Evaluation
The LightGBM violation predictor was trained on the historical spatial data and evaluated using standard regression metrics.

- **Training Size**: 22,720 hex-hour slots
- **R² Score (Coefficient of Determination)**: 0.924
- **MAE (Mean Absolute Error)**: 0.881
- **RMSE (Root Mean Square Error)**: 10.002

### Feature Importance Highlights
1. **Historical Priority Score**: Most significant indicator of future violations.
2. **Time of Day (hour_sin/hour_cos)**: Captures peak traffic behaviors.
3. **Hex ID**: Geographic specificity.
4. **Day of Week**: Weekday vs Weekend patterns.

## 4. Directory Structure
```text
gridlock-parking/
├── data/
│   ├── raw/                    # Original BTP traffic CSVs
│   └── processed/              # H3 indexed, cleaned, and aggregated CSVs
├── models/
│   ├── lgbm_violation_predictor.pkl
│   ├── hex_label_encoder.pkl
│   └── yolo11n.pt              # YOLO model weights
├── outputs/
│   ├── heatmap_updated.html    # Interactive heatmaps
│   ├── predictions.json        # 24h forecasts
│   ├── deployment_plan.json    # Patrol assignments
│   ├── detection_log.json      # CV detected violations
│   └── cv_annotated_output.mp4 # Tracked dashcam footage
├── phase1_data_pipeline/       # Cleaning, H3 mapping, IndoHCM impact
├── phase2_prediction/          # Feature engineering, LightGBM training, and inference
├── phase3_optimizer/           # Greedy coverage allocation algorithm
├── phase4_cv/                  # YOLOv11 tracking, video processing, data injection
├── phase5_dashboard/           # Streamlit UI application
├── prompts/                    # LLM context and prompt instructions
├── README.md                   
├── TECH_STACK_AND_METRICS.md
└── .gitignore                  
```
