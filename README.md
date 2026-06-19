# 🚔 Gridlock AI: Parking Intelligence System

> **Gridlock Hackathon 2.0 — Theme 1**
> *AI-Driven Enforcement System for Bengaluru Traffic Police*

Gridlock AI is an end-to-end computer vision and predictive intelligence pipeline designed to detect, analyze, and forecast curbside parking violations. It seamlessly transforms raw traffic violation data into high-resolution spatial heatmaps, predicts future gridlocks using machine learning, allocates enforcement resources dynamically, and constantly learns from real-time dashcam computer vision analysis.

---

## 🌟 Key Features

1. **Uber H3 Geospatial Indexing**: Maps raw lat/lon coordinates to resolution-8 hexagonal grids, treating the city as discrete controllable blocks.
2. **IndoHCM Impact Scoring**: Calculates the exact capacity reduction caused by a violation based on the Passenger Car Unit (PCU) weight of the vehicle class.
3. **LightGBM Forecasting**: A predictive machine learning model that learns spatio-temporal trends to accurately forecast parking violations up to 24 hours into the future.
4. **Greedy Coverage Allocation**: An algorithmic patrol optimizer that dynamically assigns police cruisers to maximize high-risk zone coverage without overlapping radii.
5. **Real-time YOLOv11 + ByteTrack**: A computer vision loop that detects stationary vehicles in curbside regions, instantly injecting real-time violations back into the historical data stream.
6. **Unified Streamlit Dashboard**: An interactive front-end application to visualize predictions, patrol assignments, and live CV dashcam operations.

---

## 🏛️ System Architecture

The pipeline consists of 5 tightly-coupled, isolated phases:

### Phase 1: Data Pipeline
- Cleans and formats unstructured Bengaluru Traffic Police (BTP) datasets.
- Implements geocoding to resolve textual locations to specific GPS points.
- Overlays the Uber H3 grid, aggregating data to synthesize spatial `priority_scores`.

### Phase 2: LightGBM Predictor
- Uses advanced feature engineering (sin/cos temporal encodings, lagging logic).
- Trains an optimized LightGBM regression model to predict the expected number of violations per hex, per hour.

### Phase 3: Patrol Optimizer
- Reads the generated `predictions.json`.
- Dynamically generates an optimal deployment schedule mapping an N number of patrol units to high-risk H3 hex clusters.

### Phase 4: CV Detection Loop
- Operates on live or recorded dashcam/CCTV footage.
- Implements Ultralytics YOLOv11 to bound vehicles and ByteTrack to track unique vehicle paths.
- Flags violations (vehicles stationary in curbside regions) and seamlessly appends this new data to the core `hex_aggregated.csv` data lake.

### Phase 5: Dashboard
- An interactive React/Streamlit web interface connecting directly to the data lake and model outputs, functioning as the command center for traffic operators.

---

## 💻 Tech Stack

- **Data Processing**: Python, Pandas, NumPy
- **Spatial Analysis**: Uber H3, Folium
- **Machine Learning**: LightGBM, Scikit-learn
- **Computer Vision**: OpenCV, Ultralytics YOLOv11, ByteTrack, FFmpeg
- **Frontend / UI**: Streamlit, Plotly Express

---

## 📊 Model Metrics

The LightGBM violation forecasting model achieves the following evaluation metrics over the `22,720` record training set:
- **R² Score**: 0.924
- **MAE**: 0.881
- **RMSE**: 10.002

For deeper breakdowns on feature importance and the complete technical specifications, see the [TECH_STACK_AND_METRICS.md](./TECH_STACK_AND_METRICS.md) document.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- `ffmpeg` installed on your system PATH for H.264 video rendering

### Installation
```bash
git clone <repository-url>
cd gridlock-parking
pip install -r requirements.txt
```

### Running the System
You can run the full pipeline sequentially or jump straight to the dashboard if the models are already built:

```bash
# 1. Run the Streamlit Dashboard
streamlit run phase5_dashboard/app.py

# 2. Run the YOLOv11 Computer Vision Loop locally
python phase4_cv/run_yolo.py

# 3. Inject new YOLO detections into the heatmap
python phase4_cv/inject_detection.py
```
