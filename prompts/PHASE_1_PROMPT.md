# PHASE 1 — Data Pipeline & Impact Scoring
## Gridlock Hackathon 2.0 | AI-Driven Parking Intelligence System

---

## 1. What This Phase Does

This phase transforms the raw hackathon CSV into three clean, analysis-ready output files:

1. `cleaned_violations.csv` — filtered, parsed, and enriched parking violations only
2. `impact_scored.csv` — every violation row with an IndoHCM-based congestion impact score
3. `hex_aggregated.csv` — violations grouped into H3 hexagonal grid cells with total impact per hex

This is the **data foundation** for every subsequent phase. Nothing else works without this.

---

## 2. Where This Sits in the Pipeline

```
violations.csv  ← YOU ARE HERE
     ↓
[PHASE 1] Clean → Score → Hex-Aggregate
     ↓
impact_scored.csv + hex_aggregated.csv
     ↓
[PHASE 2] LightGBM Prediction Model
     ↓
[PHASE 3] Patrol Optimizer
     ↓
[PHASE 4] CV Detection Loop
     ↓
[PHASE 5] Streamlit Dashboard
```

---

## 3. What To Expect After This Phase

After running Phase 1 successfully you will have:
- A clean dataset with only parking violations (no unrelated offences)
- Every violation enriched with road width, PCU weight, peak hour flag, and impact score
- A hex-level aggregated file ready to feed LightGBM and Folium
- A working interactive heatmap HTML file you can open in any browser right now

---

## 4. Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `pandas` | latest | CSV loading, cleaning, feature engineering |
| `numpy` | latest | Numerical operations |
| `h3` | 3.7.6 | Hexagonal spatial indexing |
| `h3pandas` | latest | H3 integration with pandas DataFrames |
| `osmnx` | latest | Fetch road width from OpenStreetMap |
| `folium` | latest | Interactive heatmap generation |
| `shapely` | latest | Geometry operations |
| `tqdm` | latest | Progress bars for long operations |

Install all at once:
```bash
pip install pandas numpy h3==3.7.6 h3pandas osmnx folium shapely tqdm
```

---

## 5. Directory Structure For This Phase

Create this structure before running anything:

```
gridlock-parking/
│
├── data/
│   ├── raw/
│   │   └── violations.csv          ← place the hackathon CSV here
│   └── processed/                  ← all outputs go here (create this folder)
│
├── models/                         ← empty for now, Phase 2 fills this
├── outputs/                        ← heatmap.html goes here
│
├── phase1_data_pipeline/
│   ├── clean.py                    ← Script 1
│   ├── impact_score.py             ← Script 2
│   └── spatial_index.py            ← Script 3
│
└── requirements.txt
```

---

## 6. Architecture — Three Scripts

---

### Script 1: `phase1_data_pipeline/clean.py`

**Job:** Load raw CSV, filter parking violations, parse timestamps, assign PCU weights.

```python
"""
clean.py
Phase 1 - Step 1: Load, filter, and enrich raw violations CSV.
Input:  data/raw/violations.csv
Output: data/processed/cleaned_violations.csv
"""

import pandas as pd
import numpy as np
import os

RAW_PATH = "data/raw/violations.csv"
OUT_PATH = "data/processed/cleaned_violations.csv"

# --- PCU weights from Indo-HCM 2017 ---
PCU_MAP = {
    "CAR": 1.0,
    "MAXI-CAB": 1.0,
    "SCOOTER": 0.5,
    "MOTOR CYCLE": 0.5,
    "MOTOR CY": 0.5,       # variant in data
    "MOPED": 0.5,
    "LMV": 1.5,
    "LGV": 1.5,
    "TANKER": 3.0,
    "GOODS AUTO": 3.0,
    "PASSENGER": 2.0,
    "VAN": 2.0,
}

# Vehicle parked width in metres (used in impact formula)
WIDTH_MAP = {
    "CAR": 2.5,
    "MAXI-CAB": 2.5,
    "SCOOTER": 1.0,
    "MOTOR CYCLE": 1.0,
    "MOTOR CY": 1.0,
    "MOPED": 1.0,
    "LMV": 2.5,
    "LGV": 2.5,
    "TANKER": 3.0,
    "GOODS AUTO": 2.5,
    "PASSENGER": 3.0,
    "VAN": 2.5,
}

PARKING_VIOLATION_KEYWORDS = [
    "NO PARKING",
    "WRONG PARKING",
    "PARKING IN A MAIN ROAD",
    "PARKING",
]

def load_and_clean(path):
    print(f"[1/5] Loading CSV from {path}...")
    df = pd.read_csv(path, low_memory=False)
    print(f"      Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"      Columns: {list(df.columns)}")

    # --- Normalise column names ---
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # --- Identify datetime column ---
    # Try common names from the dataset screenshots
    for col in ["incident_datetime", "created_datetime", "date_time", "datetime"]:
        if col in df.columns:
            df["incident_datetime"] = pd.to_datetime(df[col], errors="coerce")
            break

    # --- Identify violation column ---
    for col in ["violation_type", "offence_type", "violation"]:
        if col in df.columns:
            df["violation_type"] = df[col].astype(str).str.upper().str.strip()
            break

    # --- Identify vehicle type column ---
    for col in ["vehicle_type_description", "vehicle_type", "veh_type"]:
        if col in df.columns:
            df["vehicle_type_description"] = df[col].astype(str).str.upper().str.strip()
            break

    print(f"[2/5] Filtering for parking violations only...")
    mask = df["violation_type"].str.contains(
        "|".join(PARKING_VIOLATION_KEYWORDS), na=False
    )
    df = df[mask].copy()
    print(f"      Kept {len(df):,} parking violation rows")

    # --- Drop rows with missing coordinates ---
    df = df.dropna(subset=["latitude", "longitude"])
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    # --- Sanity check: Bengaluru bounding box roughly ---
    df = df[
        (df["latitude"].between(12.7, 13.2)) &
        (df["longitude"].between(77.3, 77.8))
    ]
    print(f"      After coordinate sanity check: {len(df):,} rows")

    print(f"[3/5] Extracting temporal features...")
    df["hour"] = df["incident_datetime"].dt.hour
    df["day_of_week"] = df["incident_datetime"].dt.dayofweek   # 0=Monday
    df["month"] = df["incident_datetime"].dt.month
    df["is_peak_hour"] = df["hour"].apply(
        lambda h: 1 if (8 <= h <= 10) or (17 <= h <= 20) else 0
    )
    df["t_multiplier"] = df["is_peak_hour"].apply(lambda x: 1.5 if x == 1 else 1.0)

    print(f"[4/5] Assigning PCU weights and vehicle widths...")
    # Extract base vehicle type for lookup
    def get_base_type(vtype):
        for key in PCU_MAP:
            if key in str(vtype):
                return key
        return "CAR"  # default fallback

    df["base_vehicle_type"] = df["vehicle_type_description"].apply(get_base_type)
    df["pcu_weight"] = df["base_vehicle_type"].map(PCU_MAP).fillna(1.0)
    df["w_parked"] = df["base_vehicle_type"].map(WIDTH_MAP).fillna(2.5)

    print(f"[5/5] Saving cleaned file...")
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"      Saved → {OUT_PATH}  ({len(df):,} rows)\n")
    return df

if __name__ == "__main__":
    df = load_and_clean(RAW_PATH)
    print("=== CLEAN.PY COMPLETE ===")
    print(df[["latitude", "longitude", "violation_type",
               "vehicle_type_description", "hour", "pcu_weight"]].head(10))
```

---

### Script 2: `phase1_data_pipeline/impact_score.py`

**Job:** For each violation, fetch road width using osmnx and compute the IndoHCM impact score.

**Important:** osmnx queries OpenStreetMap in real time. Doing this for 100k+ rows will take hours. This script:
- Samples unique coordinates (deduplicated to ~500 unique road segments)
- Caches road widths in a lookup dict
- Falls back to sensible defaults if OSM has no width data

```python
"""
impact_score.py
Phase 1 - Step 2: Fetch road data and calculate IndoHCM impact scores.
Input:  data/processed/cleaned_violations.csv
Output: data/processed/impact_scored.csv
"""

import pandas as pd
import numpy as np
import osmnx as ox
import os
from tqdm import tqdm

IN_PATH  = "data/processed/cleaned_violations.csv"
OUT_PATH = "data/processed/impact_scored.csv"

# --- Side friction factor per road type (Indo-HCM Table 3.4) ---
ROAD_TYPE_SF = {
    "motorway":     0.95,
    "trunk":        0.90,
    "primary":      0.85,
    "secondary":    0.80,
    "tertiary":     0.75,
    "residential":  0.70,
    "unclassified": 0.70,
    "default":      0.75,
}

# Default road widths in metres when OSM has no data
DEFAULT_WIDTH = {
    "motorway":     14.0,
    "trunk":        10.5,
    "primary":      7.0,
    "secondary":    7.0,
    "tertiary":     5.5,
    "residential":  4.0,
    "unclassified": 4.0,
    "default":      6.0,
}

def get_nearest_road_info(lat, lon):
    """
    Query OSM for the nearest road to a coordinate.
    Returns (road_width_metres, road_type, f_sf)
    Falls back to defaults on any error.
    """
    try:
        point = (lat, lon)
        G = ox.graph_from_point(point, dist=50, network_type="drive", retain_all=False)
        nearest_edge = ox.nearest_edges(G, lon, lat)
        edge_data = G.edges[nearest_edge]

        # Try to get width from OSM
        width = edge_data.get("width", None)
        if width is not None:
            if isinstance(width, list):
                width = width[0]
            width = float(str(width).replace("m", "").strip())
        
        road_type = edge_data.get("highway", "default")
        if isinstance(road_type, list):
            road_type = road_type[0]

        if width is None or width <= 0:
            width = DEFAULT_WIDTH.get(road_type, DEFAULT_WIDTH["default"])

        f_sf = ROAD_TYPE_SF.get(road_type, ROAD_TYPE_SF["default"])
        return width, road_type, f_sf

    except Exception:
        return DEFAULT_WIDTH["default"], "default", ROAD_TYPE_SF["default"]


def build_road_cache(df, sample_size=500):
    """
    Build a coordinate → road_info cache.
    We round coordinates to 4 decimal places (~11m precision)
    to group nearby points and avoid redundant OSM queries.
    """
    print(f"[1/3] Building road data cache (sampling {sample_size} unique locations)...")
    df["lat_r"] = df["latitude"].round(4)
    df["lon_r"] = df["longitude"].round(4)

    unique_coords = df[["lat_r", "lon_r"]].drop_duplicates()
    if len(unique_coords) > sample_size:
        unique_coords = unique_coords.sample(sample_size, random_state=42)

    cache = {}
    for _, row in tqdm(unique_coords.iterrows(), total=len(unique_coords),
                       desc="Fetching road widths from OSM"):
        key = (row["lat_r"], row["lon_r"])
        cache[key] = get_nearest_road_info(row["lat_r"], row["lon_r"])

    print(f"      Cache built: {len(cache)} unique road segments queried")
    return cache


def calculate_impact(df, cache):
    """
    Apply IndoHCM formula:
    Impact_Score = (W_parked / W_road) × f_sf × PCU_vehicle × T_multiplier
    """
    print(f"[2/3] Calculating IndoHCM impact scores...")
    
    road_widths = []
    road_types  = []
    f_sfs       = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring violations"):
        key = (round(row["latitude"], 4), round(row["longitude"], 4))
        if key in cache:
            w, rt, fsf = cache[key]
        else:
            # Nearest cached key fallback
            w, rt, fsf = DEFAULT_WIDTH["default"], "default", ROAD_TYPE_SF["default"]
        road_widths.append(w)
        road_types.append(rt)
        f_sfs.append(fsf)

    df["w_road"]    = road_widths
    df["road_type"] = road_types
    df["f_sf"]      = f_sfs

    # Clamp: parked width should never exceed road width
    df["w_parked"]  = df[["w_parked", "w_road"]].min(axis=1)

    df["impact_score"] = (
        (df["w_parked"] / df["w_road"]) *
        df["f_sf"] *
        df["pcu_weight"] *
        df["t_multiplier"]
    )

    # Normalise to 0-100 scale for interpretability
    max_score = df["impact_score"].max()
    df["impact_score_norm"] = (df["impact_score"] / max_score * 100).round(2)

    print(f"      Impact score stats:")
    print(df["impact_score_norm"].describe().round(2))
    return df


def run():
    df = pd.read_csv(IN_PATH, low_memory=False)
    print(f"Loaded {len(df):,} cleaned violations")

    cache = build_road_cache(df, sample_size=500)
    df    = calculate_impact(df, cache)

    print(f"[3/3] Saving impact_scored.csv...")
    df.to_csv(OUT_PATH, index=False)
    print(f"      Saved → {OUT_PATH}\n")
    return df


if __name__ == "__main__":
    df = run()
    print("=== IMPACT_SCORE.PY COMPLETE ===")
    print(df[["latitude", "longitude", "violation_type",
               "impact_score_norm", "road_type", "w_road"]].head(10))
```

---

### Script 3: `phase1_data_pipeline/spatial_index.py`

**Job:** Convert all violation coordinates into H3 hex IDs, aggregate by hex, and generate the Folium heatmap.

```python
"""
spatial_index.py
Phase 1 - Step 3: H3 hex indexing, aggregation, and Folium heatmap.
Input:  data/processed/impact_scored.csv
Output: data/processed/hex_aggregated.csv
        outputs/heatmap.html
"""

import pandas as pd
import numpy as np
import h3
import folium
from folium.plugins import HeatMap
import os
from tqdm import tqdm

IN_PATH      = "data/processed/impact_scored.csv"
HEX_OUT      = "data/processed/hex_aggregated.csv"
MAP_OUT      = "outputs/heatmap.html"
H3_RESOLUTION = 8   # ~0.7 km² per hex — good for city-block level

BENGALURU_CENTER = [12.9716, 77.5946]


def assign_hex_ids(df):
    print(f"[1/3] Assigning H3 hex IDs at resolution {H3_RESOLUTION}...")
    tqdm.pandas(desc="H3 indexing")
    df["hex_id"] = df.progress_apply(
        lambda row: h3.geo_to_h3(row["latitude"], row["longitude"], H3_RESOLUTION),
        axis=1
    )
    print(f"      Unique hexes: {df['hex_id'].nunique():,}")
    return df


def aggregate_by_hex(df):
    print(f"[2/3] Aggregating violations by hex...")

    agg = df.groupby("hex_id").agg(
        violation_count   = ("impact_score_norm", "count"),
        total_impact      = ("impact_score_norm", "sum"),
        avg_impact        = ("impact_score_norm", "mean"),
        peak_hour_count   = ("is_peak_hour", "sum"),
        lat               = ("latitude", "mean"),
        lon               = ("longitude", "mean"),
    ).reset_index()

    # Enforcement priority score:
    # 40% weight on volume, 60% weight on average impact
    agg["priority_score"] = (
        0.4 * (agg["violation_count"] / agg["violation_count"].max() * 100) +
        0.6 * (agg["avg_impact"])
    ).round(2)

    agg = agg.sort_values("priority_score", ascending=False)
    print(f"      Aggregated into {len(agg):,} hexes")
    print(f"      Top 5 hexes by priority:")
    print(agg[["hex_id", "violation_count", "avg_impact", "priority_score"]].head())
    return agg


def build_heatmap(df_raw, agg):
    print(f"[3/3] Building Folium heatmap...")

    m = folium.Map(
        location=BENGALURU_CENTER,
        zoom_start=12,
        tiles="CartoDB dark_matter"
    )

    # --- Layer 1: Raw violation heatmap ---
    heat_data = df_raw[["latitude", "longitude", "impact_score_norm"]].values.tolist()
    HeatMap(
        heat_data,
        name="Violation Density",
        min_opacity=0.3,
        radius=12,
        blur=15,
        max_zoom=13,
    ).add_to(m)

    # --- Layer 2: Top priority hex markers ---
    top_hexes = agg.head(20)
    for _, row in top_hexes.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=10,
            color="#FF4136",
            fill=True,
            fill_color="#FF4136",
            fill_opacity=0.7,
            popup=folium.Popup(
                f"""
                <b>Hex ID:</b> {row['hex_id']}<br>
                <b>Violations:</b> {int(row['violation_count'])}<br>
                <b>Avg Impact Score:</b> {row['avg_impact']:.1f}/100<br>
                <b>Priority Score:</b> {row['priority_score']:.1f}/100<br>
                <b>Peak Hour Violations:</b> {int(row['peak_hour_count'])}
                """,
                max_width=200
            ),
            tooltip=f"Priority: {row['priority_score']:.1f}"
        ).add_to(m)

    folium.LayerControl().add_to(m)

    # --- Legend ---
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:9999;
                background:#1a1a2e; padding:12px 16px; border-radius:8px;
                color:white; font-family:Arial; font-size:13px; border:1px solid #444;">
        <b>🚨 Enforcement Priority Zones</b><br><br>
        <span style="color:#FF4136;">●</span> Top 20 highest-risk hexes<br>
        <span style="background:linear-gradient(to right,blue,red);
               display:inline-block;width:80px;height:10px;"></span>
        <br>Heat = Impact Score<br><br>
        <i>IndoHCM capacity reduction model</i>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    os.makedirs("outputs", exist_ok=True)
    m.save(MAP_OUT)
    print(f"      Heatmap saved → {MAP_OUT}")


def run():
    df = pd.read_csv(IN_PATH, low_memory=False)
    print(f"Loaded {len(df):,} impact-scored violations")

    df  = assign_hex_ids(df)
    agg = aggregate_by_hex(df)

    df.to_csv(IN_PATH, index=False)   # save hex_id back into impact_scored
    agg.to_csv(HEX_OUT, index=False)
    print(f"Saved → {HEX_OUT}")

    build_heatmap(df, agg)
    return df, agg


if __name__ == "__main__":
    df, agg = run()
    print("\n=== SPATIAL_INDEX.PY COMPLETE ===")
    print(agg.head(10))
```

---

### `requirements.txt` (create this at project root)

```
pandas
numpy
h3==3.7.6
h3pandas
osmnx
folium
shapely
tqdm
lightgbm
scikit-learn
streamlit
streamlit-folium
ultralytics
opencv-python
```

---

## 7. Run Order

Run scripts **in this exact order** from the project root directory:

```bash
# Step 1
python phase1_data_pipeline/clean.py

# Step 2 (takes 5-10 mins due to OSM queries)
python phase1_data_pipeline/impact_score.py

# Step 3
python phase1_data_pipeline/spatial_index.py
```

---

## 8. Auto-Verification Checklist

After running all three scripts, verify the following. If any check fails, the pipeline has a bug — do not proceed to Phase 2.

```python
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
check(clean["latitude"].between(12.7, 13.2).all(),               "All latitudes in Bengaluru range")
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
```

Run it:
```bash
python phase1_data_pipeline/verify.py
```

---

## 9. Output Walkthrough (Handoff to Phase 2)

### What exists after Phase 1:

```
gridlock-parking/
│
├── data/
│   ├── raw/
│   │   └── violations.csv                  ← original, untouched
│   └── processed/
│       ├── cleaned_violations.csv          ← NEW: filtered parking rows only
│       ├── impact_scored.csv               ← NEW: + impact_score_norm + hex_id
│       └── hex_aggregated.csv             ← NEW: one row per hex
│
├── outputs/
│   └── heatmap.html                        ← NEW: open in browser to verify
│
└── phase1_data_pipeline/
    ├── clean.py                            ← done
    ├── impact_score.py                     ← done
    ├── spatial_index.py                    ← done
    └── verify.py                           ← done
```

### Key columns Phase 2 will consume from `impact_scored.csv`:

| Column | Type | Description |
|---|---|---|
| `hex_id` | string | H3 hex identifier at resolution 8 |
| `hour` | int | 0–23 |
| `day_of_week` | int | 0=Monday, 6=Sunday |
| `month` | int | 1–5 |
| `pcu_weight` | float | vehicle PCU from Indo-HCM |
| `impact_score_norm` | float | 0–100, congestion impact |
| `is_peak_hour` | int | 1 if 8-10AM or 5-8PM |
| `violation_type` | string | NO PARKING / WRONG PARKING / etc |
| `latitude` | float | original coordinate |
| `longitude` | float | original coordinate |

### Key columns Phase 2 will consume from `hex_aggregated.csv`:

| Column | Type | Description |
|---|---|---|
| `hex_id` | string | H3 hex identifier |
| `violation_count` | int | total violations in hex |
| `avg_impact` | float | mean impact score in hex |
| `priority_score` | float | enforcement priority 0-100 |
| `lat` / `lon` | float | hex centroid coordinates |

### Tech stack used so far:
`pandas`, `numpy`, `osmnx`, `h3==3.7.6`, `folium`, `tqdm`, `shapely`

### What Phase 2 needs to know:
- Training data = `impact_scored.csv`
- Spatial grouping key = `hex_id`
- Target variable = violation count aggregated per `hex_id + hour + day_of_week`
- The model predicts: *given a hex, an hour, and a day — how many violations will occur?*

---

*Next file: `PHASE_2_PROMPT.md`*
