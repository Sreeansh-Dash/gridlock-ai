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

# --- Toggle for skipping OSM queries ---
USE_DEFAULT_WIDTHS = True

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
    if USE_DEFAULT_WIDTHS:
        return DEFAULT_WIDTH["default"], "default", ROAD_TYPE_SF["default"]
        
    try:
        point = (lat, lon)
        G = ox.graph_from_point(point, dist=50, network_type="drive", retain_all=False)
        nearest_edge = ox.nearest_edges(G, X=lon, Y=lat) # fixed positional args issue
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
    if USE_DEFAULT_WIDTHS:
        print("[1/3] OSM queries skipped. Using default widths.")
        return {}
        
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
        if USE_DEFAULT_WIDTHS:
            w, rt, fsf = DEFAULT_WIDTH["default"], "default", ROAD_TYPE_SF["default"]
        else:
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
    print(f"      Saved -> {OUT_PATH}\n")
    return df


if __name__ == "__main__":
    df = run()
    print("=== IMPACT_SCORE.PY COMPLETE ===")
    print(df[["latitude", "longitude", "violation_type",
               "impact_score_norm", "road_type", "w_road"]].head(10))
