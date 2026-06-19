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
    "PASSENGER AUTO": 2.0, # added explicit mapping
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
    "PASSENGER AUTO": 3.0,
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
            df["incident_datetime"] = pd.to_datetime(df[col], errors="coerce", utc=True)
            break

    # --- Identify violation column ---
    for col in ["violation_type", "offence_type", "violation"]:
        if col in df.columns:
            df["violation_type"] = df[col].astype(str).str.upper().str.strip()
            break

    # --- Identify vehicle type column ---
    # Use updated_vehicle_type when available
    if "updated_vehicle_type" in df.columns and "vehicle_type" in df.columns:
        df["vehicle_type_description"] = df.apply(
            lambda r: r["updated_vehicle_type"]
                if pd.notna(r["updated_vehicle_type"]) and str(r["updated_vehicle_type"]).strip() not in ["", "NULL"]
                else r["vehicle_type"],
            axis=1
        ).astype(str).str.upper().str.strip()
    else:
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
    # Extended latitude bound from 13.2 to 13.25 to include Chikkajala records
    df = df[
        (df["latitude"].between(12.7, 13.25)) &
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
    print(f"      Saved -> {OUT_PATH}  ({len(df):,} rows)\n")
    return df

if __name__ == "__main__":
    df = load_and_clean(RAW_PATH)
    print("=== CLEAN.PY COMPLETE ===")
    print(df[["latitude", "longitude", "violation_type",
               "vehicle_type_description", "hour", "pcu_weight"]].head(10))
