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
        lambda row: h3.latlng_to_cell(row["latitude"], row["longitude"], H3_RESOLUTION),
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
    print(f"      Heatmap saved -> {MAP_OUT}")


def run():
    df = pd.read_csv(IN_PATH, low_memory=False)
    print(f"Loaded {len(df):,} impact-scored violations")

    df  = assign_hex_ids(df)
    agg = aggregate_by_hex(df)

    df.to_csv(IN_PATH, index=False)   # save hex_id back into impact_scored
    agg.to_csv(HEX_OUT, index=False)
    print(f"Saved -> {HEX_OUT}")

    build_heatmap(df, agg)
    return df, agg


if __name__ == "__main__":
    df, agg = run()
    print("\n=== SPATIAL_INDEX.PY COMPLETE ===")
    print(agg.head(10))
