import pandas as pd
import json
import os

LOG_PATH = 'outputs/detection_log.json'
HEX_AGG_PATH = 'data/processed/hex_aggregated.csv'

with open(LOG_PATH, 'r') as f:
    detections = json.load(f)

hex_agg = pd.read_csv(HEX_AGG_PATH)
new_rows = []

for det in detections:
    hid = det['hex_id']
    if hid in hex_agg['hex_id'].values:
        idx = hex_agg.index[hex_agg['hex_id'] == hid][0]
        hex_agg.at[idx, 'violation_count'] += 1
        # ensure total_impact column exists
        if 'total_impact' not in hex_agg.columns:
            hex_agg['total_impact'] = hex_agg['avg_impact'] * hex_agg['violation_count']
        
        hex_agg.at[idx, 'total_impact'] += det['impact_score_norm']
        hex_agg.at[idx, 'avg_impact'] = hex_agg.at[idx, 'total_impact'] / hex_agg.at[idx, 'violation_count']
    else:
        new_rows.append({
            'hex_id': hid,
            'violation_count': 1,
            'total_impact': det['impact_score_norm'],
            'avg_impact': det['impact_score_norm'],
            'peak_hour_count': 0,
            'lat': det['latitude'],
            'lon': det['longitude'],
            'priority_score': det['impact_score_norm'] * 0.6
        })

if new_rows:
    hex_agg = pd.concat([hex_agg, pd.DataFrame(new_rows)], ignore_index=True)

# Re-calculate priority score
hex_agg['priority_score'] = (
    0.4 * (hex_agg['violation_count'] / hex_agg['violation_count'].max() * 100) +
    0.6 * hex_agg['avg_impact']
).round(2)

hex_agg.to_csv(HEX_AGG_PATH, index=False)
print(f'Updated {HEX_AGG_PATH} with {len(detections)} detections.')
