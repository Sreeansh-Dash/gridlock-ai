import cv2
import json
import numpy as np
import random
from ultralytics import YOLO
from collections import defaultdict
import imageio
from datetime import datetime
import h3 as h3lib
import os

print('Loading YOLOv11n...')
model = YOLO('yolo11n.pt')

VIDEO_IN = 'outputs/real_dashcam.mp4'
VIDEO_OUT = 'outputs/cv_annotated_output.mp4'
LOG_OUT = 'outputs/detection_log.json'

cap = cv2.VideoCapture(VIDEO_IN)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f'Processing: {width}x{height} @ {fps:.1f}fps | {total_frames} frames')

writer = imageio.get_writer(VIDEO_OUT, fps=fps, codec='libx264',
                            quality=None, bitrate='800k',
                            ffmpeg_params=['-pix_fmt', 'yuv420p'])

# ROI config
CURB_LEFT_RATIO = 0.20
CURB_RIGHT_RATIO = 0.80
STATIONARY_THRESHOLD_PX = 15
STATIONARY_FRAMES_THRESHOLD = 15

curb_left_x = int(width * CURB_LEFT_RATIO)
curb_right_x = int(width * CURB_RIGHT_RATIO)

VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
PCU_MAP = {'car': 1.0, 'motorcycle': 0.5, 'bus': 2.0, 'truck': 3.0}
W_PARKED_MAP = {'car': 2.5, 'motorcycle': 1.0, 'bus': 3.0, 'truck': 3.0}
W_ROAD_DEFAULT = 7.0
F_SF = 0.85
H3_RESOLUTION = 8

track_history = defaultdict(list)
flagged_tracks = {}
frame_idx = 0

def is_in_curb_zone(cx, frame_width):
    return cx < int(frame_width * CURB_LEFT_RATIO) or \
           cx > int(frame_width * CURB_RIGHT_RATIO)

def compute_impact(veh_class):
    w_p = W_PARKED_MAP.get(veh_class, 2.5)
    pcu = PCU_MAP.get(veh_class, 1.0)
    score = (w_p / W_ROAD_DEFAULT) * F_SF * pcu
    return round(score * 100, 1)

def synthetic_coord():
    lat = random.uniform(12.85, 13.05)
    lon = random.uniform(77.45, 77.70)
    return round(lat, 6), round(lon, 6)

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_idx += 1

    if frame_idx % 50 == 0:
        print(f'  Frame {frame_idx}/{total_frames} | Flagged: {len(flagged_tracks)}')

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (curb_left_x, height), (0, 0, 255), -1)
    cv2.rectangle(overlay, (curb_right_x, 0), (width, height), (0, 0, 255), -1)
    frame = cv2.addWeighted(overlay, 0.08, frame, 0.92, 0)
    cv2.line(frame, (curb_left_x, 0),  (curb_left_x, height),  (0, 0, 200), 1)
    cv2.line(frame, (curb_right_x, 0), (curb_right_x, height), (0, 0, 200), 1)

    results = model.track(frame, persist=True, tracker='bytetrack.yaml',
                          classes=list(VEHICLE_CLASSES.keys()), conf=0.25, verbose=False)
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, tid, cls_id in zip(boxes, track_ids, classes):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            veh_class = VEHICLE_CLASSES.get(cls_id, 'vehicle')

            track_history[tid].append((cx, cy))
            if len(track_history[tid]) > 30:
                track_history[tid].pop(0)

            is_stationary = False
            if len(track_history[tid]) >= STATIONARY_FRAMES_THRESHOLD:
                recent = track_history[tid][-STATIONARY_FRAMES_THRESHOLD:]
                xs = [p[0] for p in recent]
                ys = [p[1] for p in recent]
                is_stationary = max(max(xs)-min(xs), max(ys)-min(ys)) < STATIONARY_THRESHOLD_PX

            is_parked = is_stationary and is_in_curb_zone(cx, width)

            if is_parked and tid not in flagged_tracks:
                flagged_tracks[tid] = {'class': veh_class, 'first_frame': frame_idx}

            if is_parked:
                colour = (0, 0, 255) # BGR
                label_txt = f'PARKED {veh_class.upper()} | -{compute_impact(veh_class)}% cap'
            else:
                colour = (0, 200, 0) # BGR
                label_txt = f'{veh_class} #{tid}'

            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, label_txt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

    cv2.putText(frame, f'Frame {frame_idx} | Violations: {len(flagged_tracks)}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Write to imageio writer (needs RGB)
    writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
writer.close()
print('Video output complete.')

detection_log = []
for tid, info in flagged_tracks.items():
    lat, lon = synthetic_coord()
    veh_class = info['class']
    try: hex_id = h3lib.latlng_to_cell(lat, lon, H3_RESOLUTION)
    except: hex_id = h3lib.geo_to_h3(lat, lon, H3_RESOLUTION)
    
    detection_log.append({
        'track_id': int(tid),
        'vehicle_class': veh_class,
        'pcu_weight': PCU_MAP.get(veh_class, 1.0),
        'latitude': lat,
        'longitude': lon,
        'hex_id': hex_id,
        'impact_score_norm': compute_impact(veh_class),
        'detected_at_frame': info['first_frame'],
        'detected_at_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source': 'cv_detection'
    })

# Always inject a few if the video didn't catch enough
if len(detection_log) < 5:
    demo = [
        (12.9352, 77.6245, 'car'),
        (12.9716, 77.5946, 'motorcycle'),
        (12.9279, 77.6271, 'car'),
        (12.9850, 77.5533, 'truck'),
        (12.9121, 77.6445, 'car'),
    ]
    for i, (lat, lon, veh) in enumerate(demo[len(detection_log):]):
        try: hex_id = h3lib.latlng_to_cell(lat, lon, H3_RESOLUTION)
        except: hex_id = h3lib.geo_to_h3(lat, lon, H3_RESOLUTION)
        detection_log.append({
            'track_id': 999+i,
            'vehicle_class': veh,
            'pcu_weight': PCU_MAP[veh],
            'latitude': lat, 'longitude': lon, 'hex_id': hex_id,
            'impact_score_norm': compute_impact(veh),
            'detected_at_frame': 100,
            'detected_at_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': 'synthetic_demo'
        })

with open(LOG_OUT, 'w') as f:
    json.dump(detection_log, f, indent=2)

print('Done.')
