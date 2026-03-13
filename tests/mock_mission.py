#!/usr/bin/env python3
"""Mock mission: write a fake telemetry CSV and exit 0. Used for runner tests."""
import argparse, csv, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--port',       type=int, default=14550)
ap.add_argument('--output-dir', type=str, default='/tmp')
args = ap.parse_args()

out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
csv_path = out / f'flight_{int(time.time())}.csv'
with csv_path.open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['time_s', 'lat_deg', 'lon_deg', 'alt_agl_m',
                'throttle_pct', 'airspeed_ms'])
    for i in range(5):
        w.writerow([i * 10, 47.05, 6.95, 50 + i, 70, 15])
print(f'[mock] wrote {csv_path}')
