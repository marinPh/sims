#!/usr/bin/env python3
"""
Post-mission visualisations for the thermal soaring simulation.

Three outputs
-------------
1. trajectory.png  — static trajectory overlaid on the probability map
2. trajectory.mp4  — animated trajectory + thermal positions
3. alt_throttle.png — altitude AGL and throttle time-series

CLI
---
    python3 soaring/data/plotter.py \
        --telemetry soaring/data/telemetry/flight_XXX.csv \
        --thermals  soaring/data/thermals/thermals_XXX.jsonl \
        --tif       soaring/data/probability_maps/val_de_ruz_may_avg.tif \
        --out-dir   soaring/data/plots/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import rasterio
from pyproj import Transformer


# ── coordinate helpers ────────────────────────────────────────────────────────

def _latlon_to_utm(lats, lons):
    """Convert arrays of lat/lon (WGS84) to UTM zone 32N (EPSG:32632)."""
    tf = Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)
    xs, ys = tf.transform(lons, lats)
    return np.asarray(xs), np.asarray(ys)


# ── load data ─────────────────────────────────────────────────────────────────

def _load_telemetry(csv_path: Path) -> dict:
    """Load telemetry CSV, return dict of numpy arrays."""
    data = np.genfromtxt(csv_path, delimiter=',', names=True)
    return {
        't':        data['time_s'],
        'lat':      data['lat_deg'],
        'lon':      data['lon_deg'],
        'alt':      data['alt_agl_m'],
        'throttle': data['throttle_pct'],
        'airspeed': data['airspeed_ms'],
    }


def _load_thermals(jsonl_path: Path) -> list[dict]:
    """Load thermals JSONL; return list of {t, thermals:[{cx,cy,R,envelope}]}."""
    frames = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def _load_tif(tif_path: Path):
    """Return (image, extent_utm) where extent = [xmin, xmax, ymin, ymax]."""
    with rasterio.open(tif_path) as src:
        img    = src.read(1).astype(float)
        bounds = src.bounds
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    return img, extent


# ── plot 1: static trajectory ─────────────────────────────────────────────────

def plot_trajectory(tel_csv: Path, tif_path: Path, out_path: Path) -> None:
    """
    Save a PNG with the UAV trajectory overlaid on the probability map.

    Trajectory line is coloured by time (viridis).
    Green = takeoff, red = landing.
    """
    tel   = _load_telemetry(tel_csv)
    img, extent = _load_tif(tif_path)

    xs, ys = _latlon_to_utm(tel['lat'], tel['lon'])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(
        img, origin='upper',
        extent=extent,
        cmap='YlOrRd', alpha=0.7,
        aspect='equal',
    )

    # trajectory coloured by time
    t_norm = (tel['t'] - tel['t'][0]) / max(tel['t'][-1] - tel['t'][0], 1)
    for i in range(len(xs) - 1):
        ax.plot(
            xs[i:i+2], ys[i:i+2],
            color=plt.cm.viridis(t_norm[i]), linewidth=1.5,
        )
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(tel['t'][0], tel['t'][-1]))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Time (s)')

    ax.plot(xs[0],  ys[0],  'go', ms=8, label='takeoff')
    ax.plot(xs[-1], ys[-1], 'ro', ms=8, label='landing')
    ax.set_xlabel('UTM Easting (m)')
    ax.set_ylabel('UTM Northing (m)')
    ax.set_title('UAV Trajectory')
    ax.legend(loc='upper left')
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'[plotter] trajectory PNG → {out_path}')


# ── plot 2: animated trajectory ───────────────────────────────────────────────

def animate_trajectory(
    tel_csv: Path,
    thermals_jsonl: Path,
    tif_path: Path,
    out_path: Path,
    fps: int = 10,
    decimate: int = 5,
) -> None:
    """
    Save an MP4 animating the UAV trajectory + thermal circles over time.

    Each frame shows:
    - cumulative trajectory tail up to current time
    - UAV position marker
    - solid circle = thermal core (R), dashed circle = downdraft (2R)
      alpha proportional to envelope
    """
    tel    = _load_telemetry(tel_csv)
    frames_th = _load_thermals(thermals_jsonl) if thermals_jsonl else []
    img, extent = _load_tif(tif_path)

    xs, ys = _latlon_to_utm(tel['lat'], tel['lon'])

    # decimate telemetry for animation frames
    idx   = np.arange(0, len(xs), decimate)
    xs_d  = xs[idx]
    ys_d  = ys[idx]
    ts_d  = tel['t'][idx]

    # build thermal lookup: for each frame find nearest thermal snapshot
    th_times = np.array([f['t'] for f in frames_th]) if frames_th else np.array([])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img, origin='upper', extent=extent,
              cmap='YlOrRd', alpha=0.7, aspect='equal')
    # Pin axes limits to the map extent so thermal circles outside the map
    # don't trigger autoscaling and shrink the view
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_autoscale_on(False)
    ax.set_xlabel('UTM Easting (m)')
    ax.set_ylabel('UTM Northing (m)')
    ax.set_title('UAV Trajectory + Thermals')

    traj_line, = ax.plot([], [], 'b-', linewidth=1.2, label='trajectory')
    uav_dot,   = ax.plot([], [], 'k^', ms=8, label='UAV')
    ax.legend(loc='upper left')

    _thermal_patches = []

    def _init():
        traj_line.set_data([], [])
        uav_dot.set_data([], [])
        return traj_line, uav_dot

    def _update(fi):
        nonlocal _thermal_patches
        for p in _thermal_patches:
            p.remove()
        _thermal_patches = []

        traj_line.set_data(xs_d[:fi+1], ys_d[:fi+1])
        uav_dot.set_data([xs_d[fi]], [ys_d[fi]])

        t_now = ts_d[fi]

        if len(th_times):
            snap_idx = int(np.argmin(np.abs(th_times - t_now)))
            for th in frames_th[snap_idx]['thermals']:
                cx, cy = th['cx'], th['cy']
                R      = th['R']
                alpha  = float(th['envelope']) * 0.6 + 0.1

                core = plt.Circle((cx, cy), R,     fill=False,
                                  color='steelblue', linewidth=1.2,
                                  alpha=alpha, linestyle='-')
                ring = plt.Circle((cx, cy), 2 * R, fill=False,
                                  color='tomato', linewidth=0.8,
                                  alpha=alpha, linestyle='--')
                ax.add_patch(core)
                ax.add_patch(ring)
                _thermal_patches.extend([core, ring])

        return [traj_line, uav_dot] + _thermal_patches

    # blit=False: avoids ghost artifacts from dynamically added/removed patches.
    # With blit=True, removed patches remain in the saved background and bleed
    # through into subsequent frames.
    ani = animation.FuncAnimation(
        fig, _update, frames=len(xs_d),
        init_func=_init, blit=False,
    )
    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(out_path), writer=writer)
    plt.close(fig)
    print(f'[plotter] animation MP4 → {out_path}')


# ── plot 3: altitude + throttle ───────────────────────────────────────────────

def plot_altitude_throttle(tel_csv: Path, out_path: Path) -> None:
    """Save a twin-axis PNG: altitude AGL (left, blue) and throttle % (right, red)."""
    tel = _load_telemetry(tel_csv)
    t   = tel['t']

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    ax1.plot(t, tel['alt'],      'b-',  linewidth=1.5, label='Altitude AGL')
    ax2.plot(t, tel['throttle'], 'r--', linewidth=1.2, label='Throttle %')

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Altitude AGL (m)', color='b')
    ax2.set_ylabel('Throttle (%)',     color='r')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(0, 105)

    lines  = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    ax1.set_title('Altitude AGL and Throttle')

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'[plotter] alt/throttle PNG → {out_path}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--telemetry', required=True, type=Path,
                        help='Telemetry CSV (flight_YYYYMMDD_HHMMSS.csv)')
    parser.add_argument('--thermals',  default=None,  type=Path,
                        help='Thermals JSONL (thermals_YYYYMMDD_HHMMSS.jsonl); '
                             'omit to skip thermal overlay')
    parser.add_argument('--tif',       required=True, type=Path,
                        help='Probability map GeoTIFF')
    parser.add_argument('--out-dir',   required=True, type=Path,
                        help='Directory for output files')
    args = parser.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    plot_trajectory(
        tel_csv  = args.telemetry,
        tif_path = args.tif,
        out_path = out / 'trajectory.png',
    )

    if args.thermals:
        animate_trajectory(
            tel_csv        = args.telemetry,
            thermals_jsonl = args.thermals,
            tif_path       = args.tif,
            out_path       = out / 'trajectory.mp4',
        )
    else:
        print('[plotter] --thermals not provided; skipping animation')

    plot_altitude_throttle(
        tel_csv  = args.telemetry,
        out_path = out / 'alt_throttle.png',
    )


if __name__ == '__main__':
    main()
