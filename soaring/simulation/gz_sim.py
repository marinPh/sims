# simulation/gz_sim.py
"""
Gazebo thermal simulation bridge.

Subscribes to UAV pose published by Gazebo, queries the ThermalField,
and publishes the resulting 3D wind vector back into the world.

Rates
-----
  field.update()          : 1 Hz   — thermal lifecycle (spawn, age, cull)
  field.query() + publish : 250 Hz — one-to-one with every pose callback

Coordinate mapping
------------------
  Gazebo world frame is ENU with origin at (0, 0).
  ThermalField works in UTM metres from the GeoTIFF origin.
  WORLD_ORIGIN_UTM must be set to the UTM coordinates of the Gazebo
  world origin so that positions align with the probability map.

Usage
-----
    cd soaring/simulation
    python3 gz_sim.py

    # verify wind injection:
    gz topic -e -t /world/default/wind
"""

import json
import time
import signal
import sys
from datetime import datetime
from pathlib import Path
import numpy as np

from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.wind_pb2 import Wind

from probability_map import load_tif
from thermal_field import ThermalField


# ── configuration ──────────────────────────────────────────────────────────────

TIFF_PATH    = '../data/probability_maps/val_de_ruz_may_avg.tif'
WORLD        = 'default'

# Must match your .sdf model name — check with: gz model --list
MODEL        = 'advanced_plane_0'

# Background (ambient) horizontal wind [m/s], (u=east, v=north)
AMBIENT_WIND = np.array([2.0, 0.5])

Z_I          = 1200.0    # convective boundary layer height [m]
SPAWN_RATE   = 1e-6      # thermal spawn density (see ThermalField docs)

# UTM coordinates (easting, northing) of the Gazebo world origin [m].
# Set this so that UAV position (Gazebo ENU) maps onto the probability map.
# Example: if your world origin sits at the south-west corner of the TIF,
# use pmap['bounds']['x_min'] and pmap['bounds']['y_min'] after load_tif().
WORLD_ORIGIN_UTM = np.array([0.0, 0.0])

FIELD_UPDATE_INTERVAL = 1.0   # sim-time seconds between field.update() calls


# ── init ───────────────────────────────────────────────────────────────────────

print('Loading probability map...')
pmap  = load_tif(TIFF_PATH)
field = ThermalField(
    pmap,
    z_i        = Z_I,
    spawn_rate = SPAWN_RATE,
    rng        = np.random.default_rng(42),
)
print(f'Map bounds : {pmap["bounds"]}')
print(f'World      : {WORLD}  model: {MODEL}')

node     = Node()
wind_pub = node.advertise(f'/world/{WORLD}/wind', Wind)

_last_field_update: float = -1.0
_cb_count: int = 0

# ── thermal log file ───────────────────────────────────────────────────────────

_thermals_dir = Path(__file__).parent.parent / 'data' / 'thermals'
_thermals_dir.mkdir(parents=True, exist_ok=True)
_run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
_thermals_path = _thermals_dir / f'thermals_{_run_ts}.jsonl'
_thermals_file = _thermals_path.open('w', buffering=1)   # line-buffered
print(f'Thermal log : {_thermals_path}')


# ── helpers ────────────────────────────────────────────────────────────────────

def _publish_wind(wx: float, wy: float, wz: float) -> None:
    msg = Wind()
    msg.enable_wind                = True
    msg.linear_velocity.x          = wx
    msg.linear_velocity.y          = wy
    msg.linear_velocity.z          = wz
    wind_pub.publish(msg)


# ── pose callback (fires at Gazebo physics rate, ~250 Hz) ─────────────────────

def pose_cb(msg: Pose_V) -> None:
    global _last_field_update, _cb_count

    # find the plane's pose in the Pose_V list
    plane_pose = None
    for p in msg.pose:
        if p.name == MODEL:
            plane_pose = p
            break
    if plane_pose is None:
        return

    # sim time from message header
    t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9

    # Gazebo ENU → UTM by adding world origin offset
    x = plane_pose.position.x + WORLD_ORIGIN_UTM[0]
    y = plane_pose.position.y + WORLD_ORIGIN_UTM[1]
    z = plane_pose.position.z   # altitude AGL [m]

    # advance thermal lifecycle at 1 Hz (gated on sim time)
    if t - _last_field_update >= FIELD_UPDATE_INTERVAL:
        field.update(t, AMBIENT_WIND)
        _last_field_update = t
        _thermals_file.write(
            json.dumps({"t": t, "thermals": field.snapshot()}) + '\n'
        )

    # query thermal lift at UAV position, publish at full 250 Hz
    wz = field.query(x, y, z)
    _publish_wind(float(AMBIENT_WIND[0]), float(AMBIENT_WIND[1]), wz)

    _cb_count += 1
    if _cb_count % 250 == 0:   # print once per second at 250 Hz
        print(
            f't={t:7.1f}s  '
            f'pos=({x:.0f}, {y:.0f}, {z:.1f}) m  '
            f'wz={wz:+.2f} m/s  '
            f'active thermals={len(field.thermals)}',
            flush=True,
        )


# ── subscribe ──────────────────────────────────────────────────────────────────

pose_topic = f'/world/{WORLD}/dynamic_pose/info'

if not node.subscribe(Pose_V, pose_topic, pose_cb):
    print(f'ERROR: failed to subscribe to {pose_topic}', file=sys.stderr)
    sys.exit(1)

print(f'Subscribed  : {pose_topic}')
print(f'Publishing  : /world/{WORLD}/wind')
print('Running — Ctrl-C to stop\n')


# ── spin ───────────────────────────────────────────────────────────────────────

_stop = False

def _handle_sigint(sig, frame):
    global _stop
    print(f'\nShutting down after {_cb_count} pose messages.')
    _thermals_file.close()
    _stop = True

signal.signal(signal.SIGINT, _handle_sigint)

while not _stop:
    time.sleep(0.1)
