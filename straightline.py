"""
straightline.py — fly the advanced_plane in a straight line at 50 m AGL.

Mission
-------
  TAKEOFF  →  straight 1 km north at 50 m  →  LOITER  →  LAND

Run after `python3 launch.py` (PX4 + Gazebo + gz_sim must already be up).
"""

from pymavlink import mavutil
import csv
import time
import threading
from datetime import datetime
from pathlib import Path

# ── telemetry logger ──────────────────────────────────────────────────────────

class TelemetryLogger:
    """
    Background thread that subscribes to MAVLink telemetry and writes a CSV.

    Columns: time_s, lat_deg, lon_deg, alt_agl_m, throttle_pct, airspeed_ms
    """

    _TELEMETRY_DIR = Path(__file__).parent / 'soaring' / 'data' / 'telemetry'

    def __init__(self, master):
        self._master   = master
        self._stop_evt = threading.Event()
        self._ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
        self.csv_path  = self._TELEMETRY_DIR / f'flight_{self._ts}.csv'
        self._thread   = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        print(f'[telemetry] logging to {self.csv_path}')

    def stop(self):
        self._stop_evt.set()
        self._thread.join(timeout=5)

    def _run(self):
        t0 = None
        # Latest values (updated by whichever message arrives first)
        state = {
            'lat': 0.0, 'lon': 0.0, 'alt_agl': 0.0,
            'throttle': 0.0, 'airspeed': 0.0,
        }
        with self.csv_path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_s', 'lat_deg', 'lon_deg',
                             'alt_agl_m', 'throttle_pct', 'airspeed_ms'])
            while not self._stop_evt.is_set():
                msg = self._master.recv_match(
                    type=['GLOBAL_POSITION_INT', 'VFR_HUD'],
                    blocking=True, timeout=0.5,
                )
                if msg is None:
                    continue
                now = time.time()
                if t0 is None:
                    t0 = now
                t = now - t0

                if msg.get_type() == 'GLOBAL_POSITION_INT':
                    state['lat']     = msg.lat     * 1e-7
                    state['lon']     = msg.lon     * 1e-7
                    state['alt_agl'] = msg.relative_alt * 1e-3   # mm → m
                elif msg.get_type() == 'VFR_HUD':
                    state['throttle'] = msg.throttle
                    state['airspeed'] = msg.airspeed

                writer.writerow([
                    f'{t:.3f}',
                    f'{state["lat"]:.7f}',
                    f'{state["lon"]:.7f}',
                    f'{state["alt_agl"]:.2f}',
                    f'{state["throttle"]:.1f}',
                    f'{state["airspeed"]:.2f}',
                ])

    @property
    def sentinel_path(self) -> Path:
        return Path(__file__).parent / '.last_flight'

    def write_sentinel(self):
        self.sentinel_path.write_text(str(self.csv_path))


# ── connection ────────────────────────────────────────────────────────────────

master = mavutil.mavlink_connection('udp:127.0.0.1:14550')
master.wait_heartbeat()
print(f"Connected: system {master.target_system}, component {master.target_component}")

# ── helpers ───────────────────────────────────────────────────────────────────

def send_heartbeat(master, stop_event):
    while not stop_event.is_set():
        master.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, 0
        )
        time.sleep(1.0)


def set_param(master, param_id, value, int_type=False):
    ptype = (mavutil.mavlink.MAV_PARAM_TYPE_INT32
             if int_type else mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(
        master.target_system, master.target_component,
        param_id.encode('utf-8'),
        float(value),
        ptype,
    )
    msg = master.recv_match(type='PARAM_VALUE', blocking=True, timeout=3)
    print(f"  {param_id} = {msg.param_value if msg else 'TIMEOUT'}")


def set_mode(master, mode_name):
    mapping = master.mode_mapping()
    if mode_name not in mapping:
        print(f"Unknown mode '{mode_name}'. Available: {list(mapping.keys())}")
        return False
    base_mode, custom_mode, sub_mode = mapping[mode_name]
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0,
        float(base_mode), float(custom_mode), float(sub_mode),
        0.0, 0.0, 0.0, 0.0
    )
    return True


def arm(master):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )
    master.motors_armed_wait()
    print("Armed")


def upload_mission(master, home_lat, home_lon, waypoints, alt):
    """
    Upload a mission:
      seq 0          : TAKEOFF at alt
      seq 1 … N      : NAV_WAYPOINT items from `waypoints`
      seq N+1        : LOITER_TO_ALT (300 m north of home, descend to 10 m)
      seq N+2        : LAND at home

    Parameters
    ----------
    waypoints : list of (lat, lon)  — altitude fixed to `alt` for all
    alt       : float  cruise altitude AGL [m]
    """
    items = []

    # seq 0 — takeoff
    items.append(dict(
        seq=0,
        frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
        command=mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        current=1, autocontinue=1,
        param1=15.0, param2=0.0, param3=0.0, param4=float('nan'),
        lat=home_lat, lon=home_lon, alt=float(alt),
    ))

    # seq 1 … N — cruise waypoints
    for i, (lat, lon) in enumerate(waypoints, start=1):
        items.append(dict(
            seq=i,
            frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            current=0, autocontinue=1,
            param1=0.0, param2=20.0, param3=0.0, param4=float('nan'),
            lat=lat, lon=lon, alt=float(alt),
        ))

    n = len(items)

    # seq N+1 — loiter down to approach alt before landing
    loiter_seq = n
    items.append(dict(
        seq=loiter_seq,
        frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
        command=mavutil.mavlink.MAV_CMD_NAV_LOITER_TO_ALT,
        current=0, autocontinue=1,
        param1=1.0, param2=80.0, param3=0.0, param4=1.0,
        lat=home_lat + 0.00270, lon=home_lon, alt=30.0,   # ~300 m north, 30 m
    ))

    # seq N+2 — land at home
    land_seq = n + 1
    items.append(dict(
        seq=land_seq,
        frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
        command=mavutil.mavlink.MAV_CMD_NAV_LAND,
        current=0, autocontinue=1,
        param1=0.0, param2=0.0, param3=0.0, param4=float('nan'),
        lat=home_lat, lon=home_lon, alt=0.0,
    ))

    # clear any stale mission from PX4's dataman before uploading
    master.mav.mission_clear_all_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
    )
    master.recv_match(type='MISSION_ACK', blocking=True, timeout=3)

    # upload
    master.mav.mission_count_send(
        master.target_system, master.target_component,
        len(items),
        mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
    )
    for item in items:
        req = master.recv_match(
            type=['MISSION_REQUEST_INT', 'MISSION_REQUEST'],
            blocking=True, timeout=5,
        )
        if req is None:
            print(f"  Timeout waiting for request at seq {item['seq']}")
            return False
        master.mav.mission_item_int_send(
            master.target_system, master.target_component,
            item['seq'], item['frame'], item['command'],
            item['current'], item['autocontinue'],
            item['param1'], item['param2'], item['param3'], item['param4'],
            int(item['lat'] * 1e7), int(item['lon'] * 1e7), item['alt'],
            mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
        )

    ack = master.recv_match(type='MISSION_ACK', blocking=True, timeout=10)
    if ack and ack.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
        print(f"Mission uploaded ({len(items)} items)")
        return land_seq   # return final seq so caller can wait for it
    print(f"Mission rejected: {ack}")
    return None


def wait_mission_done(master, final_seq, timeout=300):
    # Drain any MISSION_ITEM_REACHED messages buffered from before our mission
    # (e.g. from a residual mission PX4 auto-executed on startup)
    while master.recv_match(type=['MISSION_ITEM_REACHED', 'MISSION_RESULT'],
                            blocking=False):
        pass

    print("Monitoring mission...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = master.recv_match(
            type=['MISSION_ITEM_REACHED', 'MISSION_RESULT'],
            blocking=True, timeout=2,
        )
        if msg is None:
            continue
        seq = getattr(msg, 'seq', None)
        print(f"  Reached seq {seq}")
        if seq == final_seq:
            print("Land waypoint reached — mission complete.")
            return True
    print("Mission timed out.")
    return False


# ── mission definition ────────────────────────────────────────────────────────

HOME_LAT = 47.0500   # Val-de-Ruz, centre of probability map
HOME_LON =  6.9500
ALT      = 100.0         # cruise altitude AGL [m] — enough margin for RWTO climbout

# straight line: one waypoint 1 km north of home
WAYPOINTS = [
    (HOME_LAT + 0.009, HOME_LON),   # ≈ 1 km north
]

# ── run ───────────────────────────────────────────────────────────────────────

stop_hb = threading.Event()
threading.Thread(target=send_heartbeat, args=(master, stop_hb), daemon=True).start()

print("Setting SITL params...")
# INT32 params (enum / integer)
set_param(master, 'NAV_RCL_ACT',     0, int_type=True)   # disable RC loss failsafe
set_param(master, 'COM_LOW_BAT_ACT', 0, int_type=True)   # disable battery failsafe
set_param(master, 'GF_ACTION',       0, int_type=True)   # disable geofence
set_param(master, 'COM_DL_LOSS_T',  60, int_type=True)   # GCS loss timeout
set_param(master, 'COM_ARM_WO_GPS',  1, int_type=True)   # allow arm without GPS
# Disable preflight EKF checks that fire before SITL EKF has settled
set_param(master, 'COM_ARM_IMU_ACC', 9999.0)   # float: accel bias threshold [m/s²]
set_param(master, 'COM_ARM_IMU_GYR', 9999.0)   # float: gyro bias threshold  [rad/s]
# Takeoff: use runway takeoff (gz_advanced_plane needs ground roll, not hand-launch)
set_param(master, 'RWTO_TKOFF',      1, int_type=True)
# Airspeed: disable consistency checks (EKF has 0 wind estimate but gz_sim injects 2 m/s)
set_param(master, 'ASPD_DO_CHECKS',  0, int_type=True)
# REAL32 params (float)
set_param(master, 'FW_LND_ANG',     20)
set_param(master, 'BAT_CRIT_THR',    0)
set_param(master, 'BAT_EMERGEN_THR', 0)

# wait for EKF to settle before arming
print("Waiting for EKF to settle (20 s)...")
time.sleep(20)

tel = TelemetryLogger(master)

final_seq = upload_mission(master, HOME_LAT, HOME_LON, WAYPOINTS, ALT)
if final_seq is not None:
    tel.start()
    set_mode(master, 'MISSION')
    time.sleep(1)
    arm(master)
    wait_mission_done(master, final_seq, timeout=300)

stop_hb.set()
tel.stop()
tel.write_sentinel()
print(f'[telemetry] saved → {tel.csv_path}')
