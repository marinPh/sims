from pymavlink import mavutil
import time
import threading

master = mavutil.mavlink_connection('udp:127.0.0.1:14550')
master.wait_heartbeat()
print(f"Connected: system {master.target_system}, component {master.target_component}")

def send_heartbeat(master, stop_event):
    while not stop_event.is_set():
        master.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, 0
        )
        time.sleep(1.0)

def set_param(master, param_id, value):
    master.mav.param_set_send(
        master.target_system, master.target_component,
        param_id.encode('utf-8'),
        float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    )
    msg = master.recv_match(type='PARAM_VALUE', blocking=True, timeout=3)
    print(f"  {param_id} = {msg.param_value if msg else 'TIMEOUT'}")

def set_mode(master, mode_name):
    mapping = master.mode_mapping()
    if mode_name not in mapping:
        print(f"Available modes: {list(mapping.keys())}")
        return
    base_mode, custom_mode, sub_mode = mapping[mode_name]
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0,
        float(base_mode), float(custom_mode), float(sub_mode),
        0.0, 0.0, 0.0, 0.0
    )

def arm(master):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )
    master.motors_armed_wait()
    print("Armed")

def upload_mission(master, waypoints, home_lat, home_lon):
    items = []

    items.append({
        'seq': 0,
        'frame': mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
        'command': mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        'current': 1, 'autocontinue': 1,
        'param1': 15.0, 'param2': 0.0, 'param3': 0.0, 'param4': float('nan'),
        'lat': home_lat, 'lon': home_lon,
        'alt': float(waypoints[0][2])
    })

    for i, (lat, lon, alt) in enumerate(waypoints):
        items.append({
            'seq': i + 1,
            'frame': mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            'command': mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            'current': 0, 'autocontinue': 1,
            'param1': 0.0, 'param2': 20.0, 'param3': 0.0, 'param4': float('nan'),
            'lat': lat, 'lon': lon, 'alt': float(alt)
        })

    LOITER_RADIUS = 80.0
    APPROACH_ALT  = 10.0
    LOITER_LAT = home_lat + 0.00270   # ~300m north
    LOITER_LON = home_lon
    items.append({
        'seq': 6,
        'frame': mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
        'command': mavutil.mavlink.MAV_CMD_NAV_LOITER_TO_ALT,
        'current': 0, 'autocontinue': 1,
        'param1': 1.0, 'param2': LOITER_RADIUS,
        'param3': 0.0, 'param4': 1.0,
        'lat': LOITER_LAT, 'lon': LOITER_LON,
        'alt': APPROACH_ALT
    })

    items.append({
        'seq': 7,
        'frame': mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
        'command': mavutil.mavlink.MAV_CMD_NAV_LAND,
        'current': 0, 'autocontinue': 1,
        'param1': 0.0, 'param2': 0.0, 'param3': 0.0, 'param4': float('nan'),
        'lat': home_lat, 'lon': home_lon, 'alt': 0.0
    })

    master.mav.mission_count_send(
        master.target_system, master.target_component,
        len(items),
        mavutil.mavlink.MAV_MISSION_TYPE_MISSION
    )
    for item in items:
        msg = master.recv_match(type=['MISSION_REQUEST_INT', 'MISSION_REQUEST'],
                                blocking=True, timeout=5)
        if msg is None:
            print(f"Timeout at seq {item['seq']}")
            return False
        master.mav.mission_item_int_send(
            master.target_system, master.target_component,
            item['seq'], item['frame'], item['command'],
            item['current'], item['autocontinue'],
            item['param1'], item['param2'], item['param3'], item['param4'],
            int(item['lat'] * 1e7), int(item['lon'] * 1e7), item['alt'],
            mavutil.mavlink.MAV_MISSION_TYPE_MISSION
        )
    ack = master.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
    if ack and ack.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
        print(f"Mission uploaded: {len(items)} items")
        return True
    print(f"Mission rejected: {ack}")
    return False

def wait_mission_done(master, timeout=300):
    print("Waiting for mission to complete...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = master.recv_match(
            type=['MISSION_ITEM_REACHED', 'MISSION_RESULT'],
            blocking=True, timeout=2
        )
        if msg:
            print(f"  Reached item: {msg.seq if hasattr(msg, 'seq') else msg}")
            if hasattr(msg, 'seq') and msg.seq == 7:
                print("Land waypoint reached, mission complete.")
                return True
    print("Mission timeout.")
    return False

# ── Setup ────────────────────────────────────────────────────────────────────
BASE_LAT = 47.3977
BASE_LON =  8.5456
ALT = 80.0

waypoints = [
    (BASE_LAT + 0.001, BASE_LON + 0.000, ALT),
    (BASE_LAT + 0.001, BASE_LON + 0.001, ALT),
    (BASE_LAT + 0.000, BASE_LON + 0.001, ALT),
    (BASE_LAT - 0.001, BASE_LON + 0.001, ALT),
    (BASE_LAT - 0.001, BASE_LON + 0.000, ALT),
]

HOME_LAT = 47.3977
HOME_LON =  8.5456

# ── Start heartbeat thread ───────────────────────────────────────────────────
stop_hb = threading.Event()
threading.Thread(target=send_heartbeat, args=(master, stop_hb), daemon=True).start()

# ── Fix SITL failsafe params ─────────────────────────────────────────────────
print("Setting SITL params...")
set_param(master, 'NAV_RCL_ACT',    0)   # disable RC loss failsafe
set_param(master, 'COM_LOW_BAT_ACT', 0)  # disable battery failsafe
set_param(master, 'GF_ACTION',       0)  # disable geofence
set_param(master, 'COM_DL_LOSS_T',  60)  # GCS loss timeout 60s
set_param(master, 'FW_LND_ANG',     20)  # allow steep landing glide

time.sleep(2)  # let PX4 register GCS heartbeats

# ── Upload and execute ───────────────────────────────────────────────────────
if upload_mission(master, waypoints, HOME_LAT, HOME_LON):
    set_mode(master, 'MISSION')
    time.sleep(1)
    arm(master)          # arm first
    time.sleep(1)
    # takeoff is handled by the mission's seq 0 item — no separate takeoff command needed
    wait_mission_done(master, timeout=300)

stop_hb.set()