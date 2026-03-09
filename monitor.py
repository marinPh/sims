from pymavlink import mavutil
import time
import threading

master = mavutil.mavlink_connection('udp:127.0.0.1:14550')
master.wait_heartbeat()
print(f"Connected: system {master.target_system}, component {master.target_component}")

# ── Request all needed message streams ──────────────────────────────────────
def request_stream(master, stream_id, rate_hz):
    master.mav.request_data_stream_send(
        master.target_system, master.target_component,
        stream_id, rate_hz, 1
    )

request_stream(master, mavutil.mavlink.MAV_DATA_STREAM_ALL, 4)

# ── Telemetry state ──────────────────────────────────────────────────────────
telem = {
    'gps':      {'lat': None, 'lon': None, 'alt': None, 'fix': None, 'sats': None, 'hdop': None},
    'airspeed': {'airspeed': None, 'groundspeed': None, 'heading': None, 'climb': None, 'throttle': None},
    'battery':  {'voltage': None, 'current': None, 'remaining': None},
    'wind':     {'dir': None, 'speed': None, 'speed_z': None},
    'vfr':      {'alt': None},
    'status':   {'armed': None, 'mode': None, 'health': None},
}

HEALTH_FLAGS = {
    0:  'GYRO',       1:  'ACCEL',      2:  'MAG',
    3:  'ABS_PRESS',  4:  'DIFF_PRESS', 5:  'GPS',
    6:  'OPT_FLOW',   7:  'COMP_CTRL',  8:  'VIS_FLOW',
    9:  'POS_HORIZ',  10: 'POS_VERT',   11: 'POS_TERRAIN',
    12: 'MOTORS',     13: 'RC',         14: 'GEOFENCE',
    15: 'AHRS',       16: 'TERRAIN',    17: 'REVERSE_MOTOR',
    18: 'LOGGING',    19: 'BATTERY',    20: 'PROXIMITY',
    21: 'SATCOM',     22: 'PREARM',     23: 'OBSTACLE_AVOIDANCE',
}

def parse_health(bitmask):
    failed = [name for bit, name in HEALTH_FLAGS.items() if not (bitmask >> bit & 1)]
    return failed if failed else ['OK']

# ── Message parser thread ────────────────────────────────────────────────────
stop = threading.Event()

def parse_messages():
    while not stop.is_set():
        msg = master.recv_match(blocking=True, timeout=0.5)
        if msg is None:
            continue
        t = msg.get_type()

        if t == 'GLOBAL_POSITION_INT':
            telem['gps']['lat'] = msg.lat / 1e7
            telem['gps']['lon'] = msg.lon / 1e7
            telem['gps']['alt'] = msg.relative_alt / 1000.0
            telem['vfr']['alt'] = msg.alt / 1000.0

        elif t == 'GPS_RAW_INT':
            fix_map = {0:'NO_GPS', 1:'NO_FIX', 2:'2D', 3:'3D', 4:'DGPS', 5:'RTK_FLOAT', 6:'RTK_FIXED'}
            telem['gps']['fix']  = fix_map.get(msg.fix_type, str(msg.fix_type))
            telem['gps']['sats'] = msg.satellites_visible
            telem['gps']['hdop'] = msg.eph / 100.0

        elif t == 'VFR_HUD':
            telem['airspeed']['airspeed']    = msg.airspeed
            telem['airspeed']['groundspeed'] = msg.groundspeed
            telem['airspeed']['heading']     = msg.heading
            telem['airspeed']['climb']       = msg.climb
            telem['airspeed']['throttle']    = msg.throttle

        elif t == 'SYS_STATUS':
            telem['battery']['voltage']   = msg.voltage_battery / 1000.0
            telem['battery']['current']   = msg.current_battery / 100.0
            telem['battery']['remaining'] = msg.battery_remaining
            telem['status']['health']     = parse_health(msg.onboard_control_sensors_health)

        elif t == 'BATTERY_STATUS':
            if msg.voltages[0] != 65535:
                telem['battery']['voltage'] = msg.voltages[0] / 1000.0
            telem['battery']['remaining'] = msg.battery_remaining

        elif t == 'WIND':
            telem['wind']['dir']     = msg.direction
            telem['wind']['speed']   = msg.speed
            telem['wind']['speed_z'] = msg.speed_z

        elif t == 'HEARTBEAT':
            armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            telem['status']['armed'] = 'ARMED' if armed else 'DISARMED'
            telem['status']['mode']  = master.flightmode

parser_thread = threading.Thread(target=parse_messages, daemon=True)
parser_thread.start()

# ── Display ──────────────────────────────────────────────────────────────────
def fmt(val, unit='', decimals=2):
    if val is None:
        return '\033[33m---\033[0m'
    return f'\033[32m{val:.{decimals}f}{unit}\033[0m'

def fmt_str(val):
    if val is None:
        return '\033[33m---\033[0m'
    return f'\033[32m{val}\033[0m'

def battery_color(pct):
    if pct is None:
        return '\033[33m---\033[0m'
    color = '\033[32m' if pct > 50 else '\033[33m' if pct > 20 else '\033[31m'
    return f'{color}{pct}%\033[0m'

def health_color(flags):
    if flags is None:
        return '\033[33m---\033[0m'
    if flags == ['OK']:
        return '\033[32mOK\033[0m'
    return f'\033[31mFAIL: {", ".join(flags)}\033[0m'

first_print = True

def print_telem():
    global first_print
    g = telem['gps']
    a = telem['airspeed']
    b = telem['battery']
    w = telem['wind']
    s = telem['status']

    lines = [
        '\033[1m══════════════════════ TELEMETRY MONITOR ══════════════════════\033[0m',
        '\033[1m  STATUS\033[0m',
        f'    Armed   : {fmt_str(s["armed"])}',
        f'    Mode    : {fmt_str(s["mode"])}',
        f'    Health  : {health_color(s["health"])}',
        '\033[1m\n  GPS\033[0m',
        f'    Fix     : {fmt_str(g["fix"])}   Sats: {fmt(g["sats"], "", 0)}   HDOP: {fmt(g["hdop"], "m")}',
        f'    Lat/Lon : {fmt(g["lat"], "°", 6)} / {fmt(g["lon"], "°", 6)}',
        f'    Alt AGL : {fmt(g["alt"], "m")}   Alt MSL: {fmt(telem["vfr"]["alt"], "m")}',
        '\033[1m\n  AIRSPEED\033[0m',
        f'    Airspeed    : {fmt(a["airspeed"], " m/s")}   Groundspeed: {fmt(a["groundspeed"], " m/s")}',
        f'    Climb rate  : {fmt(a["climb"], " m/s")}   Heading: {fmt(a["heading"], "°", 0)}',
        f'    Throttle    : {fmt(a["throttle"], "%", 0)}',
        '\033[1m\n  BATTERY\033[0m',
        f'    Voltage : {fmt(b["voltage"], "V")}   Current: {fmt(b["current"], "A")}',
        f'    Charge  : {battery_color(b["remaining"])}',
        '\033[1m\n  WIND ESTIMATE\033[0m',
        f'    Direction : {fmt(w["dir"], "°", 0)}   Speed: {fmt(w["speed"], " m/s")}',
        f'    Vertical  : {fmt(w["speed_z"], " m/s")}  <- thermal proxy (positive = updraft)',
        '\033[90m\n  Ctrl+C to exit\033[0m',
    ]

    if first_print:
        print('\n'.join(lines))
        first_print = False
    else:
        line_count = len(lines) + 4
        print(f'\033[{line_count}A', end='')
        for line in lines:
            print(f'{line}\033[K')

try:
    while True:
        print_telem()
        time.sleep(0.5)

except KeyboardInterrupt:
    stop.set()
    print("\nMonitor stopped.")
