#!/usr/bin/env python3
"""
Allen (2006) thermal model injected into Gazebo SITL wind field.
Computes w(x, y) at the aircraft position and publishes to Gazebo wind topic.
"""


import math

# ── If not using ROS2, use the gz-transport Python bindings instead ──────────
# See bottom of file for the non-ROS gz-msgs approach

class AllenThermal:
    """
    Allen (2006) thermal model.

    Vertical wind speed at distance d from thermal centre:

        w(d) = w_max * (d/R) * exp(1 - d/R)   for d <= 2R  (updraft core)
        w(d) = -V_e                              for d >  2R  (environmental sink)

    where:
        w_max  : peak updraft speed (m/s)
        R      : thermal radius (m)  — radius of max updraft ring
        V_e    : environmental sink rate (m/s, positive value = downward)
    """

    def __init__(self,
                 center_lat: float, center_lon: float,
                 center_x: float = 0.0, center_y: float = 0.0,
                 w_max: float = 3.0,
                 R: float = 80.0,
                 V_e: float = 0.5,
                 z_max: float = 1500.0,
                 z_base: float = 0.0):
        """
        center_x, center_y : thermal centre in local ENU metres (from SITL origin)
        w_max  : peak updraft (m/s)
        R      : characteristic radius (m)
        V_e    : environmental sink rate (m/s)
        z_max  : top of thermal (m AGL) — updraft decays above this
        z_base : base of thermal (m AGL) — updraft starts here
        """
        self.cx    = center_x
        self.cy    = center_y
        self.w_max = w_max
        self.R     = R
        self.V_e   = V_e
        self.z_max = z_max
        self.z_base = z_base

    def updraft(self, x: float, y: float, z: float) -> float:
        """
        Returns vertical wind speed w_z (m/s) at position (x, y, z).
        Positive = upward.
        """
        d = math.sqrt((x - self.cx)**2 + (y - self.cy)**2)

        # Vertical shape: trapezoidal profile, zero below base and above top
        if z < self.z_base or z > self.z_max:
            return 0.0

        # Taper off near the top (top 20% of thermal height)
        taper_start = self.z_max - 0.2 * (self.z_max - self.z_base)
        if z > taper_start:
            vertical_scale = (self.z_max - z) / (self.z_max - taper_start)
        else:
            vertical_scale = 1.0

        # Allen horizontal profile
        if d <= 2 * self.R:
            # Updraft core: bell-shaped ring peaking at r = R
            r = d / self.R
            w = self.w_max * r * math.exp(1.0 - r)
        else:
            # Outside core: environmental sink
            w = -self.V_e

        return w * vertical_scale


class ThermalWindInjector:
    """
    Reads aircraft position from pymavlink and publishes
    wind to Gazebo using gz-transport (no ROS required).
    """

    def __init__(self, thermals: list, wind_x: float = 2.0, wind_y: float = 0.0):
        """
        thermals  : list of AllenThermal instances
        wind_x/y  : background horizontal wind (m/s, ENU frame)
        """
        self.thermals = thermals
        self.wind_x   = wind_x   # eastward
        self.wind_y   = wind_y   # northward

    def get_wind(self, x: float, y: float, z: float) -> tuple:
        """Returns (wx, wy, wz) wind vector at position."""
        wz = sum(t.updraft(x, y, z) for t in self.thermals)
        return self.wind_x, self.wind_y, wz


# ── Gazebo injection via gz-transport Python API ─────────────────────────────

def run(master, injector):
    """
    Main loop: poll aircraft position, compute wind, publish to Gazebo.
    Requires: gz-transport Python bindings  (pip install gz-transport13)
    OR: use subprocess to call gz topic pub (simpler, shown below)
    """
    import subprocess, json

    while True:
        msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
        if msg is None:
            continue

        # Convert GPS to local ENU metres relative to SITL origin
        # SITL origin is set in PX4 via PX4_HOME_LAT / PX4_HOME_LON
        lat  = msg.lat / 1e7
        lon  = msg.lon / 1e7
        alt  = msg.relative_alt / 1000.0

        # Simple flat-earth conversion (fine within ~1km)
        R_earth = 6371000.0
        origin_lat = 47.3977  # must match PX4_HOME_LAT
        origin_lon = 8.5456

        x = (lon - origin_lon) * math.cos(math.radians(origin_lat)) * math.pi/180 * R_earth
        y = (lat - origin_lat) * math.pi/180 * R_earth
        z = alt

        wx, wy, wz = injector.get_wind(x, y, z)

        # Publish to Gazebo wind topic using gz CLI (no Python bindings needed)
        # Topic: /world/default/wind  (adjust world name to match your SDF)
        wind_json = json.dumps({
            "header": {"stamp": {"sec": 0, "nsec": 0}},
            "linear_velocity": {"x": wx, "y": wy, "z": wz}
        })

        subprocess.Popen(
            ['gz', 'topic', '-t', '/world/default/wind',
             '-m', 'gz.msgs.Wind', '-p',
             f'linear_velocity: {{x: {wx}, y: {wy}, z: {wz}}}'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        print(f"  pos=({x:.1f}, {y:.1f}, {z:.1f}m)  wind=({wx:.2f}, {wy:.2f}, {wz:.2f}) m/s")

        import time; time.sleep(0.2)  # 5 Hz update


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from pymavlink import mavutil
    import time

    # Connect to SITL
    master = mavutil.mavlink_connection('udp:127.0.0.1:14550')
    master.wait_heartbeat()
    print("Connected to SITL")

    # Define thermals in local ENU coords (metres from SITL origin)
    thermals = [
        AllenThermal(
            center_lat=47.3977, center_lon=8.5456,  # for reference only
            center_x=150.0, center_y=100.0,          # 150m east, 100m north of origin
            w_max=3.5,   # strong thermal
            R=80.0,      # 80m radius
            V_e=0.5,     # 0.5 m/s environmental sink outside core
            z_max=1200.0,
            z_base=50.0
        ),
        AllenThermal(
            center_lat=47.3977, center_lon=8.5456,
            center_x=-100.0, center_y=200.0,         # second weaker thermal
            w_max=2.0,
            R=60.0,
            V_e=0.3,
            z_max=1000.0,
            z_base=30.0
        ),
    ]

    injector = ThermalWindInjector(
        thermals=thermals,
        wind_x=2.0,   # 2 m/s eastward background wind
        wind_y=0.5,
    )

    run(master, injector)