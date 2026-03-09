import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field


# ── Physical constants ────────────────────────────────────────────────────────
GRAVITY               = 9.81
R_GAS                 = 8.314
BETA                  = 0.00367
VIRTUAL_MASS          = 0.5
GROUND_LEVEL_TEMP     = 25.0
GROUND_LEVEL_PRESSURE = 101325.0
M_AIR                 = 0.029
RHO_AIR               = 1.225


def ambient_temp(z: float) -> float:
    return GROUND_LEVEL_TEMP - (6.5 * z / 1000.0)

def ambient_pressure(z: float) -> float:
    T = ambient_temp(z) + 273.15
    return GROUND_LEVEL_PRESSURE * np.exp(-M_AIR * GRAVITY * z / (R_GAS * T))


# ── Bubble (same as thermal_field_state) ─────────────────────────────────────
@dataclass
class Bubble:
    position:    np.ndarray = field(default_factory=lambda: np.zeros(3))
    spawn_time:  float      = 0.0
    temperature: float      = 30.0
    v0:          float      = 1e6
    radius:      float      = 100.0
    velocity:    np.ndarray = field(default_factory=lambda: np.zeros(3))
    volume:      float      = 0.0
    alive:       bool       = True

    def __post_init__(self):
        self.volume = self.get_volume(ambient_temp(self.position[2]))

    def get_volume(self, t_amb: float, p_amb: float = GROUND_LEVEL_PRESSURE) -> float:
        return self.v0 * (self.temperature + 273.15) / (t_amb + 273.15) * \
               (GROUND_LEVEL_PRESSURE / p_amb)

    def get_lift_acceleration(self, t_amb: float) -> float:
        delta_t = self.temperature - t_amb
        if delta_t <= 0.0:
            return 0.0
        return (GRAVITY * BETA * delta_t) / (1.0 + VIRTUAL_MASS)

    def get_drag_acceleration(self, speed: float) -> float:
        C_d   = 0.47
        area  = np.pi * self.radius ** 2
        m_eff = RHO_AIR * self.volume * (1.0 + VIRTUAL_MASS)
        if m_eff <= 0.0:
            return 0.0
        return ( C_d * RHO_AIR * area * speed ** 2) / m_eff

    def update_temperature(self, t_amb: float, cooling_rate: float, dt: float) -> None:
        self.temperature += (t_amb - self.temperature) * (1.0 - np.exp(-cooling_rate * dt))

    def update(self, dt: float, wind: np.ndarray, cooling_rate: float = 0.005):
        if not self.alive:
            return

        z     = self.position[2]
        t_amb = ambient_temp(z)
        p_amb = ambient_pressure(z)

        vz       = self.velocity[2]
        lift_acc = self.get_lift_acceleration(t_amb)
        drag_acc = self.get_drag_acceleration(abs(vz))
        net_az   = lift_acc - np.sign(vz) * drag_acc if vz != 0 else lift_acc
        self.velocity[2] += net_az * dt

        self.velocity[:2] = wind
        self.position    += self.velocity * dt

        self.update_temperature(t_amb, cooling_rate, dt)
        self.volume = self.get_volume(t_amb, p_amb)

        if self.temperature <= t_amb + 0.1:
            self.alive = False

        return lift_acc, drag_acc, net_az   # expose for logging


# ── Single bubble test ────────────────────────────────────────────────────────
dt      = 1.0
T_end   = 600
cooling = 0.01
wind    = np.array([0.0, 0.0])   # no wind — isolate vertical dynamics

b = Bubble(
    temperature=ambient_temp(0.0) + 5.0,   # 5 deg C above ambient at ground
    v0=1e5,
    radius=100.0,
)

# Logging arrays
ts           = []
altitudes    = []
temperatures = []
amb_temps    = []
volumes      = []
lift_accs    = []
drag_accs    = []
net_accs     = []
velocities   = []

for t in np.arange(0, T_end, dt):
    if not b.alive:
        break

    z     = b.position[2]
    t_amb = ambient_temp(z)

    result = b.update(dt, wind, cooling)
    if result is None:
        break
    lift_acc, drag_acc, net_az = result

    ts.append(t)
    altitudes.append(z)
    temperatures.append(b.temperature)
    amb_temps.append(t_amb)
    volumes.append(b.volume)
    lift_accs.append(lift_acc)
    drag_accs.append(drag_acc)
    net_accs.append(net_az)
    velocities.append(b.velocity[2])

ts           = np.array(ts)
altitudes    = np.array(altitudes)
temperatures = np.array(temperatures)
amb_temps    = np.array(amb_temps)
volumes      = np.array(volumes)
lift_accs    = np.array(lift_accs)
drag_accs    = np.array(drag_accs)
net_accs     = np.array(net_accs)
velocities   = np.array(velocities)

print(f"Bubble alive for {ts[-1]:.0f} s, reached max altitude {altitudes.max():.1f} m")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)

# 1. Altitude
ax = axes[0]
ax.plot(ts, altitudes, color='steelblue', linewidth=1.5)
ax.set_ylabel("Altitude (m)")
ax.set_title("Single bubble diagnostic")
ax.grid(True, alpha=0.3)
ax.fill_between(ts, altitudes, alpha=0.1, color='steelblue')

# 2. Temperature: bubble vs ambient
ax = axes[1]
ax.plot(ts, temperatures, color='tomato',     linewidth=1.5, label='Bubble temp')
ax.plot(ts, amb_temps,    color='darkorange', linewidth=1.2,
        linestyle='--', label='Ambient temp')
ax.fill_between(ts, amb_temps, temperatures,
                where=(temperatures > amb_temps),
                alpha=0.15, color='tomato', label='ΔT (buoyant)')
ax.set_ylabel("Temperature (°C)")
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

# 3. Volume
ax = axes[2]
ax.plot(ts, volumes / 1e6, color='mediumseagreen', linewidth=1.5)
ax.set_ylabel("Volume (×10⁶ m³)")
ax.grid(True, alpha=0.3)

# 4. Accelerations
ax = axes[3]
ax.plot(ts, lift_accs, color='limegreen', linewidth=1.5, label='Lift acc')
ax.plot(ts, drag_accs, color='tomato',    linewidth=1.5, label='Drag acc')
ax.plot(ts, net_accs,  color='black',     linewidth=1.2,
        linestyle='--', label='Net acc')
ax.axhline(0, color='gray', linewidth=0.8)
ax.set_ylabel("Acceleration (m/s²)")
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

# 5. Vertical velocity
ax = axes[4]
ax.plot(ts, velocities, color='mediumpurple', linewidth=1.5)
ax.axhline(0, color='gray', linewidth=0.8)
ax.set_ylabel("vz (m/s)")
ax.set_xlabel("Time (s)")
ax.grid(True, alpha=0.3)
ax.fill_between(ts, velocities, alpha=0.15, color='mediumpurple')

plt.tight_layout()
plt.savefig("bubble_single_test.png", dpi=150)
plt.show()
print("Plot saved to bubble_single_test.png")