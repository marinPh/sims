import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from dataclasses import dataclass, field
from numpy import random as npr


# ── Physical constants ────────────────────────────────────────────────────────
GRAVITY               = 9.81     # m/s²
R_GAS                 = 8.314    # J/(mol·K)
BETA                  = 0.00367  # 1/K, volumetric expansion coeff for air
VIRTUAL_MASS          = 0.5      # fraction of displaced air contributing to inertia
GROUND_LEVEL_TEMP     = 25.0     # °C at ground level
DELTA_TEMP            = 5.0      # °C, base temperature excess for spawned bubbles
GROUND_LEVEL_PRESSURE = 101325.0 # Pa at sea level
M_AIR                 = 0.029    # kg/mol, molar mass of dry air
RHO_AIR               = 1.225    # kg/m³, air density at sea level


# ── Atmosphere helpers ────────────────────────────────────────────────────────
def ambient_temp(z: float) -> float:
    """Standard lapse rate: -6.5 deg C per 1000 m."""
    return GROUND_LEVEL_TEMP - (6.5 * z / 1000.0)


def ambient_pressure(z: float) -> float:
    """Barometric formula: P = P0 * exp(-M*g*z / (R*T))."""
    T = ambient_temp(z) + 273.15  # K
    return GROUND_LEVEL_PRESSURE * np.exp(-M_AIR * GRAVITY * z / (R_GAS * T))


# ── Time-varying wind ─────────────────────────────────────────────────────────
def wind_vector(t: float, t_end: float,
                max_speed: float = 8.0,
                direction_deg: float = 45.0) -> np.ndarray:
    """
    Wind speed follows a Gaussian bell centred at t_end/2, zero at t=0 and t=t_end.
    Peak is realistic thermal soaring wind: ~8 m/s (typical convective day).

    Uses a truncated Gaussian so the tails are numerically zero at the endpoints:
        w(t) = max_speed * exp(-0.5 * ((t - mu) / sigma)^2)
        normalised so w(0) ≈ 0 and w(t_end) ≈ 0.
    """
    mu    = t_end / 2.0
    # sigma chosen so that w(0) / w(mu) < 0.01  →  sigma = mu / sqrt(2*ln(100))
    sigma = mu / np.sqrt(2.0 * np.log(100.0))
    speed = max_speed * np.exp(-0.5 * ((t - mu) / sigma) ** 2)

    rad = np.deg2rad(direction_deg)
    return speed * np.array([np.cos(rad), np.sin(rad)])


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

        if self.temperature <= t_amb + 0.2:
            self.alive = False

        return lift_acc, drag_acc, net_az   # expose for logging


# ── Spawner ───────────────────────────────────────────────────────────────────
@dataclass
class Spawner:
    spawn_rate:       float
    spawn_xy:         np.ndarray = field(default_factory=lambda: np.zeros(2))
    spawn_stddev:     float      = 10.0
    spawn_radius:     float      = 500.0
    base_temp_excess: float      = DELTA_TEMP
    bubble_v0:        float      = 1e6
    bubbles:          list       = field(default_factory=list)
    
    def spawn_bubble(self, time: float) -> Bubble:
        offset = npr.normal(0, self.spawn_stddev, size=2)
        if np.linalg.norm(offset) > self.spawn_radius:
            offset = offset / np.linalg.norm(offset) * self.spawn_radius
        pos_xy = self.spawn_xy + offset
        temp   = ambient_temp(0.0) + self.base_temp_excess + npr.normal(0, 1.0)
        bubble = Bubble(
            position=np.array([pos_xy[0], pos_xy[1], 0.0]),
            spawn_time=time,
            temperature=temp,
            v0=self.bubble_v0,
        )
        self.bubbles.append(bubble)
        return bubble

    def update(self, dt: float, wind: np.ndarray,
               cooling_rate: float = 0.005) -> None:
        for b in self.bubbles:
            if b.alive:
                b.update(dt, wind, cooling_rate)

    def cull_dead(self) -> None:
        self.bubbles = [b for b in self.bubbles if b.alive]

    @property
    def live_bubbles(self):
        return [b for b in self.bubbles if b.alive]


# ── Simulation ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    npr.seed(42)

    sim           = Spawner(spawn_rate=0.5)
    dt            = 1.0
    T_end         = 2000
    cooling       = 0.01
    WIND_MAX      = 8.0    # m/s, realistic convective day peak
    WIND_DIR_DEG  = 45.0   # NE direction

    snapshots     = {}
    wind_history  = []   # (t, speed) for the inset plot

    for t in np.arange(0, T_end, dt):
        wind = wind_vector(t, T_end, max_speed=WIND_MAX, direction_deg=WIND_DIR_DEG)
        wind_history.append((t, float(np.linalg.norm(wind))))

        n_spawn = npr.poisson(sim.spawn_rate * dt)
        for _ in range(n_spawn):
            sim.spawn_bubble(t)

        sim.update(dt, wind, cooling)
        sim.cull_dead()

        if int(t) % 100 == 0 and int(t) not in snapshots:
            if sim.live_bubbles:
                snapshots[int(t)] = np.array(
                    [[b.position[0], b.position[1], b.position[2], b.velocity[2]]
                     for b in sim.live_bubbles])
            else:
                snapshots[int(t)] = np.empty((0, 4))

    print(f"Total bubbles spawned : {len(sim.bubbles)}")
    print(f"Live at end           : {len(sim.live_bubbles)}")

    wind_history = np.array(wind_history)   # shape (N, 2): col0=t, col1=speed

    # ── Ambient temperature profile ───────────────────────────────────────────
    z_profile = np.linspace(0, 3000, 500)
    t_profile = np.array([ambient_temp(z) for z in z_profile])

    # ── Plot ──────────────────────────────────────────────────────────────────
    times = sorted(snapshots.keys())
    n     = len(times)

    # Main grid: n scatter panels + 1 wind panel on the right
    fig   = plt.figure(figsize=(5 * n + 3, 6))
    axes  = [fig.add_subplot(1, n + 1, i + 1) for i in range(n)]
    ax_w  = fig.add_subplot(1, n + 1, n + 1)

    all_vz = np.concatenate([snapshots[t][:, 3] for t in times
                              if snapshots[t].shape[0] > 0])
    vmin, vmax = (all_vz.min(), all_vz.max()) if len(all_vz) else (0, 1)

    for ax, t in zip(axes, times):
        data  = snapshots[t]
        x_min = data[:, 0].min() - 50 if data.shape[0] > 0 else -200
        x_max = data[:, 0].max() + 50 if data.shape[0] > 0 else  200

        T_min, T_max = t_profile.min(), t_profile.max()
        t_norm = (t_profile - T_min) / (T_max - T_min)
        x_line = x_min + t_norm * (x_max - x_min)

        ax.fill_betweenx(z_profile, x_min, x_line,
                         alpha=0.12, color='orange')
        ax.plot(x_line, z_profile,
                color='darkorange', linewidth=1.2, alpha=0.6, label='Ambient T')

        for z_ann in [0, 500, 1000, 1500, 2000]:
            T_ann = ambient_temp(z_ann)
            x_ann = x_min + ((T_ann - T_min) / (T_max - T_min)) * (x_max - x_min)
            ax.annotate(f'{T_ann:.1f}°C', xy=(x_ann, z_ann),
                        xytext=(4, 0), textcoords='offset points',
                        fontsize=7, color='darkorange', alpha=0.8)

        # Wind speed at this snapshot time — vertical line marker
        w_t = float(np.linalg.norm(
            wind_vector(t, T_end, max_speed=WIND_MAX, direction_deg=WIND_DIR_DEG)))
        ax.set_title(f"t={t}s  |wind|={w_t:.1f} m/s\n(n={data.shape[0]})",
                     fontsize=9)

        if data.shape[0] > 0:
            sc = ax.scatter(data[:, 0], data[:, 2], c=data[:, 3],
                            cmap='RdYlGn', vmin=vmin, vmax=vmax,
                            alpha=0.7, s=12, zorder=3)
            plt.colorbar(sc, ax=ax, label='vz (m/s)')

        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper right')

    # ── Wind profile panel ────────────────────────────────────────────────────
    ax_w.plot(wind_history[:, 0], wind_history[:, 1],
              color='steelblue', linewidth=2)
    ax_w.fill_between(wind_history[:, 0], wind_history[:, 1],
                      alpha=0.2, color='steelblue')

    # Mark snapshot times
    for t in times:
        w_t = float(np.linalg.norm(
            wind_vector(t, T_end, max_speed=WIND_MAX, direction_deg=WIND_DIR_DEG)))
        ax_w.axvline(t, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax_w.scatter([t], [w_t], color='red', zorder=5, s=30)

    ax_w.set_xlabel("Time (s)")
    ax_w.set_ylabel("|wind| (m/s)")
    ax_w.set_title(f"Wind speed\n(dir={WIND_DIR_DEG:.0f}°, peak={WIND_MAX} m/s)",
                   fontsize=9)
    ax_w.set_ylim(bottom=0)
    ax_w.grid(True, alpha=0.3)

    plt.suptitle("Thermal bubbles — ambient T background, Gaussian time-varying wind")
    plt.tight_layout()
    plt.savefig("thermal_bubbles.png", dpi=150)
    plt.show()
    print("Plot saved to thermal_bubbles.png")