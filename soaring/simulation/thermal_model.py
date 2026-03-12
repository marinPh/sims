# simulation/thermal_state.py
"""
A single thermal updraft.

Lifecycle
---------
  Gaussian envelope over the thermal's lifetime T_life:
    envelope(τ) = exp(-0.5 * ((τ - μ) / σ)²)
  where μ = 0.5·T_life (peak at midlife), σ = 0.25·T_life
  → rises smoothly, peaks at 50% of life, fades to ~0 at death

Spatial model: Bencatel (2013)
------------------------------
  Extends Allen with Gedeon downdraft ring.
  Vertical velocity at (x, y, z):

    d    = horizontal distance from thermal centre
    r    = d / R                     (normalised radius)

    Core (r ≤ 1):
      w_c(r) = w_max · envelope(τ) · (1 - r²)² · f_z(z)

    Downdraft ring (1 < r ≤ 2):
      w_d(r) = -w_sink · (r-1)·(2-r) · f_z(z)
      w_sink = w_max · envelope(τ) · SINK_RATIO

    Vertical profile (Lenschow):
      f_z(z) = (z/z_i) · exp(1 - z/z_i)    peaks at z = z_i

    Wind lean: thermal axis tilts with wind — the effective horizontal
    centre shifts with altitude:
      cx(z) = x0_current + (z / w_updraft_mean) · u_wind  [approx]
      cy(z) = y0_current + (z / w_updraft_mean) · v_wind

Drift
-----
  Centre translates with wind vector (u, v) [m/s]:
    x(t) = x0 + u · (t - t_spawn)
    y(t) = y0 + v · (t - t_spawn)

Usage
-----
    from thermal_state import ThermalState

    th = ThermalState(x0=500, y0=300, p_val=0.75,
                      t_spawn=0.0, wind=(2.0, 0.5),
                      z_i=1200.0, rng=rng)

    th.update(t=60.0)           # advance to t=60s
    w = th.query(x, y, z)       # vertical wind speed [m/s]
    alive = th.is_alive         # False when envelope ≈ 0
"""

import numpy as np

# ── constants ──────────────────────────────────────────────────────────────
W_MAX_MIN   = 0.5    # m/s  minimum peak updraft  (P=0)
W_MAX_MAX   = 5.0    # m/s  maximum peak updraft  (P=1)
T_LIFE_MIN  = 180.0  # s    minimum lifespan      (P=0)
T_LIFE_MAX  = 6000.0 # s    maximum lifespan      (P=1)
R_BASE_MIN  = 50.0   # m    minimum radius        (P=0)
R_BASE_MAX  = 200.0  # m    maximum radius        (P=1)
SINK_RATIO  = 0.5    # downdraft strength relative to peak updraft
DEAD_THRESH = 0.02   # envelope below this → thermal is dead
NOISE_SIGMA = 0.10   # log-normal noise on physical parameters


class ThermalState:
    """
    Single thermal updraft.  Mutated in place by update(t).

    Parameters
    ----------
    x0, y0   : float   spawn location [m, UTM]
    p_val    : float   spawn probability at location ∈ [0,1]
    t_spawn  : float   simulation time at spawn [s]
    wind     : (u, v)  ambient wind vector [m/s]
    z_i      : float   convective boundary layer height [m]
    rng      : np.random.Generator  for reproducible noise
    """

    def __init__(self,pos: np.ndarray, p_val: float,
                 t_spawn: float, wind: np.ndarray = np.zeros(2),
                 z_i: float = 1000.0,
                 rng: np.random.Generator = None):

        if rng is None:
            rng = np.random.default_rng()

        self.pos     = pos
        self.p_val   = float(np.clip(p_val, 0.0, 1.0))
        self.t_spawn = t_spawn
        self.wind    = wind
        self.z_i     = z_i

        # ── physical parameters derived from p_val + noise ─────────────
        def _lmap(p, v_min, v_max):
            return v_min + p * (v_max - v_min)

        def _noisy(val):
            return float(val * rng.lognormal(0.0, NOISE_SIGMA))

        self.w_max  = _noisy(_lmap(self.p_val, W_MAX_MIN, W_MAX_MAX))
        self.T_life = _noisy(_lmap(self.p_val, T_LIFE_MIN, T_LIFE_MAX))
        self.R      = _noisy(_lmap(self.p_val, R_BASE_MIN, R_BASE_MAX))

        # Gaussian lifecycle: peak at 50% of T_life
        self._mu      = 0.5  * self.T_life
        self._sigma   = 0.25 * self.T_life
        self._inv_sigma = 1.0 / self._sigma

        # ── mutable state (updated each tick) ──────────────────────────
        self.center   = np.array(pos, dtype=float)  # current centre (cx, cy)
        self.age      = 0.0    # τ = t - t_spawn  [s]
        self.envelope = 1.0    # current lifecycle scalar ∈ [0,1]
        self.is_alive = True

    # ── update ─────────────────────────────────────────────────────────────

    def update(self, t: float, ambient_wind : np.ndarray) -> bool:
        """
        Advance thermal state to simulation time t.

        Mutates: age, cx, cy, envelope, is_alive.
        Returns is_alive.
        """
        self.age = t - self.t_spawn

        # Gaussian lifecycle envelope make sure above DEAD_THRESH for at least 90% of life
        self.envelope = np.exp(
            -0.5 * ((self.age - self._mu) * self._inv_sigma) ** 2
        )

        # Kill once envelope falls below threshold after peak
        if self.age > self._mu and self.envelope < DEAD_THRESH:
            self.envelope = 0.0
            self.is_alive = False
            return False

        self.wind = ambient_wind

        # wind drift: centre translates with ambient wind since spawn
        self.center = self.pos + ambient_wind * self.age

        return True

    # ── spatial query ───────────────────────────────────────────────────────

    def query(self, x: float, y: float, z: float) -> float:
        """
        Vertical wind speed [m/s] at position (x, y, z).

        Positive = updraft, negative = downdraft (Gedeon ring).
        Returns 0 if thermal is dead or above CBL.
        """
        if not self.is_alive or z <= 0 or z >= self.z_i:
            return 0.0

        # Wind lean: axis tilts with altitude
        # centre shifts horizontally proportional to z
        w_mean = max(self.w_max * self.envelope, 0.1)   # avoid div/0
        center_z = self.center + (z / w_mean) * self.wind * 0.1  # 0.1 is a tuning factor for lean strength

        # Horizontal distance from (possibly leaned) axis
        d = np.sqrt((x - center_z[0]) ** 2 + (y - center_z[1]) ** 2)
        r = d / self.R   # normalised radius

        # Lenschow vertical profile — peaks at z = z_i
        z_n  = z / self.z_i
        f_z  = z_n * np.exp(1.0 - z_n)

        w_peak = self.w_max * self.envelope

        if r <= 1.0:
            # updraft core — Bencatel / Allen profile
            w = w_peak * (1.0 - r ** 2) ** 2 * f_z

        elif r <= 2.0:
            # Gedeon downdraft ring
            w_sink = w_peak * SINK_RATIO
            w = -w_sink * (r - 1.0) * (2.0 - r) * f_z

        else:
            w = 0.0

        return float(w)

    # ── vectorised query (numpy arrays) ────────────────────────────────────

    def query_grid(self, X: np.ndarray, Y: np.ndarray,
                   z: float) -> np.ndarray:
        """
        Evaluate vertical wind over a 2-D grid at fixed altitude z.

        Parameters
        ----------
        X, Y : np.ndarray  meshgrid arrays, same shape
        z    : float       altitude [m]

        Returns
        -------
        W : np.ndarray  same shape as X, vertical wind [m/s]
        """
        if not self.is_alive or z <= 0 or z >= self.z_i:
            return np.zeros_like(X)

        w_mean = max(self.w_max * self.envelope, 0.1)
        cx_z   = self.center[0] + (z / w_mean) * self.wind[0] * 0.1
        cy_z   = self.center[1] + (z / w_mean) * self.wind[1] * 0.1

        D2 = (X - cx_z) ** 2 + (Y - cy_z) ** 2
        R2 = D2 / (self.R ** 2)

        z_n   = z / self.z_i
        f_z   = z_n * np.exp(1.0 - z_n)
        w_peak = self.w_max * self.envelope

        W = np.zeros_like(X)

        core = R2 <= 1.0
        W[core] = w_peak * (1.0 - R2[core]) ** 2 * f_z

        ring = (R2 > 1.0) & (R2 <= 4.0)
        w_sink = w_peak * SINK_RATIO
        r_ring = np.sqrt(R2[ring])
        W[ring] = -w_sink * (r_ring - 1.0) * (2.0 - r_ring) * f_z

        return W

    # ── repr ────────────────────────────────────────────────────────────────

    def __repr__(self):
        status = 'alive' if self.is_alive else 'dead'
        return (f'ThermalState({status}  age={self.age:.0f}s'
                f'  w_max={self.w_max:.1f}m/s'
                f'  R={self.R:.0f}m'
                f'  T_life={self.T_life:.0f}s'
                f'  envelope={self.envelope:.2f}'
                f'  centre=({self.center[0]:.0f},{self.center[1]:.0f}))')