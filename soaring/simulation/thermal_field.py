# simulation/thermal_field.py
"""
ThermalField — owns the list of active ThermalState objects.

At each timestep:
  1. field.update(t, pmap, wind)  — ages existing thermals, removes dead
                                    ones, spawns new ones from pmap
  2. field.query(x, y, z)        — returns total vertical wind at a point
  3. field.query_grid(X, Y, z)   — returns 2-D lift slice over a meshgrid

Spawn logic
-----------
  At every call to update(), a new sources mask is drawn:
      sources = rng.random(pmap['p_spawn'].shape) < pmap['p_spawn'] * spawn_rate
  Every True cell spawns one ThermalState.  spawn_rate controls density.

Usage
-----
    import numpy as np
    from probability_map import load_tif
    from thermal_field import ThermalField

    pmap  = load_tif('data/probability_maps/val_de_ruz_may_avg.tif')
    field = ThermalField(pmap, z_i=1200.0, spawn_rate=1e-4, rng=np.random.default_rng(0))

    for t in np.arange(0, 3600, dt):
        field.update(t, wind=(2.0, 0.5))
        w = field.query(x_uav, y_uav, z_uav)
"""

import numpy as np
from rasterio.transform import xy as rc_xy
from thermal_model import ThermalState, SINK_RATIO


class ThermalField:
    """
    Parameters
    ----------
    pmap       : dict from probability_map.load_tif()
    z_i        : float  convective boundary layer height [m]
    spawn_rate : float  scales p_spawn before Bernoulli draw —
                        lower  → fewer thermals per tick
                        higher → denser field
    rng        : np.random.Generator
    """

    def __init__(self, pmap: dict, z_i: float = 1000.0,
                 spawn_rate: float = 1e-4,
                 rng: np.random.Generator = None):

        self.pmap       = pmap
        self.z_i        = z_i
        self.spawn_rate = spawn_rate
        self.rng        = rng or np.random.default_rng(42)

        self.thermals: list[ThermalState] = []   # active thermal list
        self.t        = 0.0
        self.n_spawned  = 0
        self.n_died     = 0
        self.last_spawned:float = 0

        # pre-compute pixel-centre coordinates for fast spawn lookups
        _h, _w = pmap['p'].shape
        _rows = np.arange(_h)
        _cols = np.arange(_w)
        _cc, _rr = np.meshgrid(_cols, _rows)  # col-major meshgrid
        _xs, _ys = rc_xy(pmap['transform'], _rr.ravel(), _cc.ravel(), offset='center')
        self._coord_x = np.array(_xs).reshape(_h, _w)
        self._coord_y = np.array(_ys).reshape(_h, _w)

    # ── update ──────────────────────────────────────────────────────────────

    def update(self, t: float, wind:np.ndarray = np.zeros(2)):
        """
        Advance the field to simulation time t.

        1. Update + reap dead thermals.
        2. Draw spawn mask from pmap, spawn new thermals.

        Parameters
        ----------
        t    : float  current simulation time [s]
        wind : (u,v)  ambient wind vector [m/s]
        """


        # ── 1. age existing thermals, remove dead ────────────────────────

        dt:int = int(t - self.last_spawned)
        self.t = t
        self.last_spawned = t   # advance to current time
        t_prev = self.last_spawned - dt   # start of this time window
        for d in range(dt):
            # use pmap['p'] (raw probability in [0,1]) for Bernoulli trials,
            # not p_spawn which is a PDF summing to 1 and gives near-zero cell probs
            sources  = self.rng.random(self.pmap['p'].shape) < (self.pmap['p'] * self.spawn_rate)
            rows, cols = np.where(sources)
            for r, c in zip(rows, cols):
                x, y = float(self._coord_x[r, c]), float(self._coord_y[r, c])
                p_val = float(self.pmap['p'][r, c])

                th = ThermalState(
                    pos    = np.array([x, y]),
                    p_val   = p_val,
                    t_spawn = t_prev + d,
                    wind    = wind,
                    z_i     = self.z_i,
                    rng     = self.rng,
                )
                self.thermals.append(th)
                self.n_spawned += 1

        alive = []
        dead_count = 0
        for th in self.thermals:
            if th.update(t, wind):
                alive.append(th)
            else:
                dead_count += 1
        self.thermals = alive
        self.n_died += dead_count

    # ── query ────────────────────────────────────────────────────────────────

    def query(self, x: float, y: float, z: float) -> float:
        """
        Total vertical wind speed [m/s] at point (x, y, z).
        Sums contributions from all active thermals.
        """
        return sum(th.query(x, y, z) for th in self.thermals)

    def query_grid(self, X: np.ndarray, Y: np.ndarray, z: float) -> np.ndarray:
        """
        Total vertical wind over a 2-D meshgrid at altitude z.
        Vectorized across all N active thermals in a single (N, H, W) pass.
        """
        if not self.thermals or z <= 0 or z >= self.z_i:
            return np.zeros_like(X, dtype=float)

        # ── spatial culling: skip thermals whose 2R zone misses the grid ──
        x_min, x_max = X.min(), X.max()
        y_min, y_max = Y.min(), Y.max()

        active = [
            th for th in self.thermals
            if (th.center[0] + 2*th.R >= x_min and th.center[0] - 2*th.R <= x_max and
                th.center[1] + 2*th.R >= y_min and th.center[1] - 2*th.R <= y_max)
        ]
        if not active:
            return np.zeros_like(X, dtype=float)

        N = len(active)

        # --- stack per-thermal parameters as (N,1,1) arrays for broadcasting ---
        centers   = np.array([th.center   for th in active])   # (N, 2)
        radii     = np.array([th.R        for th in active])   # (N,)
        w_maxes   = np.array([th.w_max    for th in active])   # (N,)
        envelopes = np.array([th.envelope for th in active])   # (N,)
        winds     = np.array([th.wind     for th in active])   # (N, 2)

        cx = centers[:, 0].reshape(N, 1, 1)   # (N,1,1)
        cy = centers[:, 1].reshape(N, 1, 1)   # (N,1,1)
        R  = radii.reshape(N, 1, 1)           # (N,1,1)

        w_peak = (w_maxes * envelopes).reshape(N, 1, 1)   # (N,1,1)

        # wind lean: axis tilts with altitude
        w_mean  = np.maximum(w_maxes * envelopes, 0.1)    # (N,)
        lean    = (z / w_mean * 0.1).reshape(N, 1, 1)     # (N,1,1)
        cx_z    = cx + lean * winds[:, 0].reshape(N, 1, 1)
        cy_z    = cy + lean * winds[:, 1].reshape(N, 1, 1)

        # squared normalised radius (N, H, W)
        D2 = (X[np.newaxis] - cx_z) ** 2 + (Y[np.newaxis] - cy_z) ** 2
        R2 = D2 / (R ** 2)

        # Lenschow vertical profile (scalar)
        z_n = z / self.z_i
        f_z = z_n * np.exp(1.0 - z_n)

        # allocate output
        W = np.zeros((N, *X.shape), dtype=float)

        # expand w_peak to (N, H, W) for masked indexing
        w_peak_full = np.broadcast_to(w_peak, (N, *X.shape))

        # updraft core: r ≤ 1  →  r² ≤ 1
        core = R2 <= 1.0
        W[core] = w_peak_full[core] * (1.0 - R2[core]) ** 2 * f_z

        # downdraft ring: 1 < r ≤ 2  →  1 < r² ≤ 4
        ring = (R2 > 1.0) & (R2 <= 4.0)
        r_ring = np.sqrt(R2[ring])
        w_sink = w_peak_full[ring] * SINK_RATIO
        W[ring] = -w_sink * (r_ring - 1.0) * (2.0 - r_ring) * f_z

        return W.sum(axis=0)

    # ── snapshot ─────────────────────────────────────────────────────────────

    def snapshot(self) -> list:
        """Return a list of dicts describing all active thermals."""
        return [
            {
                "cx":       float(th.center[0]),
                "cy":       float(th.center[1]),
                "R":        float(th.R),
                "envelope": float(th.envelope),
            }
            for th in self.thermals
        ]

    # ── status ───────────────────────────────────────────────────────────────

    def status(self) -> str:
        return (f'ThermalField  t={self.t:.0f}s'
                f'  active={len(self.thermals)}'
                f'  spawned={self.n_spawned}'
                f'  died={self.n_died}')

    def __repr__(self):
        return self.status()
