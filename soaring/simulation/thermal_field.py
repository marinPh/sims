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
from thermal_model import ThermalState


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

    # ── update ──────────────────────────────────────────────────────────────

    def update(self, t: float,wind:np.ndarray = np.zeros(2)):
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
        self.last_spawned += dt
        
        self.t = t
        
        for d in range(dt):
            sources  = self.rng.random(self.pmap['p_spawn'].shape) < (self.pmap['p_spawn'] * self.spawn_rate)
            rows, cols = np.where(sources)
            for r, c in zip(rows, cols):
                x, y  = rc_xy(self.pmap['transform'], int(r), int(c))
                p_val = float(self.pmap['p'][r, c])

                th = ThermalState(
                    pos    = np.ndarray([x, y]),
                    p_val   = p_val,
                    t_spawn = self.last_spawned + d,
                    wind    = wind,
                    z_i     = self.z_i,
                    rng     = self.rng,
                )
                self.thermals.append(th)
                self.n_spawned += 1

            
        for th in self.thermals:
            th.update(t, wind)
            
        dead_count = sum(not th.is_alive for th in self.thermals)
        self.n_died += dead_count
        self.thermals = [th for th in self.thermals if th.is_alive]

        

    # ── query ────────────────────────────────────────────────────────────────

    def query(self, x: float, y: float, z: float) -> float:
        """
        Total vertical wind speed [m/s] at point (x, y, z).
        Sums contributions from all active thermals.
        """
        return sum(th.query(x, y, z) for th in self.thermals)

    def query_grid(self, X: np.ndarray, Y: np.ndarray,
                   z: float) -> np.ndarray:
        """
        Total vertical wind over a 2-D meshgrid at altitude z.

        Parameters
        ----------
        X, Y : np.ndarray  meshgrid arrays (same shape)
        z    : float       altitude [m]

        Returns
        -------
        W : np.ndarray  vertical wind [m/s], same shape as X
        """
        W = np.zeros_like(X, dtype=float)
        for th in self.thermals:
            W += th.query_grid(X, Y, z)
        return W

    # ── status ───────────────────────────────────────────────────────────────

    def status(self) -> str:
        return (f'ThermalField  t={self.t:.0f}s'
                f'  active={len(self.thermals)}'
                f'  spawned={self.n_spawned}'
                f'  died={self.n_died}')

    def __repr__(self):
        return self.status()