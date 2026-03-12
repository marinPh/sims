# simulation/animate_field.py
"""
Animate the thermal field over time and save as MP4.

Usage
-----
    python3 animate_field.py

Output
------
    ../data/animations/thermal_field.mp4
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, '.')
from probability_map import load_tif
from thermal_field import ThermalField

# ── simulation parameters ──────────────────────────────────────────────────
TIFF_PATH   = '../data/probability_maps/val_de_ruz_may_avg.tif'
OUT_PATH    = '../data/animations/thermal_field.mp4'
WIND        = (2.0, 0.5)     # (u, v) m/s
Z_I         = 1200.0         # CBL height [m]
SPAWN_RATE  = 0.005           # tune for desired thermal density
DT          = 30.0           # simulation timestep [s]
T_END       = 3600.0         # total simulation duration [s]
FPS         = 5              # animation frames per second
DPI         = 100
QUERY_Z     = 100.0          # altitude slice for updraft heatmap [m AGL]
GRID_RES    = 60             # grid resolution (cells per axis)

# ── setup ──────────────────────────────────────────────────────────────────
import os
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

print('Loading probability map...')
pmap  = load_tif(TIFF_PATH)
rng   = np.random.default_rng(42)
field = ThermalField(pmap, z_i=Z_I, spawn_rate=SPAWN_RATE, rng=rng)

b   = pmap['bounds']
ext = [b['x_min'], b['x_max'], b['y_min'], b['y_max']]

# fixed meshgrid for updraft heatmap (reused every frame)
_gx = np.linspace(b['x_min'], b['x_max'], GRID_RES)
_gy = np.linspace(b['y_min'], b['y_max'], GRID_RES)
GX, GY = np.meshgrid(_gx, _gy)

# ── figure setup ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor('#0e0e0e')
ax.set_facecolor('#0e0e0e')

# static background — probability map
ax.imshow(pmap['p'], origin='upper', extent=ext,
          cmap='Greens', vmin=0, vmax=1, alpha=0.4, zorder=0)

# updraft heatmap (w_z evaluated on grid, updated each frame)
_wz_init    = np.zeros((GRID_RES, GRID_RES))
wz_img      = ax.imshow(
    _wz_init, origin='lower', extent=ext,
    cmap='RdBu_r', vmin=-3.0, vmax=3.0, alpha=0.6, zorder=1,
)
cbar_wz = plt.colorbar(wz_img, ax=ax, label=f'Updraft w_z [m/s]  (z={QUERY_Z:.0f} m)', shrink=0.8)
cbar_wz.ax.yaxis.label.set_color('white')
cbar_wz.ax.tick_params(colors='white')

# colourmap for thermal circles (lifecycle envelope 0→1)
cmap_thermal = plt.cm.plasma
norm_env     = Normalize(vmin=0, vmax=1)

ax.set_xlabel('Easting [m]',  color='white')
ax.set_ylabel('Northing [m]', color='white')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#444444')

title = ax.set_title('', color='white', fontsize=11)

# ── animation ──────────────────────────────────────────────────────────────
# persistent batch artists — updated each frame without add/remove per patch
core_col = PatchCollection([], zorder=2)
ring_col = PatchCollection([], zorder=2)
ax.add_collection(core_col)
ax.add_collection(ring_col)
dot_scatter = ax.scatter([], [], s=9, zorder=3)

def _draw_frame(t):
    # advance simulation
    field.update(t, wind=WIND)

    # update updraft heatmap
    W = field.query_grid(GX, GY, QUERY_Z)
    wz_img.set_data(W)

    thermals = field.thermals
    if thermals:
        envelopes = np.array([th.envelope for th in thermals])
        colors    = cmap_thermal(norm_env(envelopes))   # (N, 4)

        cores = [mpatches.Circle(th.center, radius=th.R)        for th in thermals]
        rings = [mpatches.Circle(th.center, radius=th.R * 2)    for th in thermals]

        core_col.set_paths(cores)
        core_col.set_edgecolors(colors)
        core_col.set_facecolors(np.column_stack([colors[:, :3], np.full(len(thermals), 0.15)]))
        core_col.set_linewidths(1.5)

        ring_col.set_paths(rings)
        ring_col.set_edgecolors(np.column_stack([colors[:, :3], np.full(len(thermals), 0.3)]))
        ring_col.set_facecolors('none')
        ring_col.set_linewidths(0.6)
        ring_col.set_linestyles('--')

        centers = np.array([th.center for th in thermals])
        dot_scatter.set_offsets(centers)
        dot_scatter.set_color(colors)
    else:
        core_col.set_paths([])
        ring_col.set_paths([])
        dot_scatter.set_offsets(np.empty((0, 2)))

    title.set_text(
        f't = {t/60:.1f} min  |  '
        f'active: {len(thermals)}  |  '
        f'spawned: {field.n_spawned}  died: {field.n_died}'
    )

    return [core_col, ring_col, dot_scatter, title, wz_img]


frames = np.arange(0, T_END + DT, DT)
n_frames = len(frames)

print(f'Rendering {n_frames} frames  ({T_END/60:.0f} min simulation)...')

anim = FuncAnimation(
    fig, _draw_frame,
    frames=frames,
    interval=1000 / FPS,
    blit=False,        # blit=True conflicts with add_patch
)

writer = FFMpegWriter(fps=FPS, metadata={'title': 'Thermal Field'}, bitrate=1200)

anim.save(OUT_PATH, writer=writer, dpi=DPI,
          progress_callback=lambda i, n: print(f'  frame {i+1}/{n}', end='\r'))

print(f'\n✓ Saved → {OUT_PATH}')