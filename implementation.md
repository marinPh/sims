# Autonomous Thermal Soaring — SITL Implementation Guide

**Project:** Time-Varying Bayesian Optimization for Fixed-Wing UAV Thermal Soaring  
**Platform:** Easy Glider 4 · Pixhawk 4 (PX4) · Raspberry Pi 5  
**Simulator:** PX4 SITL + Gazebo · pymavlink  
**Lab:** EPFL Laboratory of Intelligent Systems

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Architecture & Layers](#2-architecture--layers)
3. [Data Flow](#3-data-flow)
4. [Module Specifications](#4-module-specifications)
5. [Key Algorithms](#5-key-algorithms)
6. [Import Graph](#6-import-graph)
7. [Threading Model](#7-threading-model)
8. [Implementation Order](#8-implementation-order)
9. [Open Questions](#9-open-questions)

---

## 1. Project Structure

```
soaring/
├── config.py                    # single source of truth
├── main.py                      # entry point, control loop
├── requirements.txt
│
├── simulation/                  # ground truth — never imported by algorithm/
│   ├── __init__.py
│   ├── thermal_field.py         # Allen (2006) thermal model
│   └── wind_injector.py         # publishes wind to Gazebo at 5 Hz
│
├── algorithm/                   # the research contribution
│   ├── __init__.py
│   ├── gp_model.py              # GP surrogate + prior mean from hotspots
│   ├── tvbo.py                  # W-DBO acquisition + Wasserstein scoring
│   └── path_planner.py          # A* over GP expected_lift field
│
├── comms/                       # all PX4 / MAVLink interaction
│   ├── __init__.py
│   ├── mavlink_client.py        # shared_state dict, reader thread
│   └── waypointer.py            # send waypoints, wait_reached
│
├── utils/                       # shared helpers, no layer affiliation
│   ├── __init__.py
│   ├── coordinates.py           # GPS ↔ ENU conversion
│   └── polar.py                 # aircraft polar, w_z_obs estimation
│
├── data/                        # cross-cutting: reads from all layers
│   ├── __init__.py
│   ├── recorder.py              # CSV logger
│   └── plotter.py               # trajectory + GP field plots
│
└── scripts/                     # standalone runnable tools
    ├── monitor.py               # terminal telemetry display
    ├── inject_wind.py           # run wind injector standalone
    └── run_mission.py           # baseline A→B mission without GP
```

---

## 2. Architecture & Layers

Three distinct layers with one strict rule: **`simulation/` is never imported by `algorithm/`**. The GP learns purely from observations, not from ground truth.

```
┌──────────────────────────────────────────────────────────────────┐
│  SIMULATION LAYER  (ground truth, invisible to algorithm)         │
│  simulation/thermal_field.py → simulation/wind_injector.py       │
│                                         │                         │
│                               gz topic /world/default/wind        │
└─────────────────────────────────────────┼────────────────────────┘
                                          │ Gazebo physics
                                 ┌────────▼──────────────────────┐
                                 │  PX4 SITL                      │
                                 └────────┬──────────────────────┘
                                          │ MAVLink UDP:14550
┌─────────────────────────────────────────▼────────────────────────┐
│  SENSING LAYER                                                     │
│  comms/mavlink_client.py → shared_state {}                        │
└─────────────────────────────────────────┬────────────────────────┘
                                          │
┌─────────────────────────────────────────▼────────────────────────┐
│  ALGORITHM LAYER   (the research contribution)                     │
│  algorithm/gp_model.py                                            │
│       → algorithm/tvbo.py                                         │
│       → algorithm/path_planner.py                                 │
│       → comms/waypointer.py                                       │
└──────────────────────────────────────────────────────────────────┘

Cross-cutting:
  utils/      shared by all layers (coordinates, polar)
  data/       reads from all layers for logging and plotting
  scripts/    standalone entry points for each subsystem
```

---

## 3. Data Flow

```
PX4 SITL (Gazebo)
    │  MAVLink UDP:14550
    ▼
comms/mavlink_client.py
    │  reader thread — blocking=False + 10ms yield
    ▼
shared_state {}  ←─────────────────────────────────────────────┐
    │                                                           │
    ├──► scripts/monitor.py          (2 Hz, in-place terminal)  │
    │                                                           │
    ├──► simulation/wind_injector.py (5 Hz)                     │
    │         │  utils/coordinates.gps_to_enu()                 │
    │         ▼                                                  │
    │    simulation/thermal_field.py                             │
    │         │  AllenThermal.updraft(x, y, z)                   │
    │         ▼                                                  │
    │    gz topic /world/default/wind ──► Gazebo physics ───────►│
    │                                                           │
    └──► main.py  (event-driven on waypoint reached)            │
              │                                                 │
              │  w_z_obs = utils/polar.estimate_lift(climb, IAS)│
              │  x,y,z   = utils/coordinates.gps_to_enu(...)    │
              ▼                                                 │
         algorithm/tvbo.py                                      │
              │  .update(x, y, z, t, w_z_obs)                   │
              │       → gp_model.observe() + fit(weights)        │
              │  .suggest_next(current_pos, goal)                │
              │       → path_planner.plan()                      │
              │            → gp_model.expected_lift(x,y,z)       │
              ▼                                                 │
         comms/waypointer.py                                    │
              │  .send_next(lat, lon, alt) ────────────────────►│
              │  .wait_reached(...)                              │
              ▼                                                 │
         data/recorder.py                                       │
              │  w_z_injected ← simulation/thermal_field        │
              │  .log(t, pos, w_z_obs, w_z_injected,            │
              │       gp_mean, gp_var)                           │
              ▼
         data/plotter.py  (on exit)
```

---

## 4. Module Specifications

### 4.1 `config.py`

Single source of truth. All other modules import from here. No logic — data only.

```python
ORIGIN_LAT = 47.3977          # must match PX4_HOME_LAT in SITL
ORIGIN_LON  = 8.5456

POINT_A = (0.0,   0.0,   80.0)   # ENU metres from origin
POINT_B = (800.0, 400.0, 80.0)

# Ground truth — simulation/ only, never read by algorithm/
THERMALS = [
    dict(cx=150, cy=100, w_max=3.5, R=80,  V_e=0.5, z_max=1200, z_base=50),
    dict(cx=500, cy=250, w_max=2.8, R=100, V_e=0.4, z_max=1000, z_base=30),
    dict(cx=700, cy=150, w_max=3.2, R=70,  V_e=0.6, z_max=1100, z_base=40),
]
WIND_BG = dict(x=2.0, y=0.5)

# 2D geographical survey — used to initialise GP mean function
PRIOR_HOTSPOTS = [
    (160, 110, 1.0),   # (x_enu, y_enu, strength)
    (490, 260, 0.8),
    (710, 140, 0.9),
]

GRID_RESOLUTION = 20.0        # metres between A* nodes
GRID_Z_MIN      = 40.0
GRID_Z_MAX      = 200.0
GRID_Z_STEP     = 20.0        # altitude layers for A*

CRUISE_ALT      = 80.0        # metres AGL
ACCEPTANCE_R    = 25.0        # waypoint reached threshold (metres)
CRUISE_AIRSPEED = 14.0        # m/s

# Easy Glider 4 polar: (airspeed m/s, sink_rate m/s)
POLAR = [(10, -0.8), (12, -0.65), (14, -0.7), (16, -0.85), (20, -1.2)]

PX4_PARAMS = {
    'NAV_RCL_ACT': 0, 'COM_LOW_BAT_ACT': 0, 'GF_ACTION': 0,
    'COM_DL_LOSS_T': 60, 'FW_LND_ANG': 20,
    'BAT_CRIT_THR': 0, 'BAT_EMERGEN_THR': 0,
}

GAZEBO_WORLD = 'default'
```

---

### 4.2 `comms/mavlink_client.py`

Owns the MAVLink connection and `shared_state`. **No other module calls `recv_match`.**

**Exports:** `connect()`, `shared_state`, `shared_state_lock`, `start_reader()`, `start_heartbeat()`, `set_mode()`, `set_param()`, `arm()`

**`shared_state` keys:**

| Key | MAVLink source | Unit |
|-----|---------------|------|
| `lat`, `lon` | GLOBAL_POSITION_INT | degrees |
| `alt` | GLOBAL_POSITION_INT | m AGL |
| `alt_msl` | GLOBAL_POSITION_INT | m MSL |
| `fix`, `sats`, `hdop` | GPS_RAW_INT | —, count, m |
| `airspeed`, `groundspeed` | VFR_HUD | m/s |
| `heading`, `climb`, `throttle` | VFR_HUD | deg, m/s, % |
| `voltage`, `current`, `battery_pct` | SYS_STATUS | V, A, % |
| `wind_dir`, `wind_spd`, `wind_z` | WIND | deg, m/s, m/s |
| `armed`, `mode` | HEARTBEAT | string |
| `health` | SYS_STATUS | list[str] |

Reader thread uses `recv_match(blocking=False)` + `time.sleep(0.01)`. Never stalls.

---

### 4.3 `comms/waypointer.py`

**`upload_mission(master, waypoints, home_lat, home_lon)`**
Full mission: `TAKEOFF → WPs → LOITER_TO_ALT → LAND`. Used by `run_mission.py`.

**`send_next(master, lat, lon, alt)`**
Single `MAV_CMD_NAV_WAYPOINT`. Called by `main.py` after each TV-BO suggestion.

**`wait_reached(shared_state, lat, lon, alt, radius_m, timeout) -> bool`**
Polls `shared_state` every 0.5s via Haversine distance. Non-blocking on reader thread.

---

### 4.4 `simulation/thermal_field.py`

Allen (2006) model. See [Section 5.1](#51-allen-2006-thermal-model).

**`AllenThermal(cx, cy, w_max, R, V_e, z_max, z_base)`**
- `.updraft(x, y, z) -> float` — m/s, positive = upward

**`ThermalField(thermals, wind_bg_x, wind_bg_y)`**
- `.get_wind(x, y, z) -> (wx, wy, wz)`
- `.from_config() -> ThermalField`

> ⚠️ Never imported by `algorithm/`. GP learns from observations only.

---

### 4.5 `simulation/wind_injector.py`

Daemon thread at 5 Hz. Reads `shared_state` → `ThermalField.get_wind()` → `gz topic`.

**`start(shared_state, thermal_field, stop_event) -> Thread`**

```bash
gz topic -l | grep wind          # verify topic name
gz topic -e -t /world/default/wind   # verify injection
```

---

### 4.6 `algorithm/gp_model.py`

GP surrogate. See [Section 5.2](#52-gp-kernel-design) for kernel.

**`GPModel(prior_hotspots)`**
- Prior mean: `m(x,y) = Σ strength_i · exp(−||pos − hotspot_i||² / 2l²)`
- Kernel: Matérn-5/2 spatial ⊗ Matérn-3/2 temporal

**`.observe(x, y, z, t, w_z)`** — appends to dataset

**`.fit(weights=None)`** — refits `[l_S, l_T, σ_f, σ_n]` by marginal likelihood

**`.predict(x, y, z) -> (mu, sigma2)`**

**`.expected_lift(x, y, z) -> float`** — TV-BO acquisition, consumed by `path_planner`

---

### 4.7 `algorithm/tvbo.py`

W-DBO wrapper around `GPModel`. See [Section 5.4](#54-w-dbo-relevancy-scoring).

**`TVBO(gp_model)`**

**`.update(x, y, z, t, w_z)`**
- Computes Wasserstein relevancy weights for all past observations
- Calls `gp_model.observe()` + `gp_model.fit(weights)`

**`.suggest_next(current_pos_enu, goal_enu) -> (x, y, z)`**
- Delegates to `path_planner.plan()`

Static baseline: all weights = 1.0 → standard GP-BO, no time-variation.

---

### 4.8 `algorithm/path_planner.py`

A* over `gp_model.expected_lift`. See [Section 5.5](#55-a-on-gp-field).

**`plan(start_enu, goal_enu, gp_model) -> list[(x,y,z)]`**

Edge cost: `distance(u→v) − λ · expected_lift(v)`

Replanning triggered after every `tvbo.update()`.

---

### 4.9 `utils/coordinates.py`

Flat-earth GPS ↔ ENU. Valid within ~2 km of `ORIGIN`.

- `gps_to_enu(lat, lon, alt) -> (x, y, z)`
- `enu_to_gps(x, y, z) -> (lat, lon, alt)`
- `haversine(lat1, lon1, lat2, lon2) -> float` — metres

Imported by: `wind_injector`, `waypointer`, `path_planner`, `recorder`, `monitor`.

---

### 4.10 `utils/polar.py`

- `polar_sink(airspeed) -> float` — interpolates `config.POLAR`, returns sink m/s
- `estimate_lift(climb_rate, airspeed) -> float`
  - `w_z_obs = climb_rate − polar_sink(airspeed)`
  - Positive = climbing faster than polar predicts = updraft present

---

### 4.11 `data/recorder.py`

One row per observation (per waypoint reached).

| Column | Description |
|--------|-------------|
| `t` | seconds from flight start |
| `lat`, `lon`, `alt` | GPS position |
| `x`, `y`, `z` | ENU metres |
| `w_z_obs` | observed lift from polar estimation |
| `w_z_injected` | ground truth from `ThermalField` |
| `gp_mean` | GP posterior mean at observation |
| `gp_var` | GP posterior variance |
| `gp_expected_lift` | TV-BO acquisition value |

**`Recorder`:** `.log(obs_dict)`, `.save_csv(path)`, `.get_dataframe()`

---

### 4.12 `data/plotter.py`

Called by `main.py` on exit.

- **`plot_trajectory_3d(df, thermals)`** — 3D path coloured by `w_z_obs`, thermal cylinders
- **`plot_topdown_gp(df, gp_model, thermals, prior_hotspots)`** — GP mean heatmap + trajectory + centers
- **`plot_timeseries(df)`** — `w_z_obs` vs `w_z_injected` vs `gp_mean` over time

---

### 4.13 `scripts/monitor.py`

Standalone terminal monitor. In-place display via cursor-up codes, 2 Hz.

Sections: STATUS · GPS · AIRSPEED · BATTERY · WIND (PX4) · THERMAL (injected) · GP (expected lift) · NEXT WP

```bash
python3 scripts/monitor.py
```

---

### 4.14 `scripts/inject_wind.py`

Standalone wind injector for testing Gazebo physics independently.

```bash
python3 scripts/inject_wind.py
```

---

### 4.15 `scripts/run_mission.py`

Baseline A→B flight without GP. Fixed waypoint sequence. Used to:
- Validate the full PX4/Gazebo/pymavlink stack
- Record ground truth trajectory
- Establish baseline cumulative thermal time

```bash
python3 scripts/run_mission.py
```

---

### 4.16 `main.py`

**Startup:**
```
connect → set PX4_PARAMS → start threads (reader, heartbeat, wind_injector, monitor)
→ init GPModel(prior_hotspots) → init TVBO → upload_mission(A→B) → arm
```

**Control loop:**
```
while not reached(B):
    wait_reached(current_target)
    w_z  = polar.estimate_lift(climb, airspeed)
    pos  = coordinates.gps_to_enu(lat, lon, alt)
    tvbo.update(*pos, t, w_z)
    recorder.log(pos, w_z, thermal_field.get_wind(*pos), gp_model.predict(*pos))
    next_wp = tvbo.suggest_next(pos, POINT_B)
    waypointer.send_next(*coordinates.enu_to_gps(next_wp))

recorder.save_csv(...)
plotter.plot_trajectory_3d(...)
plotter.plot_topdown_gp(...)
plotter.plot_timeseries(...)
stop_event.set()
```

---

## 5. Key Algorithms

### 5.1 Allen (2006) Thermal Model

```
         ⎧ w_max · (d/R) · exp(1 − d/R)    d ≤ 2R   (updraft core)
w(d) =   ⎨
         ⎩ −V_e                             d > 2R   (environmental sink)
```

Peak updraft at `d = R` — ring structure, not a point maximum.

Vertical taper (top 20% of thermal height):
```
scale = (z_max − z) / (0.2 · (z_max − z_base))    near top
scale = 0.0                                         outside [z_base, z_max]
```

---

### 5.2 GP Kernel Design

Separable spatio-temporal kernel (Lawrance 2011):

```
k((x,t), (x′,t′)) = k_S(x, x′) · k_T(t, t′)
```

- `k_S` Matérn-5/2: twice differentiable, appropriate for smooth thermal fields
- `k_T` Matérn-3/2: once differentiable, appropriate for thermal evolution

Hyperparameters `[l_S, l_T, σ_f, σ_n]` fit by marginal likelihood maximisation.

Prior mean from geographical survey:
```
m(x,y) = Σ_i  strength_i · exp( −||[x,y] − hotspot_i||² / 2·l_prior² )
```

---

### 5.3 TV-BO Acquisition Function

```
α(x,y,z) = μ(x,y,z) · Φ( μ(x,y,z) / σ(x,y,z) )
```

`Φ` = standard normal CDF. Equivalent to Expected Improvement at zero threshold. High where GP is confident and predicts strong updraft. `path_planner` maximises `α` along the route to B.

---

### 5.4 W-DBO Relevancy Scoring

Bardou et al. (NeurIPS 2024). Downweights stale observations as the thermal field evolves:

```
w_i = exp( −γ · W₂(P_current, P_at_t_i) )
```

`W₂` = 2-Wasserstein distance between Gaussians (closed form). Low `w_i` = misleading old observation.

**Static baseline:** all `w_i = 1.0` → standard GP-BO. Control condition.

---

### 5.5 A\* on GP Field

```
f(n) = g(n) + h(n)
g(n) = Σ [ distance(prev→n) − λ · α(n) ]    accumulated cost
h(n) = Euclidean distance to goal             admissible heuristic
```

| `λ` | Behaviour |
|-----|-----------|
| `0` | Straight A→B, no thermal seeking |
| Low | Mild preference for high-lift regions |
| High | Aggressive detour into thermals |

---

## 6. Import Graph

```
config.py  ←── imported by all modules below

utils/coordinates.py   ←── config
utils/polar.py         ←── config

simulation/thermal_field.py    ←── config, utils/coordinates
simulation/wind_injector.py    ←── simulation/thermal_field,
                                    comms/mavlink_client, utils/coordinates

comms/mavlink_client.py   ←── config
comms/waypointer.py       ←── comms/mavlink_client, utils/coordinates, config

algorithm/gp_model.py     ←── config, utils/coordinates
algorithm/tvbo.py         ←── algorithm/gp_model, config
algorithm/path_planner.py ←── algorithm/gp_model, utils/coordinates, config

data/recorder.py   ←── simulation/thermal_field, utils/coordinates, config
data/plotter.py    ←── data/recorder

scripts/monitor.py      ←── comms/mavlink_client, simulation/thermal_field,
                              algorithm/gp_model, utils/coordinates
scripts/inject_wind.py  ←── simulation/wind_injector, comms/mavlink_client
scripts/run_mission.py  ←── comms/mavlink_client, comms/waypointer,
                              data/recorder, data/plotter, config

main.py  ←── everything

!! algorithm/ never imports simulation/ !!
```

---

## 7. Threading Model

| Thread | Module | Rate | Mechanism |
|--------|--------|------|-----------|
| `reader` | comms/mavlink_client | ~100 Hz | `blocking=False` + 10ms sleep |
| `heartbeat` | comms/mavlink_client | 1 Hz | `time.sleep(1.0)` |
| `wind_injector` | simulation/wind_injector | 5 Hz | `time.sleep(0.2)` |
| `monitor` | scripts/monitor | 2 Hz | `time.sleep(0.5)` |
| `main` loop | main | event-driven | `wait_reached()` polling |

All share `shared_state` via `threading.Lock`. Lock held only for dict access — never during computation or I/O. `stop_event = threading.Event()` passed to all threads, set on `KeyboardInterrupt`.

---

## 8. Implementation Order

| Step | Module | Validation test |
|------|--------|----------------|
| 1 | `config.py` | Import and print — no errors |
| 2 | `utils/coordinates.py` | Round-trip GPS→ENU→GPS within 1mm |
| 3 | `utils/polar.py` | `estimate_lift(0.5, 14.0)` ≈ 1.2 m/s |
| 4 | `comms/mavlink_client.py` | `shared_state` populates within 5s of SITL connect |
| 5 | `simulation/thermal_field.py` | Allen profile plot — peak at `d=R`, sink outside `2R` |
| 6 | `simulation/wind_injector.py` | `gz topic -e` confirms `wz > 0` over thermal centre |
| 7 | `scripts/monitor.py` | In-place display, no stalling over 60s |
| 8 | `comms/waypointer.py` | Aircraft reaches single test waypoint within timeout |
| 9 | `scripts/run_mission.py` | Full A→B baseline flight completes and lands |
| 10 | `data/recorder.py` | CSV written with correct schema |
| 11 | `data/plotter.py` | 3 plots render without error on baseline CSV |
| 12 | `algorithm/gp_model.py` | Prior mean matches hotspot locations; posterior updates on 3 synthetic obs |
| 13 | `algorithm/tvbo.py` | `weights=1.0` for static case; `suggest_next()` points toward hotspot |
| 14 | `algorithm/path_planner.py` | `λ=0` → straight line; high `λ` → detour through hotspot |
| 15 | `main.py` | GP-guided A→B flight; plotter shows GP mean converging to thermal ground truth |

---

## 9. Open Questions

| # | Question | Affects | Priority |
|---|----------|---------|----------|
| 1 | What is `λ` (A* lift weight)? Needs empirical tuning. | path_planner | High |
| 2 | What is `l_prior` (hotspot prior length scale)? Depends on survey resolution. | gp_model | High |
| 3 | Does Gazebo wind topic affect aircraft physics or only visuals? Verify with climb rate check over static thermal. | wind_injector | High |
| 4 | Easy Glider 4 polar — is `config.POLAR` accurate or placeholder? | utils/polar | High |
| 5 | How many observations before GP fit is reliable? Cold-start sensitivity analysis needed. | gp_model, tvbo | Medium |
| 6 | `γ` in W-DBO Wasserstein weighting — how to set for static baseline? | tvbo | Medium |
| 7 | Does A* replan every observation or only when GP change exceeds a threshold? | path_planner | Medium |
| 8 | EPFL WiRE lab — can they provide better vertical thermal profiles for `config.THERMALS`? | simulation | Low |