#!/usr/bin/env python3
"""
test_runner.py — Parallel PX4+Gazebo trial runner.

Manages N_WORKERS simultaneous simulation stacks. Each worker claims
trials lazily (one at a time) from a CSV registry, runs the full stack,
and writes results to results/<algo>_<seed>/.

Usage
-----
    python3 test_runner.py \\
        --algorithms straightline.py algo_b.py algo_c.py \\
        --seeds seeds.txt \\
        --workers 2 \\
        --speed-factor 10 \\
        --px4-dir ~/PX4-Autopilot
"""

from __future__ import annotations
import argparse
import csv
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── constants ──────────────────────────────────────────────────────────────────

ROOT_DIR          = Path(__file__).parent
SIM_DIR           = ROOT_DIR / 'soaring' / 'simulation'
RESULTS           = ROOT_DIR / 'results'
LOGS              = ROOT_DIR / 'logs'

MAVLINK_BASE_PORT = 14550   # slot i → 14550 + i  (patched into px4-rc.mavlink per slot)
GZ_POLL_INTERVAL  = 2.0
GZ_WAIT_TIMEOUT   = 90      # seconds to wait for Gazebo per trial
BOOT_SLEEP        = 5.0     # wall-clock seconds after Gazebo ready
MISSION_TIMEOUT   = 3600    # wall-clock seconds max per mission

REGISTRY_COLS = [
    'algo', 'script', 'seed', 'status',
    'csv_path', 'start_time', 'end_time', 'exit_code',
]

# ── data classes ───────────────────────────────────────────────────────────────

@dataclass
class Trial:
    algo: str
    script: str
    seed: int
    status: str = 'pending'

# ── trial registry ─────────────────────────────────────────────────────────────

class TrialRegistry:
    """Thread-safe CSV registry of all (algo, seed) trials."""

    def __init__(self, path: Path):
        self.path  = Path(path)
        self._lock = threading.Lock()

    def generate(self, scripts: list[str], seeds: list[int]) -> None:
        """Write trials.csv if it does not already exist (same scripts+seeds)."""
        with self._lock:
            if self.path.exists():
                # Reuse only if the registry was built for the same inputs.
                # Different seeds file or script list → start fresh.
                existing = list(csv.DictReader(self.path.open()))
                existing_algos = {r['algo'] for r in existing}
                expected_algos = {Path(s).stem for s in scripts}
                existing_seeds = {int(r['seed']) for r in existing}
                if existing_algos == expected_algos and existing_seeds == set(seeds):
                    return
                print(f'[runner] registry mismatch — regenerating {self.path}')
                self.path.unlink()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=REGISTRY_COLS)
                w.writeheader()
                for script in scripts:
                    algo = Path(script).stem
                    for seed in seeds:
                        w.writerow(dict(
                            algo=algo, script=script, seed=seed,
                            status='pending', csv_path='',
                            start_time='', end_time='', exit_code='',
                        ))

    def _read_rows(self) -> list[dict]:
        with self.path.open(newline='') as f:
            return list(csv.DictReader(f))

    def _write_rows(self, rows: list[dict]) -> None:
        tmp = self.path.with_suffix('.tmp')
        with tmp.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=REGISTRY_COLS)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, self.path)

    def claim_next(self) -> Trial | None:
        """Atomically claim the next pending/failed trial. Returns None if none left."""
        with self._lock:
            rows = self._read_rows()
            for row in rows:
                if row['status'] in ('pending', 'failed'):
                    row['status']     = 'running'
                    row['start_time'] = datetime.now().isoformat(timespec='seconds')
                    self._write_rows(rows)
                    return Trial(
                        algo=row['algo'], script=row['script'],
                        seed=int(row['seed']), status='running',
                    )
            return None

    def reset_stale_running(self) -> int:
        """Reset any 'running' rows to 'pending' (called on startup for crash recovery).

        Returns the number of rows reset.
        """
        with self._lock:
            rows = self._read_rows()
            count = 0
            for row in rows:
                if row['status'] == 'running':
                    row['status'] = 'pending'
                    row['start_time'] = ''
                    count += 1
            if count:
                self._write_rows(rows)
                print(f'[runner] reset {count} stale running trial(s) to pending')
            return count

    def mark_done(self, trial: Trial, csv_path: str, exit_code: int) -> None:
        self._update(trial, status='done', csv_path=str(csv_path),
                     exit_code=str(exit_code))

    def mark_failed(self, trial: Trial, exit_code: int) -> None:
        self._update(trial, status='failed', exit_code=str(exit_code))

    def _update(self, trial: Trial, **kwargs) -> None:
        with self._lock:
            rows = self._read_rows()
            updated = False
            for row in rows:
                if (row['algo'] == trial.algo
                        and int(row['seed']) == trial.seed
                        and row['status'] == 'running'):
                    row.update(kwargs)
                    row['end_time'] = datetime.now().isoformat(timespec='seconds')
                    updated = True
                    break
            if not updated:
                raise RuntimeError(
                    f'No running row found for {trial.algo} seed={trial.seed}'
                )
            self._write_rows(rows)


# ── slot helpers ───────────────────────────────────────────────────────────────

def slot_mavlink_port(slot: int) -> int:
    """GCS UDP port that the mission script binds to (udp server mode).

    PX4's px4-rc.mavlink is patched per-slot to add -o <port> to mavlink start,
    so PX4 instance N sends to 14550+N.  The mission script listens there.
    Formula: 14550 + slot  (14550, 14551, 14552, …)
    """
    return MAVLINK_BASE_PORT + slot


def slot_env(slot: int, seed: int, speed_factor: int) -> dict[str, str]:
    """Build the OS environment dict for a worker slot."""
    env = os.environ.copy()
    env['PX4_INSTANCE']         = str(slot)
    env['GZ_PARTITION']         = f'slot{slot}'
    env['HEADLESS']             = '1'
    env['PX4_SIM_SPEED_FACTOR'] = str(speed_factor)
    env['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    env.setdefault('PX4_HOME_LAT', '47.0500')
    env.setdefault('PX4_HOME_LON', '6.9500')
    env.setdefault('PX4_HOME_ALT', '0')
    return env


# ── process helpers ────────────────────────────────────────────────────────────

def _start_proc(cmd: list[str], cwd: Path, env: dict,
                log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open('wb')
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), env=env,
        stdout=log_file, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    log_file.close()   # child inherited its own copy; close parent's handle
    return proc


def _kill(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    time.sleep(1.5)
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def _gz_ready(partition: str, world: str = 'default') -> bool:
    """Return True if Gazebo is publishing the world topic for this partition."""
    env = os.environ.copy()
    env['GZ_PARTITION'] = partition
    try:
        r = subprocess.run(
            ['gz', 'topic', '-l'],
            capture_output=True, timeout=5, env=env,
        )
        return f'/world/{world}'.encode() in r.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ── sim stack ──────────────────────────────────────────────────────────────────

class SimStack:
    """
    Runs one complete trial: PX4+Gazebo → gz_sim → mission → teardown.

    Returns (exit_code: int, csv_path: str | None).

    PX4 is launched directly (not via `make`) so that the `-i <slot>` flag
    can be passed — the make target hardcodes no `-i` flag and PX4 reads
    instance exclusively from that CLI arg, not from PX4_INSTANCE env var.
    """

    WORLD     = 'default'
    SIM_MODEL = 'gz_advanced_plane'

    def __init__(self, slot: int, trial: Trial,
                 px4_dir: Path, speed_factor: int):
        self.slot               = slot
        self.trial              = trial
        self.px4_dir            = px4_dir
        self.speed_factor       = speed_factor
        self.env                = slot_env(slot, trial.seed, speed_factor)
        self.port               = slot_mavlink_port(slot)
        self.partition          = f'slot{slot}'
        self.log_dir            = LOGS / f'slot{slot}' / f'{trial.algo}_{trial.seed}'
        self.result_dir         = RESULTS / f'{trial.algo}_{trial.seed}'
        self._gz_server_proc: subprocess.Popen | None = None
        self._px4_proc: subprocess.Popen | None       = None
        self._gz_proc: subprocess.Popen | None        = None
        self._mission_proc: subprocess.Popen | None   = None

    def run(self) -> tuple[int, str | None]:
        try:
            return self._run()
        except Exception as e:
            print(f'[slot{self.slot}] ERROR: {e}', flush=True)
            return 1, None
        finally:
            self._teardown()

    def _run(self) -> tuple[int, str | None]:
        tag = f'[slot{self.slot}|{self.trial.algo}|seed={self.trial.seed}]'

        # ── shared env setup ────────────────────────────────────────────────
        build_dir    = self.px4_dir / 'build' / 'px4_sitl_default'
        instance_dir = build_dir / f'instance_{self.slot}'
        instance_dir.mkdir(parents=True, exist_ok=True)

        # Build a unified env with gz_env.sh values merged in
        gz_env_sh = build_dir / 'rootfs' / 'gz_env.sh'
        base_env = self.env.copy()
        if gz_env_sh.exists():
            for line in gz_env_sh.read_text().splitlines():
                line = line.strip()
                if line.startswith('export '):
                    line = line[len('export '):]
                if '=' in line and not line.startswith('#'):
                    k, _, v = line.partition('=')
                    for ek, ev in base_env.items():
                        v = v.replace(f'${ek}', ev).replace(f'${{{ek}}}', ev)
                    base_env.setdefault(k, v)

        worlds_dir = base_env.get(
            'PX4_GZ_WORLDS',
            str(self.px4_dir / 'Tools' / 'simulation' / 'gz' / 'worlds'),
        )
        world_sdf = Path(worlds_dir) / f'{self.WORLD}.sdf'

        # ── 0. stale lock cleanup ────────────────────────────────────────────
        px4_lock = Path(f'/tmp/px4_lock-{self.slot}')
        if px4_lock.exists():
            print(f'{tag} removing stale {px4_lock}', flush=True)
            px4_lock.unlink(missing_ok=True)

        # ── 1. Gazebo server (explicit, per-slot partition) ──────────────────
        # PX4's px4-rc.gzsim detects "already running" via `gz topic -l`.
        # That check is partition-aware, but races when slots start together.
        # We launch Gazebo ourselves (PX4_GZ_STANDALONE=1 tells PX4 to skip
        # its own gz launch) so each slot has an isolated Gazebo server.
        print(f'{tag} starting Gazebo (partition={self.partition})', flush=True)
        self._gz_server_proc = _start_proc(
            ['gz', 'sim', '--verbose=1', '-r', '-s', str(world_sdf)],
            cwd=instance_dir,
            env=base_env,
            log_path=self.log_dir / 'gazebo.log',
        )

        # ── 2. Wait for Gazebo world ─────────────────────────────────────────
        deadline = time.monotonic() + GZ_WAIT_TIMEOUT
        while time.monotonic() < deadline:
            if _gz_ready(self.partition, self.WORLD):
                break
            if self._gz_server_proc.poll() is not None:
                print(f'{tag} Gazebo exited early', flush=True)
                return 2, None
            time.sleep(GZ_POLL_INTERVAL)
        else:
            print(f'{tag} Gazebo timeout', flush=True)
            return 3, None

        print(f'{tag} Gazebo ready — launching PX4', flush=True)

        # ── 3. PX4 (standalone: don't touch Gazebo, just connect) ───────────
        # Patch px4-rc.mavlink in a per-slot copy of etc/ so PX4 sends to the
        # right GCS port (-o 14550+slot).  Without -o, PX4 defaults to 14550
        # for all instances, making parallel GCS connections impossible.
        import shutil, re
        instance_etc = instance_dir / 'etc'
        shutil.copytree(str(build_dir / 'etc'), str(instance_etc),
                        dirs_exist_ok=True)
        mavlink_rc = instance_etc / 'init.d-posix' / 'px4-rc.mavlink'
        # Read from the ORIGINAL build etc (not instance_etc) so we always
        # patch a clean base regardless of previous runs.
        src_rc     = build_dir / 'etc' / 'init.d-posix' / 'px4-rc.mavlink'
        rc_text    = src_rc.read_text()
        gcs_port   = slot_mavlink_port(self.slot)
        # Strip any existing -o PORT flags from the GCS line first (idempotent),
        # then add exactly one -o <gcs_port>.
        canonical  = 'mavlink start -x -u $udp_gcs_port_local -r 4000000 -f'
        target     = f'{canonical} -o {gcs_port}'
        patched    = re.sub(
            re.escape(canonical) + r'( -o \d+)*',
            target,
            rc_text,
        )
        if patched == rc_text:
            print(f'{tag} WARN: px4-rc.mavlink patch not applied — GCS port may be wrong',
                  flush=True)
        mavlink_rc.write_text(patched)

        px4_env = base_env.copy()
        px4_env['PX4_SIM_MODEL']     = self.SIM_MODEL
        px4_env['PX4_GZ_WORLD']      = self.WORLD
        px4_env['PX4_GZ_STANDALONE'] = '1'   # skip gz launch/detection in rcS
        px4_env['GZ_IP']             = '127.0.0.1'

        self._px4_proc = _start_proc(
            [str(build_dir / 'bin' / 'px4'),
             '-i', str(self.slot),
             '-d', str(instance_etc),
             '-w', str(instance_dir)],
            cwd=instance_dir,
            env=px4_env,
            log_path=self.log_dir / 'px4.log',
        )

        # ── 4. gz_sim.py — thermal bridge (seeded per trial) ────────────────
        time.sleep(3.0)   # model loading buffer before subscribing to poses
        print(f'{tag} Gazebo ready — launching gz_sim', flush=True)
        self._gz_proc = _start_proc(
            [sys.executable, 'gz_sim.py',
             '--seed',     str(self.trial.seed),
             '--instance', str(self.slot),
             '--world',    self.WORLD],
            cwd=SIM_DIR,
            env=base_env,
            log_path=self.log_dir / 'gz_sim.log',
        )

        # ── 5. PX4 boot delay ────────────────────────────────────────────────
        boot = max(BOOT_SLEEP, 15.0 / self.speed_factor)
        print(f'{tag} waiting {boot:.1f}s for PX4 boot', flush=True)
        time.sleep(boot)

        # ── 6. Mission script ────────────────────────────────────────────────
        self.result_dir.mkdir(parents=True, exist_ok=True)
        script_path = ROOT_DIR / self.trial.script
        self._mission_proc = _start_proc(
            [sys.executable, str(script_path),
             '--port',       str(self.port),
             '--output-dir', str(self.result_dir)],
            cwd=ROOT_DIR,
            env=self.env,
            log_path=self.log_dir / 'mission.log',
        )

        # ── 7. Wait for mission to exit ──────────────────────────────────────
        deadline = time.monotonic() + MISSION_TIMEOUT
        while time.monotonic() < deadline:
            rc = self._mission_proc.poll()
            if rc is not None:
                print(f'{tag} mission exited code={rc}', flush=True)
                return rc, self._find_csv()
            time.sleep(2.0)

        print(f'{tag} mission timed out', flush=True)
        return 4, None

    def _find_csv(self) -> str | None:
        csvs = sorted(self.result_dir.glob('flight_*.csv'))
        return str(csvs[-1]) if csvs else None

    def _teardown(self) -> None:
        for proc in (self._mission_proc, self._gz_proc, self._px4_proc,
                     self._gz_server_proc):
            _kill(proc)


# ── worker function ────────────────────────────────────────────────────────────

def run_worker(slot: int, registry: TrialRegistry,
               px4_dir: Path, speed_factor: int) -> None:
    """Run trials on a fixed slot until the registry is exhausted."""
    while True:
        trial = registry.claim_next()
        if trial is None:
            break
        stack = SimStack(slot, trial, px4_dir, speed_factor)
        exit_code, csv_path = stack.run()
        if exit_code == 0 and csv_path:
            registry.mark_done(trial, csv_path=csv_path, exit_code=exit_code)
            print(f'[runner] DONE  {trial.algo} seed={trial.seed}', flush=True)
        else:
            registry.mark_failed(trial, exit_code=exit_code)
            print(f'[runner] FAIL  {trial.algo} seed={trial.seed} '
                  f'rc={exit_code}', flush=True)


# ── plot ───────────────────────────────────────────────────────────────────────

DEFAULT_TIF = (ROOT_DIR / 'soaring' / 'data' / 'probability_maps'
               / 'val_de_ruz_may_avg.tif')


def plot_all(registry_path: Path, tif_path: Path = DEFAULT_TIF) -> None:
    """Generate trajectory.png and throttle.png for every completed trial.

    Reads the trial registry to find each flight CSV, then calls the
    existing plotter functions.  Outputs land next to the CSV:
        results/<algo>_<seed>/trajectory.png
        results/<algo>_<seed>/throttle.png
    """
    import sys as _sys
    _sys.path.insert(0, str(ROOT_DIR / 'soaring' / 'data'))
    from plotter import plot_trajectory, plot_altitude_throttle

    if not registry_path.exists():
        print('[plot] no registry found — nothing to plot')
        return

    rows = list(csv.DictReader(registry_path.open()))
    done = [r for r in rows if r['status'] == 'done' and r['csv_path']]
    if not done:
        print('[plot] no completed trials in registry')
        return

    if not tif_path.exists():
        print(f'[plot] TIF not found: {tif_path} — trajectory plots skipped')
        tif_path = None

    for row in done:
        tel_csv = Path(row['csv_path'])
        if not tel_csv.exists():
            print(f'[plot] missing CSV {tel_csv} — skipping')
            continue

        out_dir = tel_csv.parent
        label   = f"{row['algo']} seed={row['seed']}"

        if tif_path:
            plot_trajectory(
                tel_csv  = tel_csv,
                tif_path = tif_path,
                out_path = out_dir / 'trajectory.png',
            )
        plot_altitude_throttle(
            tel_csv  = tel_csv,
            out_path = out_dir / 'throttle.png',
        )
        print(f'[plot] {label} → {out_dir}')

    print(f'[plot] done — {len(done)} trial(s) plotted.')


# ── clean ──────────────────────────────────────────────────────────────────────

def clean(registry_path: Path) -> None:
    """Delete all artifacts from previous runs."""
    import shutil

    # Discover PX4 instance dirs dynamically (build dir may vary)
    default_px4 = Path.home() / 'PX4-Autopilot'
    build_dir   = default_px4 / 'build' / 'px4_sitl_default'
    px4_instances = list(build_dir.glob('instance_*')) if build_dir.exists() else []

    targets = [
        (LOGS,                          'process logs'),
        (RESULTS,                       'results'),
        (ROOT_DIR / 'soaring' / 'data' / 'thermals', 'thermal logs'),
        (ROOT_DIR / 'soaring' / 'data' / 'telemetry', 'telemetry CSVs'),
        *[(p, f'px4 {p.name}') for p in px4_instances],
    ]
    for path, label in targets:
        if path.exists():
            shutil.rmtree(path)
            print(f'[clean] removed {label:20s}  {path}')

    for f in [registry_path, ROOT_DIR / '.last_flight']:
        if f.exists():
            f.unlink()
            print(f'[clean] removed {f}')

    for lock in Path('/tmp').glob('px4_lock-*'):
        lock.unlink(missing_ok=True)
        print(f'[clean] removed {lock}')

    print('[clean] done.')


# ── summary ────────────────────────────────────────────────────────────────────

def summarise(registry_path: Path) -> None:
    """Aggregate all done trials into results/summary.csv."""
    rows = list(csv.DictReader(registry_path.open()))
    done = [r for r in rows if r['status'] == 'done' and r['csv_path']]
    if not done:
        print('[runner] no completed trials to summarise')
        return

    summary_path = RESULTS / 'summary.csv'
    RESULTS.mkdir(parents=True, exist_ok=True)
    with summary_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['algo', 'seed', 'duration_s', 'n_rows', 'csv_path'])
        for row in done:
            csv_p = Path(row['csv_path'])
            if not csv_p.exists():
                continue
            try:
                flight_rows = list(csv.DictReader(csv_p.open()))
                duration = float(flight_rows[-1]['time_s']) if flight_rows else 0.0
                w.writerow([row['algo'], row['seed'],
                             f'{duration:.1f}', len(flight_rows), str(csv_p)])
            except Exception:
                pass
    print(f'[runner] summary → {summary_path}')


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    DEFAULT_PX4 = Path.home() / 'PX4-Autopilot'

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--algorithms', nargs='+', metavar='SCRIPT',
                    help='Mission planner scripts (e.g. straightline.py algo_b.py)')
    ap.add_argument('--seeds', type=Path, default=Path('seeds.txt'),
                    help='Text file with one integer seed per line (default: seeds.txt)')
    ap.add_argument('--workers', type=int, default=2,
                    help='Parallel simulation stacks (default: 2)')
    ap.add_argument('--speed-factor', type=int, default=10,
                    help='PX4_SIM_SPEED_FACTOR (default: 10)')
    ap.add_argument('--px4-dir', type=Path, default=DEFAULT_PX4,
                    help=f'Path to PX4-Autopilot (default: {DEFAULT_PX4})')
    ap.add_argument('--registry', type=Path, default=Path('trials.csv'),
                    help='Trial registry CSV path (default: trials.csv)')
    ap.add_argument('--clean', action='store_true',
                    help='Delete logs/, results/, thermals/, telemetry/, '
                         'trials.csv, .last_flight, and stale /tmp/px4_lock-* files, then exit.')
    ap.add_argument('--plot', action='store_true',
                    help='Generate trajectory.png and throttle.png for all completed '
                         'trials in the registry, then exit.')
    ap.add_argument('--tif', type=Path, default=DEFAULT_TIF,
                    help=f'Probability map GeoTIFF for trajectory plots (default: {DEFAULT_TIF})')
    args = ap.parse_args()

    if args.clean:
        clean(args.registry)
        return

    if args.plot:
        plot_all(args.registry, tif_path=args.tif)
        return

    if not args.algorithms:
        ap.error('--algorithms is required unless --clean or --plot is used')

    px4_dir = args.px4_dir.expanduser().resolve()
    if not px4_dir.exists():
        print(f'ERROR: PX4 dir not found: {px4_dir}', file=sys.stderr)
        sys.exit(1)

    seeds_text = args.seeds.read_text().strip()
    if not seeds_text:
        print('ERROR: seeds file is empty', file=sys.stderr)
        sys.exit(1)
    seeds = [int(line) for line in seeds_text.splitlines() if line.strip()]

    registry = TrialRegistry(args.registry)
    registry.generate(args.algorithms, seeds)
    registry.reset_stale_running()   # recover from any previous crash

    n_total = len(args.algorithms) * len(seeds)
    print(f'[runner] {n_total} trials | {args.workers} workers | '
          f'speed×{args.speed_factor} | px4={px4_dir}')

    # Each slot runs its own worker thread that lazily claims trials.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(run_worker, slot, registry,
                        px4_dir, args.speed_factor)
            for slot in range(args.workers)
        ]
        for f in as_completed(futures):
            f.result()   # re-raise worker exceptions

    summarise(args.registry)
    plot_all(args.registry, tif_path=args.tif)
    print('[runner] all done.')


if __name__ == '__main__':
    main()
