#!/usr/bin/env python3
# launch.py
"""
Launch everything needed for the thermal soaring simulation in one command.

Start order
-----------
  1. PX4 SITL + Gazebo   (make px4_sitl gz_advanced_plane)
  2. Wait for Gazebo to publish topics
  3. Thermal wind bridge  (simulation/gz_sim.py)

Usage
-----
    python3 launch.py [--px4-dir ~/PX4-Autopilot] [--no-wind-bridge]

Ctrl-C shuts down all child processes cleanly.
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from datetime import datetime

# ── configuration ──────────────────────────────────────────────────────────────

DEFAULT_PX4_DIR  = Path.home() / 'PX4-Autopilot'
MAKE_TARGET      = 'px4_sitl gz_advanced_plane -j6'
SIM_DIR          = Path(__file__).parent / 'soaring' / 'simulation'
ROOT_DIR         = Path(__file__).parent

GZ_READY_TOPIC   = '/world/default'   # substring to look for in `gz topic -l`
GZ_WAIT_TIMEOUT  = 90                 # seconds to wait for Gazebo
GZ_POLL_INTERVAL = 2.0                # seconds between readiness polls

# ── ANSI colour prefixes per process ──────────────────────────────────────────

COLOURS = {
    'px4'      : '\033[36m',   # cyan
    'gz_sim'   : '\033[33m',   # yellow
    'mission'  : '\033[32m',   # green
}
RESET = '\033[0m'

# ── process registry ──────────────────────────────────────────────────────────

_procs: dict[str, subprocess.Popen] = {}


def _stream_output(name: str, stream) -> None:
    """Read lines from stream and print with a coloured prefix."""
    colour = COLOURS.get(name, '')
    prefix = f'{colour}[{name}]{RESET} '
    try:
        for raw in stream:
            line = raw.decode(errors='replace').rstrip('\n')
            if line:
                print(prefix + line, flush=True)
    except ValueError:
        pass   # stream closed


def _start(name: str, cmd: list[str], cwd: Path) -> subprocess.Popen:
    """Spawn a subprocess and attach a stdout/stderr reader thread."""
    print(f'\033[1m[launcher] starting {name}\033[0m  {" ".join(cmd)}')
    proc = subprocess.Popen(
        cmd,
        cwd        = str(cwd),
        stdout     = subprocess.PIPE,
        stderr     = subprocess.STDOUT,
        env        = os.environ,
        preexec_fn = os.setsid,   # own process group for clean kill
    )
    _procs[name] = proc
    t = threading.Thread(target=_stream_output, args=(name, proc.stdout),
                         daemon=True)
    t.start()
    return proc


def _shutdown(sig=None, frame=None) -> None:
    """Terminate all child processes."""
    print('\n\033[1m[launcher] shutting down…\033[0m')
    for name, proc in _procs.items():
        if proc.poll() is None:
            print(f'[launcher] stopping {name} (pgid {os.getpgid(proc.pid)})')
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
    time.sleep(1.5)
    for proc in _procs.values():
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    print('[launcher] done.')
    sys.exit(0)


# ── Gazebo readiness check ─────────────────────────────────────────────────────

def _gz_is_ready() -> bool:
    """Return True if `gz topic -l` lists the default world topic."""
    try:
        result = subprocess.run(
            ['gz', 'topic', '-l'],
            capture_output = True,
            timeout        = 5,
        )
        return GZ_READY_TOPIC.encode() in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _wait_for_gazebo() -> bool:
    """Poll until Gazebo is ready or timeout expires. Returns True on success."""
    print(f'[launcher] waiting for Gazebo (up to {GZ_WAIT_TIMEOUT}s)…', flush=True)
    deadline = time.monotonic() + GZ_WAIT_TIMEOUT
    while time.monotonic() < deadline:
        if _gz_is_ready():
            print('[launcher] Gazebo is ready.')
            return True
        time.sleep(GZ_POLL_INTERVAL)
        # bail early if PX4 already died
        if _procs.get('px4') and _procs['px4'].poll() is not None:
            print('[launcher] PX4 process exited unexpectedly.', file=sys.stderr)
            return False
    print('[launcher] timed out waiting for Gazebo.', file=sys.stderr)
    return False


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--px4-dir', type=Path, default=DEFAULT_PX4_DIR,
                        help=f'Path to PX4-Autopilot (default: {DEFAULT_PX4_DIR})')
    parser.add_argument('--no-wind-bridge', action='store_true',
                        help='Skip launching gz_sim.py (wind bridge)')
    parser.add_argument('--no-mission', action='store_true',
                        help='Skip launching the straight-line mission script')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip post-mission plots after mission completes')
    args = parser.parse_args()

    px4_dir = args.px4_dir.expanduser().resolve()
    if not px4_dir.exists():
        print(f'ERROR: PX4 directory not found: {px4_dir}', file=sys.stderr)
        print('Pass the correct path with --px4-dir', file=sys.stderr)
        sys.exit(1)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── 1. PX4 SITL + Gazebo ──────────────────────────────────────────────────
    # Spawn plane at Val-de-Ruz (centre of the thermal probability map)
    os.environ.setdefault('PX4_HOME_LAT', '47.0500')
    os.environ.setdefault('PX4_HOME_LON', '6.9500')
    os.environ.setdefault('PX4_HOME_ALT', '0')     # flat Gazebo world: z=0 is ground
    _start('px4', ['make'] + MAKE_TARGET.split(), cwd=px4_dir)

    # ── 2. wait for Gazebo ────────────────────────────────────────────────────
    if not _wait_for_gazebo():
        _shutdown()

    # small extra delay — Gazebo is up but model may still be loading
    time.sleep(3.0)

    # ── 3. thermal wind bridge ────────────────────────────────────────────────
    if not args.no_wind_bridge:
        os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
        _start('gz_sim', [sys.executable, 'gz_sim.py'], cwd=SIM_DIR)

    # ── 4. straight-line mission ──────────────────────────────────────────────
    if not args.no_mission:
        # extra delay: PX4 needs a few seconds to finish booting after Gazebo
        print('[launcher] waiting 15 s for PX4 to finish booting…')
        time.sleep(15.0)
        _start('mission', [sys.executable, 'straightline.py'], cwd=ROOT_DIR)

    print('\n\033[1m[launcher] all processes running — Ctrl-C to stop\033[0m\n')

    def _run_plotter() -> None:
        """Run post-mission visualisations after the mission script exits."""
        sentinel = ROOT_DIR / '.last_flight'
        if not sentinel.exists():
            print('[launcher] .last_flight sentinel not found — skipping plots')
            return
        tel_path = Path(sentinel.read_text().strip())
        if not tel_path.exists():
            print(f'[launcher] telemetry file not found: {tel_path}')
            return

        # find the thermals JSONL with the closest timestamp
        thermals_dir = ROOT_DIR / 'soaring' / 'data' / 'thermals'
        thermals_arg = []
        if thermals_dir.exists():
            jsonl_files = sorted(thermals_dir.glob('thermals_*.jsonl'))
            if jsonl_files:
                # pick the one whose timestamp is closest to the telemetry file
                def _ts(p):
                    try:
                        return datetime.strptime(p.stem.split('_', 1)[1], '%Y%m%d_%H%M%S')
                    except ValueError:
                        return datetime.min
                tel_ts = _ts(tel_path)
                best   = min(jsonl_files, key=lambda p: abs((_ts(p) - tel_ts).total_seconds()))
                thermals_arg = ['--thermals', str(best)]

        tif_path  = ROOT_DIR / 'soaring' / 'data' / 'probability_maps' / 'val_de_ruz_may_avg.tif'
        out_dir   = ROOT_DIR / 'soaring' / 'data' / 'plots'
        plotter   = ROOT_DIR / 'soaring' / 'data' / 'plotter.py'

        cmd = [
            sys.executable, str(plotter),
            '--telemetry', str(tel_path),
            '--tif',       str(tif_path),
            '--out-dir',   str(out_dir),
        ] + thermals_arg

        print(f'[launcher] running plotter: {" ".join(cmd)}')
        result = subprocess.run(cmd, cwd=str(ROOT_DIR))
        if result.returncode == 0:
            print(f'[launcher] plots saved to {out_dir}')
        else:
            print(f'[launcher] plotter exited with code {result.returncode}')

    # wait for any process to exit unexpectedly
    # mission exiting normally (done) is fine — don't shutdown for it
    while True:
        for name, proc in list(_procs.items()):
            if proc.poll() is not None:
                if name == 'mission' and proc.returncode == 0:
                    print(f'[launcher] {name} finished successfully.')
                    del _procs[name]
                    if not args.no_plots:
                        _run_plotter()
                else:
                    print(f'\033[31m[launcher] {name} exited (code {proc.returncode})\033[0m')
                    _shutdown()
                break
        time.sleep(1.0)


if __name__ == '__main__':
    main()
