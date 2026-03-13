"""
Smoke test: run 4 trials (2 algos × 2 seeds) through the full registry
→ worker → SimStack (mocked _run) → summarise pipeline.
No PX4 or Gazebo required.
"""
import csv
import sys
import subprocess
from pathlib import Path
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed

import test_runner
from test_runner import (
    TrialRegistry, Trial, SimStack, summarise, run_worker,
)


def _mock_run(self):
    """Replace SimStack._run to execute mock_mission directly."""
    out = self.result_dir
    out.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(
        [sys.executable,
         str(Path(__file__).parent / 'mock_mission.py'),
         '--port', str(self.port),
         '--output-dir', str(out)],
    ).returncode
    return rc, self._find_csv()


def test_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(SimStack, '_run', _mock_run)
    monkeypatch.setattr(test_runner, 'RESULTS', tmp_path / 'results')
    monkeypatch.setattr(test_runner, 'LOGS',    tmp_path / 'logs')

    registry_path = tmp_path / 'trials.csv'
    reg = TrialRegistry(registry_path)
    reg.generate(['straightline.py', 'tests/mock_mission.py'], [1, 2])

    # Run with 2 workers — should complete all 4 trials
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(run_worker, slot, reg, Path('/fake/px4'), 10)
            for slot in range(2)
        ]
        for f in as_completed(futures):
            f.result()

    rows = list(csv.DictReader(registry_path.open()))
    done   = [r for r in rows if r['status'] == 'done']
    failed = [r for r in rows if r['status'] == 'failed']
    assert len(done) == 4, f'Expected 4 done, got {done}'
    assert len(failed) == 0

    # Check summary
    summarise(registry_path)
    summary = tmp_path / 'results' / 'summary.csv'
    assert summary.exists()
    summary_rows = list(csv.DictReader(summary.open()))
    assert len(summary_rows) == 4
