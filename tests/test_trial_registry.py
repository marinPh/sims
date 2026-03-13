import csv
from pathlib import Path
import pytest
from test_runner import TrialRegistry, Trial

@pytest.fixture
def reg(tmp_path):
    return TrialRegistry(tmp_path / 'trials.csv')

def test_generate_creates_correct_count(reg):
    reg.generate(['algo_a.py', 'algo_b.py'], [1, 2, 3])
    rows = list(csv.DictReader(reg.path.open()))
    assert len(rows) == 6   # 2 algos × 3 seeds

def test_generate_skips_if_exists(reg):
    reg.generate(['a.py'], [1])
    reg.generate(['b.py'], [2])   # must not overwrite
    rows = list(csv.DictReader(reg.path.open()))
    assert len(rows) == 1

def test_claim_returns_pending_trial(reg):
    reg.generate(['a.py'], [99])
    trial = reg.claim_next()
    assert trial is not None
    assert trial.seed == 99
    assert trial.script == 'a.py'

def test_claim_marks_running(reg):
    reg.generate(['a.py'], [99])
    reg.claim_next()
    rows = list(csv.DictReader(reg.path.open()))
    assert rows[0]['status'] == 'running'

def test_claim_returns_none_when_exhausted(reg):
    reg.generate(['a.py'], [1])
    reg.claim_next()
    assert reg.claim_next() is None

def test_mark_done(reg):
    reg.generate(['a.py'], [1])
    trial = reg.claim_next()
    reg.mark_done(trial, csv_path='/tmp/x.csv', exit_code=0)
    rows = list(csv.DictReader(reg.path.open()))
    assert rows[0]['status'] == 'done'
    assert rows[0]['csv_path'] == '/tmp/x.csv'

def test_mark_failed(reg):
    reg.generate(['a.py'], [1])
    trial = reg.claim_next()
    reg.mark_failed(trial, exit_code=1)
    rows = list(csv.DictReader(reg.path.open()))
    assert rows[0]['status'] == 'failed'

def test_failed_trial_is_reclaimed(reg):
    reg.generate(['a.py'], [1])
    trial = reg.claim_next()
    reg.mark_failed(trial, exit_code=1)
    trial2 = reg.claim_next()
    assert trial2 is not None
    assert trial2.seed == 1

def test_update_raises_if_not_running(reg):
    reg.generate(['a.py'], [1])
    trial = Trial(algo='a', script='a.py', seed=1, status='running')
    # trial was never claimed, so no row has status='running'
    import pytest
    with pytest.raises(RuntimeError, match='No running row'):
        reg.mark_done(trial, csv_path='/tmp/x.csv', exit_code=0)

def test_mark_done_records_exit_code(reg):
    reg.generate(['a.py'], [1])
    trial = reg.claim_next()
    reg.mark_done(trial, csv_path='/tmp/x.csv', exit_code=0)
    rows = list(csv.DictReader(reg.path.open()))
    assert rows[0]['exit_code'] == '0'

def test_reset_stale_running(reg):
    reg.generate(['a.py'], [1, 2])
    reg.claim_next()   # marks first row 'running'
    reg.claim_next()   # marks second row 'running'
    count = reg.reset_stale_running()
    assert count == 2
    rows = list(csv.DictReader(reg.path.open()))
    assert all(r['status'] == 'pending' for r in rows)
    # After reset, both trials should be claimable again
    assert reg.claim_next() is not None
    assert reg.claim_next() is not None
