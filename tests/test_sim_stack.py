import os
from test_runner import slot_mavlink_port, slot_env

def test_slot_0_port():
    assert slot_mavlink_port(0) == 14550

def test_slot_1_port():
    assert slot_mavlink_port(1) == 14560

def test_slot_2_port():
    assert slot_mavlink_port(2) == 14570

def test_slot_env_required_keys():
    env = slot_env(slot=1, seed=42, speed_factor=10)
    assert env['PX4_INSTANCE']          == '1'
    assert env['GZ_PARTITION']          == 'slot1'
    assert env['HEADLESS']              == '1'
    assert env['PX4_SIM_SPEED_FACTOR']  == '10'

def test_slot_env_inherits_path():
    env = slot_env(slot=0, seed=0, speed_factor=1)
    assert 'PATH' in env

def test_slot_env_different_slots_different_partitions():
    e0 = slot_env(0, 1, 10)
    e1 = slot_env(1, 1, 10)
    assert e0['GZ_PARTITION'] != e1['GZ_PARTITION']
    assert e0['PX4_INSTANCE'] != e1['PX4_INSTANCE']
