"""
Synthetic data generator for a solar-powered IoT device fleet.

Produces a self-contained SQLite database with realistic daily battery telemetry for a
fleet of devices spread across a few geographic/climate clusters, with a known set of
injected fault episodes (ground truth) used later to measure precision/recall of the
detection engine honestly, without any manual labeling step.

No real company, device, or telemetry data is used anywhere in this project — this script
is the only source of data for the whole repository.
"""
import random
import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np

RANDOM_SEED = 42
N_DAYS = 240
START_DATE = date(2025, 1, 1)
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "fleet.db"

# --- Geographic/climate clusters ---------------------------------------------------------
# Each cluster has its own coordinates, ambient climate, and typical panel recharge amount
# (a sunnier/warmer region recharges more per day) — this is what the clustering step in
# train_pipeline.py is expected to recover purely from the data, without being told the
# cluster labels.
@dataclass
class ClusterProfile:
    name: str
    lat: float
    lon: float
    avg_temp: float
    avg_humidity: float
    charge_delta_mean: float   # typical daily recharge amount (V) in this climate
    n_devices: int


CLUSTERS = [
    ClusterProfile("North-Coastal", lat=52.5, lon=4.9, avg_temp=14.0, avg_humidity=78.0, charge_delta_mean=1.4, n_devices=55),
    ClusterProfile("South-Arid", lat=37.2, lon=-3.6, avg_temp=24.0, avg_humidity=38.0, charge_delta_mean=2.3, n_devices=50),
    ClusterProfile("Central-Continental", lat=48.1, lon=11.6, avg_temp=17.5, avg_humidity=60.0, charge_delta_mean=1.8, n_devices=60),
    ClusterProfile("Alpine-Highland", lat=46.8, lon=9.8, avg_temp=9.0, avg_humidity=65.0, charge_delta_mean=1.1, n_devices=35),
]

BATTERY_FLOOR = 10.5    # V, floor for gradual decline/undercharge scenarios
BATTERY_CEILING = 15.5  # V, physically full
SHOCK_PHYSICAL_MIN = 3.0  # V, floor ONLY for a catastrophic shock crash — must be able to
                          # go below the engine's shock-detection threshold (8.0V), unlike
                          # a slow undercharge which never gets that low


@dataclass
class Device:
    sn: str
    cluster: ClusterProfile
    resting_level: float        # this device's own healthy battery_min baseline
    charge_baseline: float      # this device's own healthy daily recharge amount
    first_seen_offset: int      # day index (0-based) the device joins the fleet
    lat: float = 0.0
    lon: float = 0.0
    structurally_different: bool = False  # the deliberate false-positive trap (see build_fleet)
    fault: dict = field(default_factory=dict)  # injected fault spec, or {}


def build_fleet(rng):
    devices = []
    idx = 0
    for cluster in CLUSTERS:
        for _ in range(cluster.n_devices):
            idx += 1
            sn = f"DEV{idx:05d}"
            # Most devices sit close to their cluster's typical level; a deliberate minority
            # are structurally different but perfectly healthy (the classic false-positive
            # trap for any detector that compares devices to their neighbors instead of to
            # their own history: panel orientation/local shading make some devices run a
            # couple of volts lower than the rest of an otherwise identical cluster).
            structurally_different = rng.random() < 0.08
            resting_level = rng.normal(12.6, 0.25)
            if structurally_different:
                resting_level -= rng.uniform(1.8, 2.6)
            charge_baseline = max(0.3, rng.normal(cluster.charge_delta_mean, 0.25))
            first_seen_offset = 0 if rng.random() < 0.85 else rng.integers(0, N_DAYS - 40)
            lat = cluster.lat + rng.normal(0, 0.35)
            lon = cluster.lon + rng.normal(0, 0.35)
            devices.append(Device(sn, cluster, resting_level, charge_baseline, int(first_seen_offset),
                                   lat, lon, structurally_different))
    return devices


# --- Fault injection -----------------------------------------------------------------------
FAULT_TYPES = ["SHOCK", "UNDERCHARGE", "CHRONIC_LEVEL", "FLATLINE", "MONOTONIC_ONSET", "RECURRING_SHUTDOWN"]


def pick_faulty_devices(devices, rng, n_per_type=6):
    # Structurally-different devices are reserved as the false-positive trap (see
    # build_fleet) — assigning them a real fault too would make it impossible to tell
    # "detected because trap" from "detected because genuinely faulty" in the ground truth.
    healthy_adult = [d for d in devices if d.first_seen_offset == 0 and not d.structurally_different]
    young = [d for d in devices if d.first_seen_offset > 0 and not d.structurally_different]
    rng.shuffle(healthy_adult)
    rng.shuffle(young)
    assignment = {}
    pool = iter(healthy_adult)
    for ftype in ["SHOCK", "UNDERCHARGE", "CHRONIC_LEVEL", "FLATLINE"]:
        for _ in range(n_per_type):
            d = next(pool)
            assignment[d.sn] = ftype
    # the two "young device" fault types need devices that actually join the fleet late
    young_pool = iter(young)
    for ftype in ["MONOTONIC_ONSET", "RECURRING_SHUTDOWN"]:
        for _ in range(min(n_per_type, len(young))):
            try:
                d = next(young_pool)
            except StopIteration:
                break
            assignment[d.sn] = ftype
    return assignment


def simulate_device(device, fault_type, rng, forced_fault_start=None):
    """Returns (rows, ground_truth_episodes). rows = list of (day_index, bmin, bmax, temp,
    humidity) or None for a day with no telemetry at all (device silent that day)."""
    rows = [None] * N_DAYS
    episodes = []

    # Sparse, uncorrelated single-day network blips for every device — exercises gap-aware
    # streak counting in the engine; must NEVER be flagged as an anomaly by itself.
    blip_days = set(rng.choice(N_DAYS, size=rng.integers(3, 9), replace=False))

    level = device.resting_level
    fault_start = fault_end = None
    if fault_type:
        fault_start = forced_fault_start if forced_fault_start is not None else \
            int(rng.integers(device.first_seen_offset + 5, N_DAYS - 20))

    shutdown_cycle_state = {"phase": "decline", "phase_day": 0, "cycle": 0}
    monotonic_peak = device.resting_level + device.charge_baseline

    for day in range(device.first_seen_offset, N_DAYS):
        if day in blip_days:
            continue  # no telemetry this day — pure noise, not a fault

        weather = max(0.25, rng.normal(1.0, 0.28))  # shared-ish daily weather noise
        bmin = level + rng.normal(0, 0.08)
        bmax = min(BATTERY_CEILING, bmin + device.charge_baseline * weather + rng.normal(0, 0.06))

        in_fault_window = fault_start is not None and day >= fault_start

        if fault_type == "SHOCK" and in_fault_window and fault_end is None:
            bmin = max(SHOCK_PHYSICAL_MIN, device.resting_level - rng.uniform(5.0, 7.5))
            bmax = bmin + rng.uniform(0.1, 0.5)
            if day - fault_start >= rng.integers(2, 4):
                fault_end = day
                episodes.append((device.sn, fault_start, fault_end, fault_type))

        elif fault_type == "UNDERCHARGE" and in_fault_window and fault_end is None:
            bmax = bmin + rng.uniform(0.05, 0.35)  # stops recharging properly
            bmin = max(BATTERY_FLOOR, bmin - rng.uniform(0.03, 0.08))  # slow bleed
            level = bmin
            if day - fault_start >= rng.integers(14, 22):
                fault_end = day
                episodes.append((device.sn, fault_start, fault_end, fault_type))

        elif fault_type == "CHRONIC_LEVEL" and in_fault_window and fault_end is None:
            bmin = max(BATTERY_FLOOR, device.resting_level - rng.uniform(1.6, 2.2))
            bmax = bmin + device.charge_baseline * 0.6 * weather
            if day - fault_start >= rng.integers(12, 20):
                fault_end = day
                episodes.append((device.sn, fault_start, fault_end, fault_type))

        elif fault_type == "FLATLINE" and in_fault_window and fault_end is None:
            frozen_value = device.resting_level + device.charge_baseline * 0.5
            bmin = bmax = frozen_value + rng.normal(0, 0.005)  # sensor stuck: near-zero variance
            if day - fault_start >= rng.integers(10, 18):
                fault_end = day
                episodes.append((device.sn, fault_start, fault_end, fault_type))

        elif fault_type == "MONOTONIC_ONSET" and in_fault_window and fault_end is None:
            monotonic_peak -= rng.uniform(0.12, 0.22)
            bmax = max(BATTERY_FLOOR, monotonic_peak) + rng.normal(0, 0.03)
            bmin = bmax - rng.uniform(0.1, 0.3)
            if bmax <= BATTERY_FLOOR + 0.3:
                fault_end = day
                episodes.append((device.sn, fault_start, fault_end, fault_type))
                # device dies: no more telemetry for the rest of the horizon
                for d2 in range(day + 1, N_DAYS):
                    rows[d2] = "DEAD"
                break

        elif fault_type == "RECURRING_SHUTDOWN" and in_fault_window:
            st = shutdown_cycle_state
            if st["phase"] == "decline":
                bmax = max(BATTERY_FLOOR + 0.5, level - st["phase_day"] * 0.35)
                bmin = bmax - rng.uniform(0.1, 0.3)
                st["phase_day"] += 1
                if st["phase_day"] >= rng.integers(3, 5):
                    st["phase"] = "silent"
                    st["phase_day"] = 0
                    st["silent_len"] = int(rng.integers(2, 4))
                    st["episode_start"] = fault_start if st["cycle"] == 0 else st["episode_start"]
            elif st["phase"] == "silent":
                rows[day] = None
                st["phase_day"] += 1
                if st["phase_day"] >= st["silent_len"]:
                    st["phase"] = "recharged"
                continue
            elif st["phase"] == "recharged":
                # panel recharges at zero load while the device is off: comes back HIGHER
                # than it was before the outage, never crashed — this is what makes it
                # invisible to a plain "returned from silence, still low" rule.
                level = min(BATTERY_CEILING - 1, level + rng.uniform(1.0, 1.8))
                bmin, bmax = level, level + device.charge_baseline * weather
                st["phase"] = "decline"
                st["phase_day"] = 0
                st["cycle"] += 1
                if st["cycle"] >= 4:
                    fault_end = day
                    episodes.append((device.sn, fault_start, fault_end, fault_type))
                    fault_type = None  # heals afterwards, back to normal simulation

        temp = rng.normal(device.cluster.avg_temp, 4.0)
        humidity = float(np.clip(rng.normal(device.cluster.avg_humidity, 8.0), 5, 100))
        rows[day] = (round(bmin, 3), round(bmax, 3), round(temp, 2), round(humidity, 2))

    return rows, episodes


def force_short_life_contamination_case(devices, fault_assignment, rng):
    """Deliberately construct one device whose observed history is SHORT and whose fault
    covers most of it — the scenario where baseline contamination actually bites (a fault
    that is a small fraction of a long history barely moves a median; one that IS most of a
    short history drags it a lot). Mirrors a real case found in production: a device with 15
    days of total history where 6 of them (40%) were a confirmed fault.
    Picked from an unassigned device so it doesn't clash with the other fault scenarios."""
    unassigned = [d for d in devices if d.sn not in fault_assignment and not d.structurally_different]
    target = unassigned[0]
    target.first_seen_offset = N_DAYS - 26   # entirely post-TRAIN_CUTOFF: no clustering
                                              # history either, exercises the cluster safety
                                              # net (assign_fallback_clusters) at the same time
    target.fault = {"forced_fault_start": target.first_seen_offset + 3}
    fault_assignment[target.sn] = "UNDERCHARGE"
    return target.sn


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    DB_PATH.parent.mkdir(exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""CREATE TABLE devices (
        sn TEXT PRIMARY KEY, latitude REAL, longitude REAL, first_seen_day INTEGER)""")
    cur.execute("""CREATE TABLE telemetry (
        sn TEXT, day_index INTEGER, date TEXT, battery_min REAL, battery_max REAL,
        temp_internal REAL, humidity REAL)""")
    cur.execute("""CREATE TABLE ground_truth_faults (
        sn TEXT, start_day INTEGER, end_day INTEGER, fault_type TEXT)""")
    cur.execute("""CREATE TABLE ground_truth_healthy_traps (
        sn TEXT, reason TEXT)""")  # devices deliberately built to fool a naive detector

    devices = build_fleet(rng)
    fault_assignment = pick_faulty_devices(devices, rng)
    contamination_demo_sn = force_short_life_contamination_case(devices, fault_assignment, rng)

    all_episodes = []
    rows_to_insert = []
    traps = []
    for d in devices:
        cur.execute("INSERT INTO devices VALUES (?,?,?,?);",
                    (d.sn, d.lat, d.lon, d.first_seen_offset))
        fault_type = fault_assignment.get(d.sn)
        if d.structurally_different:
            traps.append((d.sn, "structurally_low_but_stable"))
        forced_start = d.fault.get("forced_fault_start")
        rows, episodes = simulate_device(d, fault_type, rng, forced_fault_start=forced_start)
        all_episodes.extend(episodes)
        for day_index, r in enumerate(rows):
            if r is None or r == "DEAD":
                continue
            bmin, bmax, temp, hum = r
            the_date = (START_DATE + timedelta(days=day_index)).isoformat()
            rows_to_insert.append((d.sn, day_index, the_date, bmin, bmax, temp, hum))

    cur.executemany("INSERT INTO telemetry VALUES (?,?,?,?,?,?,?);", rows_to_insert)
    cur.executemany("INSERT INTO ground_truth_faults VALUES (?,?,?,?);", all_episodes)
    cur.executemany("INSERT INTO ground_truth_healthy_traps VALUES (?,?);", traps)
    con.commit()

    n_devices = len(devices)
    n_rows = len(rows_to_insert)
    print(f"Synthetic fleet generated: {n_devices} devices, {n_rows:,} telemetry rows, "
          f"{N_DAYS} days ({START_DATE} -> {START_DATE + timedelta(days=N_DAYS - 1)}).")
    print(f"Injected fault episodes: {len(all_episodes)} across {len(fault_assignment)} devices.")
    print(f"'Healthy trap' devices (structurally different, must NOT alarm): {len(traps)}.")
    print(f"Baseline-contamination demo device: {contamination_demo_sn} "
          f"(short life, fault covers most of its observed history).")
    print(f"Database written to: {DB_PATH}")
    con.close()


if __name__ == "__main__":
    main()
