"""
Online/incremental pipeline: scores only a rolling window of recent days, using the
clustering and baselines already frozen by train_pipeline.py — never refits anything.

Why a separate module instead of just calling train_pipeline.py every day: cost. The batch
pipeline reprocesses the ENTIRE history every run — fine weekly, wasteful daily once the
history is months long. This module reuses train_pipeline's channel logic directly (DRY) on
a small window, so its cost stays roughly constant as the dataset grows instead of scaling
with total history.

Production note: the system this project is based on deliberately does the OPPOSITE trade —
it duplicates the rule engine into a fully separate daily module rather than importing the
batch one. That decoupling means an in-progress edit to the weekly pipeline (mid-refactor,
not yet validated) can never change what the already-running daily job produces until the
weekly job is explicitly re-run and its frozen artifacts replaced. This demo optimizes for
readability (one engine, not two copies to keep in sync); production optimized for isolation.
Which one is right depends on how often the rule engine actually changes and how bad a silent
daily/weekly mismatch would be — worth stating explicitly rather than picking one silently.
"""
import pickle
import sqlite3
from pathlib import Path

import pandas as pd

import train_pipeline as tp

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ROLLING_WINDOW_DAYS = 45  # generous margin over the longest lookback any channel needs


def load_frozen_artifacts():
    cluster_map = pd.read_pickle(MODELS_DIR / "cluster_map.pkl")
    baselines = pd.read_pickle(MODELS_DIR / "device_baselines.pkl")
    return cluster_map, baselines


def run_daily(as_of_day=None):
    """as_of_day: score up to and including this day index. Defaults to the latest day
    present in the data (i.e. "today" in a real deployment)."""
    telemetry, _ = tp.load_daily_dataset()
    if as_of_day is None:
        as_of_day = int(telemetry["day_index"].max())

    window_start = max(0, as_of_day - ROLLING_WINDOW_DAYS)
    window = telemetry[(telemetry["day_index"] >= window_start) & (telemetry["day_index"] <= as_of_day)]

    cluster_map, baselines = load_frozen_artifacts()
    # Same channel logic as the batch pipeline (see module docstring for why this demo
    # imports it rather than duplicating it), applied to a small window instead of full history.
    result = tp.run_engine(window, cluster_map, baselines, confirmed_fault_days=set())

    latest_day = result[result["day_index"] == as_of_day]
    n_devices = len(latest_day)
    n_alerts = int(latest_day["is_anomalous"].sum())
    print(f"Day {as_of_day}: {n_devices} devices scored, {n_alerts} anomalous.")
    if n_alerts:
        cols = ["sn", "battery_min", "battery_max", "primary_channel", "active_channels"]
        print(latest_day[latest_day["is_anomalous"] == 1][cols].to_string(index=False))
    return latest_day


if __name__ == "__main__":
    if not (MODELS_DIR / "cluster_map.pkl").exists():
        raise SystemExit(
            "No frozen artifacts found — run train_pipeline.py at least once before "
            "daily_inference.py (mirrors production: the daily job depends on the weekly "
            "job having run first)."
        )
    run_daily()
