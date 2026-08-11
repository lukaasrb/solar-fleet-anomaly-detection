"""
Weekly/offline pipeline: rebuilds the daily aggregate, (re)fits geographic/electrical
clustering on the Train split only, freezes per-device baselines, and runs the full
multi-channel rule engine over the entire history.

Design principles (the parts worth reading if you're reviewing this as a portfolio piece):

1. SELF-REFERENTIAL baselines, not fleet-wide comparison. A device that is structurally
   different from its neighbors (different panel orientation, local shading) but stable
   over time is not anomalous — comparing it to *itself* over time, instead of to its
   cluster peers on a given day, removes a whole class of false positives for free. See
   `freeze_device_baselines()`.

2. GAP-AWARE persistence counting. A missing day of telemetry is not the same as a bad
   reading. Counting consecutive "bad" days naively over a calendar grid either inflates
   or resets streaks around a gap, depending on which direction the condition runs. See
   `consecutive_run_ignoring_gaps()`.

3. BASELINE DECONTAMINATION. If a device's "what is normal" statistic is computed over
   its whole history, and part of that history is a confirmed real fault, the fault
   dilutes its own reference point — the model becomes less sensitive to catching it (or
   future faults with the same shape). Once a human has confirmed an episode as a true
   fault, that window is excluded from every baseline calculation from then on. See
   `decontaminated_median()`. This never uses ground truth directly — only episodes a
   human has actually reviewed and confirmed (see `validate.py`, `simulate_manual_review`)
   — exactly mirroring how you would run this in production with no oracle available.

4. TRAIN/TEST SPLIT for the clustering and baselines (not for the rule engine itself,
   which scores the whole timeline) — so precision on the Test period is an honest,
   out-of-time estimate, not one computed on data the model's own statistics were fit on.

5. Multi-channel voting with a PRIORITY label + a FULL active-channel list. Channels are
   not independent (e.g. UNDERCHARGE nearly always co-occurs with CHRONIC_LEVEL) — a
   single "which channel fired" label would silently discard information on ~1 in 4
   flagged days in this dataset.
"""
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "fleet.db"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

N_DAYS_TOTAL = 240
TRAIN_CUTOFF_DAY = 168          # ~70% Train, 30% Test (out-of-time validation)
N_CLUSTERS = 4
MIN_CLUSTER_SIZE = 8            # under-sized clusters get merged into their nearest neighbor

# --- Channel thresholds (all self-referential unless noted) ------------------------------
SHOCK_ABSOLUTE_FLOOR = 8.0          # V — catastrophic, cluster-independent
SHOCK_PERSISTENCE_DAYS = 2

UNDERCHARGE_MARGIN = 1.1             # V below the device's own historical recharge ceiling
UNDERCHARGE_MIN_STREAK = 5           # consecutive REAL days (gap-aware)

CHRONIC_Z_THRESHOLD = -2.0
CHRONIC_PERSISTENCE_DAYS = 3
CHRONIC_SELF_MARGIN = 0.3            # V below the device's OWN historical battery_min

FLATLINE_NOISE_FLOOR = 0.08          # V, daily oscillation below this = frozen sensor
FLATLINE_MIN_STREAK = 5
FLATLINE_NEAR_FULL_MARGIN = 0.3      # a saturated (fully charged) device isn't "frozen"

YOUNG_DEVICE_MAX_AGE_DAYS = 30
MONOTONIC_MIN_STREAK = 6
MONOTONIC_CUMULATIVE_DROP = 2.2      # V lost from local peak before flagging
MONOTONIC_ABSOLUTE_GATE = 0.35       # bmax must also be within this margin of the device's
                                      # OWN resting floor — a few cloudy days off a lucky peak
                                      # isn't a fault; being back down near your own floor is

SHUTDOWN_GAP_MIN, SHUTDOWN_GAP_MAX = 2, 5
SHUTDOWN_CYCLES_THRESHOLD = 3
SHUTDOWN_WINDOW_DAYS = 40            # generous: 3 cycles of (3-5d decline + 2-4d silence)
                                      # span up to ~27 days worst case, plus margin


# ==========================================================================================
# Utilities
# ==========================================================================================
def consecutive_run_ignoring_gaps(group, flag_col, real_col="is_real_day"):
    """Streak of consecutive days where `flag_col` is truthy, counted ONLY over days with
    real telemetry. Gap days are invisible: they neither reset nor inflate the streak — it
    picks up again exactly where it left off at the next real day."""
    real = group[real_col].astype(bool)
    flags = group.loc[real, flag_col]
    run = flags.eq(1).groupby((flags == 0).cumsum()).cumsum()
    return run.reindex(group.index).ffill().fillna(0).astype(int)


def decontaminated_median(df, value_col, confirmed_fault_days):
    """Per-sn median of value_col, excluding days inside confirmed (human-reviewed) fault
    episodes. Falls back to the un-decontaminated median for a device whose entire history
    happens to fall inside a confirmed episode (rare, but losing the baseline entirely would
    make the device invisible to the engine — worse than a slightly contaminated baseline)."""
    if confirmed_fault_days:
        mask = df.apply(lambda r: (r["sn"], r["day_index"]) in confirmed_fault_days, axis=1)
    else:
        mask = pd.Series(False, index=df.index)
    clean = df.loc[~mask].groupby("sn")[value_col].median()
    raw = df.groupby("sn")[value_col].median()
    return clean.combine_first(raw)


def monotonic_decline_metrics(group):
    """Per-device, gap-aware: (consecutive real days without a new local peak, cumulative
    drop from that local peak). The peak resets to the current value on ANY day the device
    recovers — this is what makes it "monotonic decline", not just "currently below its
    best day ever" (which a noisy but perfectly healthy series would trip constantly)."""
    real = group["is_real_day"].astype(bool)
    values = group.loc[real, "battery_max"]
    streak, drop = [], []
    peak = None
    run = 0
    for v in values:
        if peak is None or v > peak:
            peak = v
            run = 0
        else:
            run += 1
        streak.append(run)
        drop.append(peak - v)
    streak_s = pd.Series(streak, index=values.index)
    drop_s = pd.Series(drop, index=values.index)
    return (streak_s.reindex(group.index).ffill().fillna(0),
            drop_s.reindex(group.index).ffill().fillna(0))


def expand_confirmed_days(confirmed_episodes):
    """[(sn, start_day, end_day), ...] -> {(sn, day), ...} for O(1) membership tests."""
    days = set()
    for sn, start, end in confirmed_episodes:
        for d in range(start, end + 1):
            days.add((sn, d))
    return days


# ==========================================================================================
# Dataset assembly
# ==========================================================================================
def load_daily_dataset():
    con = sqlite3.connect(DB_PATH)
    telemetry = pd.read_sql("SELECT * FROM telemetry;", con)
    devices = pd.read_sql("SELECT * FROM devices;", con)
    con.close()
    telemetry["charge_delta"] = telemetry["battery_max"] - telemetry["battery_min"]
    telemetry["split"] = np.where(telemetry["day_index"] < TRAIN_CUTOFF_DAY, "train", "test")
    return telemetry, devices


# ==========================================================================================
# Clustering (Train split only)
# ==========================================================================================
def fit_clustering(telemetry, devices):
    train = telemetry[telemetry["split"] == "train"]
    profile = train.groupby("sn").agg(
        avg_temp=("temp_internal", "mean"),
        avg_humidity=("humidity", "mean"),
    ).reset_index()

    clean_delta = train[(train["charge_delta"] > 0.1) & (train["charge_delta"] < 6.0)]
    signature = clean_delta.groupby("sn")["charge_delta"].median().rename("charge_signature").reset_index()

    matrix = devices.merge(profile, on="sn", how="inner").merge(signature, on="sn", how="inner")
    feature_cols = ["latitude", "longitude", "avg_temp", "avg_humidity", "charge_signature"]

    scaler = StandardScaler()
    X = scaler.fit_transform(matrix[feature_cols])

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    matrix["cluster_raw"] = kmeans.fit_predict(X)

    # Merge under-sized clusters into their nearest surviving centroid — protects the
    # Z-score channel from degenerating on a handful of members.
    centers = kmeans.cluster_centers_.copy()
    labels = matrix["cluster_raw"].copy()
    while True:
        counts = labels.value_counts()
        undersized = counts[counts < MIN_CLUSTER_SIZE].index.tolist()
        if not undersized or len(counts) <= 1:
            break
        smallest = counts.loc[undersized].idxmin()
        others = [c for c in counts.index if c != smallest]
        nearest = min(others, key=lambda c: np.linalg.norm(centers[smallest] - centers[c]))
        labels = labels.replace(smallest, nearest)
    matrix["cluster"] = labels

    MODELS_DIR.mkdir(exist_ok=True)
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODELS_DIR / "kmeans.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    with open(MODELS_DIR / "feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    print(f"Clustering fit on {len(matrix)} devices (Train split): "
          f"{matrix['cluster'].value_counts().sort_index().to_dict()}")
    return matrix[["sn", "cluster"]]


def assign_fallback_clusters(devices, cluster_map):
    """A device with NO telemetry before TRAIN_CUTOFF_DAY (onboarded late, or with a fault
    severe enough that it barely produced clean readings) never enters `fit_clustering`'s
    Train-only matrix. Without a fallback it would silently disappear from every downstream
    table — the inner join in `run_engine` drops it, fault and all, exactly the kind of
    device most likely to be having one. Falls back to the fleet's single most common
    cluster; production systems can do better (e.g. group by whatever coarse category is
    available — company, region, hardware batch), but "some cluster" beats "invisible"."""
    missing = devices[~devices["sn"].isin(cluster_map["sn"])]
    if missing.empty:
        return cluster_map
    fallback_cluster = cluster_map["cluster"].mode().iloc[0]
    extra = pd.DataFrame({"sn": missing["sn"], "cluster": fallback_cluster})
    print(f"Cluster safety net: {len(extra)} device(s) with no Train-period history assigned "
          f"the fleet's most common cluster ({fallback_cluster}) instead of being silently dropped.")
    return pd.concat([cluster_map, extra], ignore_index=True)


# ==========================================================================================
# Frozen per-device baselines (decontaminated)
# ==========================================================================================
def freeze_device_baselines(telemetry, confirmed_fault_days):
    charge_median = decontaminated_median(
        telemetry.assign(day_index=telemetry["day_index"]), "charge_delta", confirmed_fault_days
    ).rename("charge_delta_baseline")
    max_median = decontaminated_median(telemetry, "battery_max", confirmed_fault_days).rename("battery_max_baseline")
    min_median = decontaminated_median(telemetry, "battery_min", confirmed_fault_days).rename("battery_min_baseline")
    max_historical = telemetry.groupby("sn")["battery_max"].max().rename("battery_max_ceiling_seen")
    first_seen = telemetry.groupby("sn")["day_index"].min().rename("first_seen_day")

    baseline = pd.concat([charge_median, max_median, min_median, max_historical, first_seen], axis=1).reset_index()
    MODELS_DIR.mkdir(exist_ok=True)
    baseline.to_pickle(MODELS_DIR / "device_baselines.pkl")
    print(f"Baselines frozen for {len(baseline)} devices "
          f"({len(confirmed_fault_days)} confirmed-fault days excluded from the calculation).")
    return baseline


# ==========================================================================================
# Multi-channel rule engine
# ==========================================================================================
CHANNEL_PRIORITY = ["SHOCK", "UNDERCHARGE", "CHRONIC_LEVEL", "MONOTONIC_ONSET", "RECURRING_SHUTDOWN", "FLATLINE"]


def run_engine(telemetry, cluster_map, baselines, confirmed_fault_days):
    grid = telemetry.merge(cluster_map, on="sn", how="inner").sort_values(["sn", "day_index"])
    grid["is_real_day"] = True

    # Dense per-device calendar so gap days are explicit rows (is_real_day=False), which is
    # what makes the gap-aware streak helper meaningful. Built explicitly (not via
    # groupby.apply(reindex)) to avoid depending on pandas' index-stacking behaviour.
    first_seen_by_sn = grid.groupby("sn")["day_index"].min()
    full_index = pd.MultiIndex.from_tuples(
        [(sn, d) for sn, first in first_seen_by_sn.items() for d in range(int(first), N_DAYS_TOTAL)],
        names=["sn", "day_index"],
    )
    dense = grid.set_index(["sn", "day_index"]).reindex(full_index).reset_index()
    dense["is_real_day"] = dense["is_real_day"].fillna(False)
    dense["cluster"] = dense.groupby("sn")["cluster"].transform(lambda s: s.ffill().bfill())
    dense = dense.sort_values(["sn", "day_index"]).reset_index(drop=True)
    dense = dense.merge(baselines, on="sn", how="left")

    by_sn = dense.groupby("sn", sort=False)

    # --- channel A: SHOCK — absolute, immediate, bypasses every self-referential baseline
    dense["cond_shock"] = (dense["battery_min"] < SHOCK_ABSOLUTE_FLOOR).astype("Int64").fillna(0)
    dense["shock"] = (by_sn["cond_shock"].transform(
        lambda s: s.rolling(SHOCK_PERSISTENCE_DAYS, min_periods=SHOCK_PERSISTENCE_DAYS).sum()
        >= SHOCK_PERSISTENCE_DAYS)).astype(int)

    # --- channel B: UNDERCHARGE — self-referential floor, gap-aware streak, delayed
    # confirmation (a single day that recovers tomorrow is very rarely a real fault)
    dense["recharge_floor"] = dense["battery_max_baseline"] - UNDERCHARGE_MARGIN
    dense["cond_undercharge"] = ((dense["battery_max"] < dense["recharge_floor"]) & dense["is_real_day"]).astype(int)
    dense["undercharge_streak"] = dense.groupby("sn", group_keys=False).apply(
        lambda g: consecutive_run_ignoring_gaps(g, "cond_undercharge"))
    dense["undercharge_raw"] = (dense["undercharge_streak"] >= UNDERCHARGE_MIN_STREAK).astype(int)

    grp = dense.groupby("sn", sort=False)
    prev_flag = grp["undercharge_raw"].shift(1).fillna(0)
    next_flag = grp["undercharge_raw"].shift(-1)
    next_real = grp["is_real_day"].shift(-1)
    first_day_of_episode = (dense["undercharge_raw"] == 1) & (prev_flag == 0)
    confirmed_next_day = (next_flag == 1) | (next_real == False)  # noqa: E712 — explicit tri-state check
    dense["undercharge"] = np.where(first_day_of_episode & ~confirmed_next_day.fillna(True), 0, dense["undercharge_raw"]).astype(int)

    # --- channel C: CHRONIC_LEVEL — persistent Z-score deviation vs cluster, same day
    cluster_stats = dense[dense["is_real_day"]].groupby(["day_index", "cluster"])["battery_min"].agg(["median", "std"])
    cluster_stats["std_safe"] = cluster_stats["std"].fillna(0).clip(lower=0.15)
    dense = dense.merge(cluster_stats[["median", "std_safe"]], on=["day_index", "cluster"], how="left")
    by_sn = dense.groupby("sn", sort=False)  # dense was reassigned above — must re-group
    dense["z_battery_min"] = (dense["battery_min"] - dense["median"]) / dense["std_safe"]
    # Cross-sectional Z-score ALONE would permanently flag any device that is structurally
    # lower than its cluster peers but stable over time (the classic false-positive trap —
    # see freeze_device_baselines). Gating on the device's OWN historical floor as well means
    # only a device that is currently below where IT normally sits can trigger this channel.
    dense["cond_chronic"] = (
        (dense["z_battery_min"] < CHRONIC_Z_THRESHOLD) &
        (dense["battery_min"] < (dense["battery_min_baseline"] - CHRONIC_SELF_MARGIN))
    ).astype("Int64").fillna(0)
    dense["chronic_level"] = (by_sn["cond_chronic"].transform(
        lambda s: s.rolling(CHRONIC_PERSISTENCE_DAYS, min_periods=CHRONIC_PERSISTENCE_DAYS).sum()
        >= CHRONIC_PERSISTENCE_DAYS)).astype(int)

    # --- channel D: FLATLINE — near-zero daily oscillation, excluding a saturated (full)
    # battery, which is healthy behaviour with the same "no oscillation" signature
    dense["near_full"] = dense["battery_max"] >= (dense["battery_max_ceiling_seen"] - FLATLINE_NEAR_FULL_MARGIN)
    dense["cond_flatline"] = ((dense["charge_delta"] < FLATLINE_NOISE_FLOOR) & ~dense["near_full"]).astype(int)
    dense["flatline"] = (by_sn["cond_flatline"].transform(
        lambda s: s.rolling(FLATLINE_MIN_STREAK, min_periods=FLATLINE_MIN_STREAK).sum()
        >= FLATLINE_MIN_STREAK)).astype(int)

    # --- channel E: MONOTONIC_ONSET — young devices only, decline since day one, never recovers
    dense["device_age"] = dense["day_index"] - dense["first_seen_day"]
    metrics = dense.groupby("sn", group_keys=False).apply(
        lambda g: pd.DataFrame(dict(zip(["monotonic_streak", "drop_from_peak"], monotonic_decline_metrics(g))),
                                index=g.index))
    dense["monotonic_streak"] = metrics["monotonic_streak"]
    dense["drop_from_peak"] = metrics["drop_from_peak"]
    dense["monotonic_onset"] = (
        (dense["device_age"] <= YOUNG_DEVICE_MAX_AGE_DAYS) &
        (dense["monotonic_streak"] >= MONOTONIC_MIN_STREAK) &
        (dense["drop_from_peak"] >= MONOTONIC_CUMULATIVE_DROP) &
        (dense["battery_max"] < (dense["battery_min_baseline"] + MONOTONIC_ABSOLUTE_GATE))
    ).astype(int)

    # --- channel F: RECURRING_SHUTDOWN — young devices, repeated short-silence cycles that
    # each return HIGHER than before the gap (zero-load recharge), invisible to any rule
    # that treats "returned from silence" as a symptom only when the return is degraded
    events = []
    for sn, g in dense[dense["device_age"] <= (YOUNG_DEVICE_MAX_AGE_DAYS + SHUTDOWN_WINDOW_DAYS)].groupby("sn"):
        g = g.sort_values("day_index")
        real_idx = g.index[g["is_real_day"]].tolist()
        for i in range(1, len(real_idx)):
            prev_day = g.loc[real_idx[i - 1], "day_index"]
            this_day = g.loc[real_idx[i], "day_index"]
            gap = this_day - prev_day - 1
            if SHUTDOWN_GAP_MIN <= gap <= SHUTDOWN_GAP_MAX:
                events.append((sn, this_day))
    events_df = pd.DataFrame(events, columns=["sn", "day_index"]) if events else pd.DataFrame(columns=["sn", "day_index"])
    events_df["is_shutdown_event"] = 1
    dense = dense.merge(events_df, on=["sn", "day_index"], how="left")
    by_sn = dense.groupby("sn", sort=False)  # dense was reassigned above — must re-group
    dense["is_shutdown_event"] = dense["is_shutdown_event"].fillna(0)
    dense["shutdown_cycles_recent"] = by_sn["is_shutdown_event"].transform(
        lambda s: s.rolling(SHUTDOWN_WINDOW_DAYS, min_periods=1).sum())
    dense["recurring_shutdown"] = (
        (dense["device_age"] <= YOUNG_DEVICE_MAX_AGE_DAYS + SHUTDOWN_WINDOW_DAYS) &
        (dense["shutdown_cycles_recent"] >= SHUTDOWN_CYCLES_THRESHOLD)
    ).astype(int)

    # --- final verdict: priority label + full active-channel list -----------------------
    channel_cols = ["shock", "undercharge", "chronic_level", "monotonic_onset", "recurring_shutdown", "flatline"]
    dense["is_anomalous"] = (dense[channel_cols].sum(axis=1) > 0).astype(int)

    conditions = [dense[c.lower()] == 1 for c in
                  ["shock", "undercharge", "chronic_level", "monotonic_onset", "recurring_shutdown", "flatline"]]
    dense["primary_channel"] = np.select(conditions, CHANNEL_PRIORITY, default="NOMINAL")

    def active_list(row):
        active = [name for name, col in zip(
            ["SHOCK", "UNDERCHARGE", "CHRONIC_LEVEL", "MONOTONIC_ONSET", "RECURRING_SHUTDOWN", "FLATLINE"],
            channel_cols) if row[col] == 1]
        return "+".join(active) if active else "NONE"
    dense["active_channels"] = dense.apply(active_list, axis=1)

    result = dense[dense["is_real_day"]][
        ["sn", "day_index", "battery_min", "battery_max", "cluster", "split",
         "is_anomalous", "primary_channel", "active_channels"]
    ].reset_index(drop=True)
    return result


def build_episodes(result):
    """Collapse consecutive anomalous days (same device) into episodes — matches how a
    technician actually reads the output (one ticket per event, not one row per day)."""
    episodes = []
    for sn, g in result[result["is_anomalous"] == 1].groupby("sn"):
        days = sorted(g["day_index"].tolist())
        start = prev = days[0]
        for d in days[1:]:
            if d - prev <= 2:  # small gaps inside an episode don't split it
                prev = d
                continue
            episodes.append((sn, start, prev))
            start = prev = d
        episodes.append((sn, start, prev))
    return pd.DataFrame(episodes, columns=["sn", "start_day", "end_day"])


def run_pipeline(confirmed_fault_episodes=None):
    """confirmed_fault_episodes: [(sn, start_day, end_day), ...] already reviewed and
    confirmed as real faults (see validate.simulate_manual_review) — NOT the full ground
    truth. Empty/None on the very first run, before any review has happened yet."""
    confirmed_fault_episodes = confirmed_fault_episodes or []
    confirmed_days = expand_confirmed_days(confirmed_fault_episodes)

    telemetry, devices = load_daily_dataset()
    cluster_map = fit_clustering(telemetry, devices)
    cluster_map = assign_fallback_clusters(devices, cluster_map)
    baselines = freeze_device_baselines(telemetry, confirmed_days)
    result = run_engine(telemetry, cluster_map, baselines, confirmed_days)
    episodes = build_episodes(result)

    MODELS_DIR.mkdir(exist_ok=True)
    cluster_map.to_pickle(MODELS_DIR / "cluster_map.pkl")
    print(f"Engine run complete: {result['is_anomalous'].sum()} anomalous device-days, "
          f"{len(episodes)} episodes.")
    return result, episodes


if __name__ == "__main__":
    run_pipeline()
