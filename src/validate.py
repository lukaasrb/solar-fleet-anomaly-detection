"""
End-to-end validation against the synthetic ground truth — and a live demonstration of the
baseline-decontamination methodology (see train_pipeline.py docstring, point 3):

  PASS 1 — run the engine with NO decontamination (nothing has been reviewed yet, exactly
           the state of a brand-new deployment).
  REVIEW — simulate a human confirming the episodes the engine already flagged that turn out
           to overlap a real injected fault (never a full ground-truth peek: only what the
           engine itself surfaced gets reviewed, exactly like Verdetti_Manuali in production).
  PASS 2 — rebuild baselines with those confirmed episodes excluded, and measure again.

Precision/recall are computed by OVERLAP, not exact date match (a fault detected one day
late is still "detected", the same convention used against real manual verdicts in
production — an exact-match metric would understate detection quality).
"""
import sqlite3
from pathlib import Path

import pandas as pd

import train_pipeline as tp

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "fleet.db"


def load_ground_truth():
    con = sqlite3.connect(DB_PATH)
    faults = pd.read_sql("SELECT * FROM ground_truth_faults;", con)
    traps = pd.read_sql("SELECT * FROM ground_truth_healthy_traps;", con)
    con.close()
    return faults, traps


def overlaps(a_start, a_end, b_start, b_end):
    return a_start <= b_end and b_start <= a_end


def score(episodes, ground_truth):
    """For each ground-truth fault, was it caught by at least one detected episode on the
    same device with an overlapping date range? For each detected episode, does it overlap
    ANY ground-truth fault on that device (true positive) or none (false positive)?"""
    detected_by_sn = {sn: g[["start_day", "end_day"]].values.tolist()
                       for sn, g in episodes.groupby("sn")}

    recall_hits = 0
    for row in ground_truth.itertuples(index=False):
        candidates = detected_by_sn.get(row.sn, [])
        if any(overlaps(row.start_day, row.end_day, s, e) for s, e in candidates):
            recall_hits += 1
    recall = recall_hits / len(ground_truth) if len(ground_truth) else float("nan")

    gt_by_sn = {sn: g[["start_day", "end_day"]].values.tolist() for sn, g in ground_truth.groupby("sn")}
    tp_episodes, fp_episodes = [], []
    for row in episodes.itertuples(index=False):
        candidates = gt_by_sn.get(row.sn, [])
        if any(overlaps(row.start_day, row.end_day, s, e) for s, e in candidates):
            tp_episodes.append(row)
        else:
            fp_episodes.append(row)
    precision = len(tp_episodes) / len(episodes) if len(episodes) else float("nan")
    return precision, recall, tp_episodes, fp_episodes


def simulate_manual_review(episodes, ground_truth):
    """A human reviews exactly the episodes the model already flagged (never anything it
    missed) and confirms the ones that genuinely overlap an injected fault. Returns the
    confirmed list in the (sn, start_day, end_day) shape train_pipeline expects."""
    gt_by_sn = {sn: g[["start_day", "end_day"]].values.tolist() for sn, g in ground_truth.groupby("sn")}
    confirmed = []
    for row in episodes.itertuples(index=False):
        candidates = gt_by_sn.get(row.sn, [])
        if any(overlaps(row.start_day, row.end_day, s, e) for s, e in candidates):
            confirmed.append((row.sn, row.start_day, row.end_day))
    return confirmed


def check_healthy_traps(episodes, traps):
    flagged = set(episodes["sn"]).intersection(set(traps["sn"]))
    return flagged


def report(label, precision, recall, n_episodes, n_flagged_traps, n_traps):
    print(f"\n--- {label} ---")
    print(f"  Episodes detected: {n_episodes}")
    print(f"  Precision: {precision:.1%}   Recall: {recall:.1%}")
    print(f"  'Healthy trap' devices incorrectly flagged: {n_flagged_traps}/{n_traps}")


def main():
    faults, traps = load_ground_truth()

    print("=" * 78)
    print(" PASS 1 — no decontamination (fresh deployment, nothing reviewed yet)")
    print("=" * 78)
    result_1, episodes_1 = tp.run_pipeline(confirmed_fault_episodes=None)
    precision_1, recall_1, tp_1, fp_1 = score(episodes_1, faults)
    trap_hits_1 = check_healthy_traps(episodes_1, traps)
    report("PASS 1", precision_1, recall_1, len(episodes_1), len(trap_hits_1), len(traps))

    confirmed = simulate_manual_review(episodes_1, faults)
    print(f"\nSimulated manual review: {len(confirmed)} of the {len(episodes_1)} detected "
          f"episodes confirmed as real faults (this is the ONLY information PASS 2 gets "
          f"that PASS 1 didn't have).")

    print("\n" + "=" * 78)
    print(" PASS 2 — baselines decontaminated using the confirmed episodes above")
    print("=" * 78)
    result_2, episodes_2 = tp.run_pipeline(confirmed_fault_episodes=confirmed)
    precision_2, recall_2, tp_2, fp_2 = score(episodes_2, faults)
    trap_hits_2 = check_healthy_traps(episodes_2, traps)
    report("PASS 2", precision_2, recall_2, len(episodes_2), len(trap_hits_2), len(traps))

    print("\n" + "=" * 78)
    print(" SUMMARY")
    print("=" * 78)
    print(f"  Precision: {precision_1:.1%} -> {precision_2:.1%}   "
          f"({'+' if precision_2 >= precision_1 else ''}{(precision_2 - precision_1) * 100:.1f} pts)")
    print(f"  Recall:    {recall_1:.1%} -> {recall_2:.1%}   "
          f"({'+' if recall_2 >= recall_1 else ''}{(recall_2 - recall_1) * 100:.1f} pts)")
    print(f"  False positive episodes: {len(fp_1)} -> {len(fp_2)}")

    print("\n--- Per-channel breakdown (PASS 2, primary channel of each detected episode) ---")
    primary_by_episode = (
        result_2[result_2["is_anomalous"] == 1]
        .sort_values("day_index")
        .groupby("sn")["primary_channel"]
        .agg(lambda s: s.value_counts().idxmax())
    )
    ep2 = episodes_2.copy()
    ep2["primary_channel"] = ep2["sn"].map(primary_by_episode)
    gt_by_sn = {sn: g[["start_day", "end_day"]].values.tolist() for sn, g in faults.groupby("sn")}
    ep2["is_tp"] = ep2.apply(
        lambda r: any(overlaps(r["start_day"], r["end_day"], s, e) for s, e in gt_by_sn.get(r["sn"], [])), axis=1)
    print(ep2.groupby("primary_channel")["is_tp"].agg(["sum", "count"]).rename(
        columns={"sum": "true_positives", "count": "total_episodes"}).to_string())

    print("\n--- Recall by injected fault type ---")
    fault_hit = []
    detected_by_sn = {sn: g[["start_day", "end_day"]].values.tolist() for sn, g in episodes_2.groupby("sn")}
    for row in faults.itertuples(index=False):
        hit = any(overlaps(row.start_day, row.end_day, s, e) for s, e in detected_by_sn.get(row.sn, []))
        fault_hit.append((row.fault_type, hit))
    fault_df = pd.DataFrame(fault_hit, columns=["fault_type", "detected"])
    print(fault_df.groupby("fault_type")["detected"].agg(["sum", "count"]).rename(
        columns={"sum": "detected", "count": "total"}).to_string())


if __name__ == "__main__":
    main()
