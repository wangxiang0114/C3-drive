"""Recompute the manuscript's synthetic pair comparison and gate counterfactual."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
import json

RECORDS = Path(__file__).with_name("records")


def grouped_rows(path):
    groups = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            groups[int(row["run"])].append(row)
    if set(groups) != set(range(1, 13)):
        raise ValueError(f"{path} must contain runs 1 through 12.")
    return groups.values()


def summarize(records=RECORDS):
    selected, uniform = [], []
    for group in grouped_rows(records / "all_pair_validation_metrics.csv"):
        chosen = [row for row in group if row["functional_selected"] == "True"]
        pairs = {(row["pair_i"], row["pair_j"]) for row in group}
        if len(group) != 6 or len(pairs) != 6 or len(chosen) != 1 or len({row["old_hash"] for row in group}) != 1:
            raise ValueError("Each run must contain six distinct pairs from one checkpoint and one selected pair.")
        selected.append(float(chosen[0]["worst_historical_drop"]))
        uniform.append(mean(float(row["worst_historical_drop"]) for row in group))

    forced_losses, gated_losses, forced_gains, gated_gains = [], [], [], []
    proposals = 0
    for group in grouped_rows(records / "canonical_gate_clause_audit.csv"):
        if len({row["event"] for row in group}) != len(group):
            raise ValueError("Gate events must be unique within each run.")
        proposals += len(group)
        losses = [max(0.0, float(row["audit_history_drop"])) for row in group]
        gains = [float(row["audit_current_gain"]) for row in group]
        accepted = [int(row["passed"]) for row in group]
        if set(accepted) - {0, 1}:
            raise ValueError("Expected binary gate decisions.")
        forced_losses.append(mean(losses))
        gated_losses.append(mean(loss * keep for loss, keep in zip(losses, accepted)))
        forced_gains.append(mean(gains))
        gated_gains.append(mean(gain * keep for gain, keep in zip(gains, accepted)))

    return {
        "evidence": "synthetic_tracking_score",
        "runs": len(selected),
        "pair_candidates": len(selected) * 6,
        "pair_mean_worst_historical_loss": {"functional": mean(selected), "uniform": mean(uniform)},
        "gate_proposals": proposals,
        "gate_mean_historical_loss": {"forced": mean(forced_losses), "accepted_policy": mean(gated_losses)},
        "gate_mean_current_gain": {"forced": mean(forced_gains), "accepted_policy": mean(gated_gains)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=RECORDS)
    args = parser.parse_args()
    print(json.dumps(summarize(args.records), indent=2, allow_nan=False))
