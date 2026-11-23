"""Assemble plotting inputs from local run files."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

import improve
import long_boundary
import quota_revision


def load_doc(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def final_score(path):
    return np.mean([row[5] for row in load_doc(path)["stages"] if row[2] == 11])


def order_scores(score):
    runs = improve.OUT / "runs"
    quota_runs = quota_revision.OUT / "runs"
    values = np.zeros((3, 3, 12))
    for i, order in enumerate(improve.ORDERS):
        for j in range(12):
            run = j + 1
            values[i, 0, j] = final_score(runs / f"order_{order}_full_{run:02d}/result.json.gz")
            values[i, 1, j] = final_score(quota_runs / f"{order}_quota_risk_{run:02d}/result.json.gz")
            values[i, 2, j] = (score[j, 1, -1].mean() if order == "default" else
                               final_score(runs / f"order_{order}_random_memory_{run:02d}/result.json.gz"))
    return values


def long_metrics():
    names = [row[0] for row in long_boundary.CONFIGS]
    score = np.zeros((12, len(names), 18, 15))
    resources = {name: np.zeros(score.shape[:3]) for name in
                 ("network_bytes", "router_detector_bytes", "replay_bytes", "total_bytes")}
    counts = np.zeros((len(names), 18))
    rejected = np.zeros_like(counts)
    for i, name in enumerate(names):
        for j in range(12):
            doc = load_doc(long_boundary.OUT / "runs" / f"{name}_run{j+1:02d}/result.json.gz")
            for row in doc["stages"]:
                score[j, i, row[2], row[4]] = row[5]
            for row in doc["resources"]:
                for key, values in resources.items():
                    values[j, i, row["phase"]] = row[key]
            for event in doc["events"]:
                if event["slots_before"] == 4:
                    phase = event["decision_block"] // 64
                    counts[i, phase] += 1
                    rejected[i, phase] += not event["passed"]
    current = np.stack([score[:, :, p, k] for p, k in enumerate(long_boundary.SEQUENCE)], axis=2)
    historical = np.zeros(score.shape[:3])
    for phase in range(1, 18):
        domains = sorted(set(long_boundary.SEQUENCE[:phase]))
        drops = [score[:, :, long_boundary.SEQUENCE.index(domain):phase, domain].max(2)
                 - score[:, :, phase, domain] for domain in domains]
        historical[:, :, phase] = np.maximum(0, np.stack(drops, axis=-1).max(-1))
    modes = [names.index(name) for name in ("M4_B400", "M4_B400_strict", "M4_B400_no_gate")]
    rates = np.divide(rejected, counts, out=np.full_like(counts, np.nan), where=counts > 0)
    return {"long_scores": score, "current": current, "historical": historical,
            "gate_rejection_rates": rates[modes], **resources}


def detector_metrics():
    names = ("appearance", "hidden", "coupled", "gradual")
    with np.load(improve.OUT / "single_block_calibration.npz") as baseline:
        threshold = float(baseline["threshold"])
    docs = {name: [load_doc(improve.OUT / "runs" / f"condition_{name}_full_{i:02d}/result.json.gz")
                   for i in range(1, 13)] for name in names}

    def window(doc, start, end):
        blocks = doc["blocks"][start:end]
        sequential = [row[2] + 1 - start for row in blocks if row[9]]
        single = [row[2] + 1 - start for row in blocks if row[-1] > threshold]
        events = [event for event in doc["events"] if start <= event["trigger_block"] < end]
        return (bool(sequential), any(event["trigger_block"] + 9 < end for event in events),
                any(event["passed"] and event["decision_block"] < end for event in events),
                next(iter(sequential), 64), next(iter(single), 64))

    return {
        "detector_outcomes": np.array([[window(doc, 64, 128) for doc in docs[name]] for name in names]),
        "normal_outcomes": np.array([window(doc, 0, 64) for doc in docs["coupled"]]),
        "detector_traces": np.array([[[row[i] for i in (2, 7, 8, 10, 15)]
                                      for row in docs[name][0]["blocks"][:128]]
                                     for name in ("coupled", "hidden")]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file.")
    with np.load(args.replay_arrays) as replay:
        arrays = {key: replay[key] for key in ("score", "success", "coverage")}
    arrays.update(order_scores=order_scores(arrays["score"]))
    arrays.update(long_metrics())
    arrays.update(detector_metrics())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    print(args.output)


if __name__ == "__main__":
    main()
