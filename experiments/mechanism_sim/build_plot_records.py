"""Build figure input files from local logs and checkpoints."""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import improve
import numpy as np
import quota_revision

from collect_plot_inputs import load_doc

REPLAY_VARIANTS = ("full", "random_memory", "coverage_only", "pareto_only")
QUOTA_VARIANTS = ("quota_risk", "quota_random")


def restore(engine, arrays, commit):
    prefix = f"commit{commit}_"
    model = engine.Model(arrays["base"], arrays["U"])
    model.V = arrays[prefix + "V"]
    model.occupied = arrays[prefix + "occupied"]
    model.slot_age = arrays[prefix + "slot_age"]
    model.routes = [
        {
            "mu": mu,
            "slot": int(arrays[prefix + "route_slot"][i]),
            "precision": arrays[prefix + "route_precision"][i],
            "cal": arrays[prefix + f"route_cal_{i}"],
            "source_domain": int(arrays[prefix + "route_sources"][i]),
        }
        for i, mu in enumerate(arrays[prefix + "route_mu"])
    ]
    return model


def compare_pairs(engine, run, event, model, memory, samples, registry, world):
    block, decision, domain = (event[k] for k in ("trigger_block", "decision_block", "training_domain"))
    history = sorted(({0} | {route["source_domain"] for route in model.routes}) - {domain})
    domains = list(range(len(world)))
    before = engine.evaluate(model, world, domains, run, 92, decision)
    rows = []
    try:
        for i, j in combinations(np.flatnonzero(model.occupied).tolist(), 2):
            engine.MERGE_STRATEGY = (i, j)
            candidate = engine.propose(model, memory, samples, registry, world, domain, run, block, "full")
            if (candidate["pair_i"], candidate["pair_j"]) != (i, j) or candidate["candidate"] is None:
                raise ValueError(f"Run {run} cannot evaluate adapter pair {(i, j)}.")
            after = engine.evaluate(candidate["candidate"], world, domains, run, 92, decision)
            gate, _, _ = engine.gate_metrics(model, candidate["candidate"], world, domain, run, decision)
            rows.append({
                "run": run, "trigger_block": block, "decision_block": decision, "old_hash": model.fingerprint(),
                "pair_i": i, "pair_j": j, "functional_distance": float(candidate["functional_distance"]),
                "worst_historical_drop": float(max(before[k][0] - after[k][0] for k in history)),
                "current_gain": float(after[domain][0] - before[domain][0]),
                "accepted_after_both_gates": int(candidate["accepted"] and gate["passed"]),
            })
    finally:
        engine.MERGE_STRATEGY = None
    best = min(rows, key=lambda row: row["functional_distance"])
    for row in rows:
        row["functional_selected"] = row is best
    return rows


def same_pool(engine, run, event, model, proposal, quota=False):
    pool, candidate = proposal["pool"], proposal["candidate"]
    rows, summary = [], []
    for variant in QUOTA_VARIANTS if quota else REPLAY_VARIANTS:
        if quota:
            rng = engine.rng_for(run, 40, event["trigger_block"], 1)
            selected, _ = quota_revision.selector(engine, pool, model, candidate, variant, rng)
        else:
            rng = engine.rng_for(run, 40, event["trigger_block"], len(proposal["attempt_logs"]))
            selected, detail = engine.select_memory(pool, model, candidate, variant, rng)
            keep = detail["selected"]
            risk = detail["retention_risk"]
            features = detail["features"]
            cover = ((features[:, None] - features[keep][None]) ** 2).sum(2).min(1)
            summary.append({
                "variant": variant, "pool_C1_fraction": float(pool["C"].mean()),
                "selected_C1_fraction": float(pool["C"][keep].mean()),
                "pool_severity_positive_fraction": float((pool["S"] > 0).mean()),
                "selected_severity_positive_fraction": float((pool["S"][keep] > 0).mean()),
                "pool_risk_positive_fraction": float((risk > 0).mean()),
                "selected_risk_positive_fraction": float((risk[keep] > 0).mean()),
                "selected_current_fraction": float((pool["source"][keep] == event["training_domain"]).mean()),
                "mean_nearest_selected_squared_distance": float(cover.mean()),
            })
        for domain in np.unique(pool["source"]):
            available = int((pool["source"] == domain).sum())
            kept = int((selected["source"] == domain).sum())
            rows.append({
                "run": run, "event": event["event"], "variant": variant, "domain": int(domain),
                "pool_count": available, "selected_count": kept, "retention_fraction": kept / available,
            })
    return rows, summary


def collect_run(run):
    directory = improve.OUT / "runs" / f"order_default_full_{run:02d}"
    log = load_doc(directory / "result.json.gz")
    if len(log["events"]) != len(log["audits"]):
        raise ValueError(f"Run {run} is missing matched event audits.")
    gates = []
    for event, audit in zip(log["events"], log["audits"]):
        if (audit["block"], bool(audit["accepted"]), audit["old_hash"]) != (
            event["decision_block"], bool(event["passed"]), event["model_before"]
        ):
            raise ValueError(f"Run {run}, event {event['event']} has mismatched audit metadata.")
        gates.append({
            "run": run, "event": event["event"], "passed": event["passed"],
            "audit_history_drop": audit["historical_max_drop"], "audit_current_gain": audit["current_gain"],
            "slots_before": event["slots_before"], "merge_committed": event["merge_committed"],
        })

    engine = improve.load_engine()
    world = engine.worlds(run)
    memory = engine.empty_memory()
    commit = 0
    with np.load(directory / "data" / f"checkpoint_run{run:02d}_full.npz") as arrays:
        for event in log["events"]:
            model = restore(engine, arrays, commit)
            domain, block = event["training_domain"], event["trigger_block"]
            samples, registry = engine.collect(model, world, domain, run, block)
            proposal = engine.propose(model, memory, samples, registry, world, domain, run, block, "full")
            if model.occupied.sum() == engine.M:
                pairs = compare_pairs(engine, run, event, model, memory, samples, registry, world)
                replay_rows, composition = same_pool(engine, run, event, model, proposal)
                quota_rows, _ = same_pool(engine, run, event, model, proposal, quota=True)
                return gates, pairs, replay_rows, quota_rows, composition
            if event["passed"]:
                commit += 1
                memory = proposal["memory"]
                if not np.array_equal(memory["id"], arrays[f"commit{commit}_memory_ids"]):
                    raise ValueError(f"Run {run}, commit {commit} does not reconstruct replay memory.")
    raise ValueError(f"Run {run} has no full adapter pool.")


def write_records(output):
    output.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=3) as pool:
        runs = list(pool.map(collect_run, range(1, 13)))
    files = ("canonical_gate_clause_audit.csv", "all_pair_validation_metrics.csv",
             "same_pool_domain_retention.csv", "quota_same_pool.csv")
    for index, file in enumerate(files):
        rows = [row for run in runs for row in run[index]]
        with (output / file).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)

    functional = [next(row["worst_historical_drop"] for row in pairs if row["functional_selected"])
                  for _, pairs, *_ in runs]
    uniform = [float(np.mean([row["worst_historical_drop"] for row in pairs])) for _, pairs, *_ in runs]

    # The base run logs all replay proposals in run/event order. Select the first pool containing a critical event.
    with np.load(improve.OLD / "data" / "memory_candidates.npz") as memory:
        rows = memory["rows"]
        critical = rows[rows[:, 6] == 1]
        if not len(critical):
            raise ValueError("The base run logs contain no critical replay events.")
        run_id, event_id = critical[0, :2]
        pool = rows[(rows[:, 0] == run_id) & (rows[:, 1] == event_id)]
        np.savez_compressed(output / "critical_event_reference.npz",
                            rows=pool[:, [3, 4, 5, 6, 7]], selected=pool[:, 8].astype(bool))
    write_diagnostics(output, runs, functional, uniform)


def write_diagnostics(output, runs, functional, uniform):
    with (output / "all_pair_audit.json").open("w", encoding="utf-8") as handle:
        json.dump({"functional": functional, "uniform_expectation": uniform}, handle, indent=2)
    with (improve.OLD / "data" / "update_events.csv").open(encoding="utf-8-sig", newline="") as handle:
        events = [row for row in csv.DictReader(handle) if row["variant"] == "full"]
    phases = []
    for phase, domain in enumerate(improve.ORDERS["default"]):
        rows = [row for row in events if int(row["decision_block"]) // 64 == phase]
        commits = sum(int(row["passed"]) for row in rows)
        phases.append({"phase": phase + 1, "domain": domain, "commits": commits,
                       "completed_proposals": len(rows), "rejections": len(rows) - commits})
    trace = {}
    for condition in ("coupled", "appearance", "hidden", "gradual"):
        path = improve.OUT / "runs" / f"condition_{condition}_full_01" / "result.json.gz"
        first_shift = [event for event in load_doc(path)["events"] if 64 <= event["trigger_block"] < 128]
        trace[condition] = {"events": [
            {"trigger": event["trigger_time"], "candidate_ready": event["trigger_block"] + 9,
             "validation_state_start": event["trigger_block"] + 11,
             "final_gate_and_commit_decision": event["decision_time"]}
            for event in first_shift
        ]}
    gates = [row for run, *_ in runs for row in run]
    composition = {}
    for variant in REPLAY_VARIANTS:
        rows = [row for run in runs for row in run[4] if row["variant"] == variant]
        composition[variant] = {name: float(np.mean([row[name] for row in rows]))
                                for name in rows[0] if name != "variant"}
    with (output / "protocol_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump({"canonical_phase_counts_30runs": phases, "run1_event_trace": trace,
                   "selection_summary": composition,
                   "canonical_gate_denominators": {
                       "all_completed_proposals": len(gates),
                       "all_commits": sum(row["passed"] for row in gates),
                       "saturated_completed_proposals": sum(row["slots_before"] == 4 for row in gates),
                       "committed_merges": sum(row["merge_committed"] for row in gates),
                   }}, handle, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New local directory for derived figure inputs")
    args = parser.parse_args()
    write_records(args.output)
