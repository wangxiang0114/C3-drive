"""Evaluate a calibrated drift detector on recorded, held-out episodes."""

from statistics import median

from sklearn.metrics import roc_auc_score

from .data import check_split_overlap, episodes
from .detector import Detector, episode_blocks
from .reporting import write_json
from .train import device_for, load


def trace_episode(vectors, state):
    detector = Detector(state["profiles"], state["domain_slots"], state["block_size"], state["threshold"])
    trace = []
    first_alarm = None
    for index, vector in enumerate(vectors, 1):
        alarm = detector.route_block(vector)
        if alarm and first_alarm is None:
            first_alarm = index
        trace.append({"block": index, "evidence": detector.evidence, "slot": detector.slot})
    return {
        "blocks": len(trace),
        "peak_evidence": detector.peak,
        "first_alarm_block": first_alarm,
        "trace": trace,
    }


def summarize(clean, shifted):
    clean_alarms = sum(row["first_alarm_block"] is not None for row in clean)
    alarms = [row["first_alarm_block"] for row in shifted if row["first_alarm_block"] is not None]
    return {
        "clean_episodes": len(clean),
        "shifted_episodes": len(shifted),
        "clean_false_alarm_rate": clean_alarms / len(clean),
        "shift_detection_rate": len(alarms) / len(shifted),
        "median_first_alarm_block": median(alarms) if alarms else None,
        "auroc_peak_evidence": float(roc_auc_score(
            [0] * len(clean) + [1] * len(shifted),
            [row["peak_evidence"] for row in [*clean, *shifted]],
        )),
    }


def run_drift_evaluation(args):
    domains = list(dict.fromkeys(["clean", *args.domains]))
    if len(domains) == 1:
        raise ValueError("Select at least one shifted domain for comparison with clean episodes.")
    groups = {domain: episodes(args.data, domain, "test")[:args.limit] for domain in domains}
    protected = [args.checkpoint, *(path for paths in groups.values() for path in paths)]
    if args.output.resolve() in {path.resolve() for path in protected}:
        raise ValueError("Choose a report path separate from the checkpoint and episode files.")
    check_split_overlap([(domain, "test", paths) for domain, paths in groups.items()])
    model, state = load(args.checkpoint, device_for(args.device))
    if "profiles" not in state:
        raise ValueError("Use a calibrated checkpoint to evaluate its drift detector.")
    scores = {}
    skipped = {}
    for domain, paths in groups.items():
        scores[domain] = []
        skipped[domain] = []
        for path in paths:
            vectors = episode_blocks(model, path, state["block_size"], args.batch_size)
            item = path.relative_to(args.data).as_posix()
            if len(vectors):
                scores[domain].append({"episode": item, **trace_episode(vectors, state)})
            else:
                skipped[domain].append(item)
        if not scores[domain]:
            raise ValueError(f"No complete detector blocks in {domain}/test.")
    write_json(args.output, {
        "checkpoint": str(args.checkpoint),
        "checkpoint_stage": state.get("stage"),
        "split": "test",
        "block_size": state["block_size"],
        "threshold": state["threshold"],
        "comparisons": {domain: summarize(scores["clean"], scores[domain]) for domain in domains[1:]},
        "episodes": scores,
        "skipped_incomplete_episodes": skipped,
    })
    print(args.output, flush=True)
