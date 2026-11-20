"""Run the paper's sequential evaluation protocol with the local CARLA workflow."""

import hashlib
import json
import math
import os
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .adapt import continual, metrics, rollouts
from .data import check_route, episodes
from .reporting import score_summary, write_json, write_score_matrix
from .train import device_for, load


@contextmanager
def run_lock(directory):
    """Exclude another writer and release the lock even if the process exits."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".benchmark.lock").open("a+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another benchmark is writing to {directory}.") from exc
        yield


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def recording_inventory(root, manifest, domains):
    """Check that recordings belong to their stated manifest splits."""
    route_splits = {split: {(r["start"], r["end"]) for r in routes}
                    for split, routes in manifest["routes"].items()}
    inventory = []
    seen = {}
    for domain in domains:
        splits = ("train", "validation", "fit", "calibrate", "threshold") if domain == "clean" else (
            "train", "fit", "calibrate"
        )
        for split in splits:
            for path in episodes(root, domain, split):
                with np.load(path, allow_pickle=False) as archive:
                    metadata = check_route(archive, path, domain, split, seen)
                route = metadata["route"]
                if metadata["town"] != manifest["town"] or (route["start"], route["end"]) not in route_splits[split]:
                    raise ValueError(f"{path} is outside the benchmark manifest's {split} split.")
                stat = path.stat()
                inventory.append([path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns])
    return inventory


def configuration(args, manifest, domains):
    keys = ("batch_size", "candidates", "host", "port", "ticks", "traffic", "limit", "seed",
            "steps", "lr", "min_gain", "max_score_drop", "max_completion_drop", "max_collision_increase")
    for key in keys[-4:]:
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{key.replace('_', '-')} must be finite and nonnegative.")
    for split in ("stream", "gate", "test"):
        if args.limit is not None and args.limit > len(manifest["routes"][split]):
            raise ValueError(f"--limit exceeds the number of routes in {split}.")
    return {
        "version": 1,
        "protocol": "c3drive_carla_closed_loop",
        "domains": domains,
        "town": manifest["town"],
        "device": str(device_for(args.device)),
        "carla_root": args.carla_root or os.environ.get("CARLA_ROOT"),
        "data": str(args.data.resolve()),
        "recordings": recording_inventory(args.data, manifest, domains),
        "inputs": {key: {"path": str(getattr(args, key).resolve()), "sha256": sha256(getattr(args, key))}
                   for key in ("checkpoint", "teacher", "manifest")},
        "settings": {key: getattr(args, key) for key in keys},
    }


def checkpoint_resources(model, state, checkpoint):
    params = list(model.parameters())
    replay = state["memory"]
    profiles = state["profiles"]

    def tensor_bytes(groups):
        return sum(t.numel() * t.element_size() for group in groups.values() for t in group.values())

    return {
        "model_parameters": sum(p.numel() for p in params),
        "model_parameter_bytes": sum(p.numel() * p.element_size() for p in params),
        "adapter_parameters": sum(p.numel() for p in model.adapters.parameters()),
        "adapter_slots": model.spec["slots"],
        "domain_slots": state["domain_slots"],
        "occupied_slots": len({slot for slot in state["domain_slots"].values() if slot >= 0}),
        "replay_budget": state["memory_budget"],
        "replay_items_by_domain": {domain: len(group["feature"]) for domain, group in replay.items()},
        "replay_tensor_bytes": tensor_bytes(replay),
        "profile_tensor_bytes": tensor_bytes(profiles),
        "checkpoint_file_bytes": checkpoint.stat().st_size,
    }


def evaluate(env, manifest, model, state, domains, args):
    result = {}
    for domain in domains:
        routes = rollouts(env, manifest, model, state, domain, "test", args)
        result[domain] = {"metrics": metrics(routes), "routes": routes}
    return result


def stage_directory(directory, index, domain):
    return directory / f"stage-{index:02d}-{domain}"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def completed_stages(directory, domains, initial):
    reports = []
    active = initial
    for index, domain in enumerate(domains):
        stage = stage_directory(directory, index, domain)
        if not stage.exists():
            break
        record = read_json(stage / "stage.json")
        if record["index"] != index or record["arrival_domain"] != domain or (
            set(record["test"]) != set(domains[:index + 1])
        ):
            raise ValueError(f"Invalid benchmark stage record: {stage}.")
        if record["accepted"] is True:
            active = stage / "accepted.pt"
        if sha256(active) != record["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint no longer matches {stage / 'stage.json'}.")
        reports.append(record)
    if len(list(directory.glob("stage-*"))) != len(reports):
        raise ValueError("Stage directories do not form a consecutive prefix of the assigned domain order.")
    return reports, active


def run_stage(directory, index, domains, active, args, env, manifest):
    domain = domains[index]
    stage = stage_directory(directory, index, domain)
    with TemporaryDirectory(prefix=".stage-", dir=directory) as scratch:
        pending = Path(scratch) / "stage"
        pending.mkdir()
        accepted_path = pending / "accepted.pt"
        device = device_for(args.device)
        model, state = load(active, device)
        accepted = None
        before = None
        reason = "initial_checkpoint"
        if index:
            before = evaluate(env, manifest, model, state, [domain], args)[domain]
            del model, state
            decision_path = pending / "decision.json"
            update = Namespace(**(vars(args) | {
                "checkpoint": active, "output": accepted_path, "report": decision_path, "domain": domain,
            }))
            continual(update, env, manifest)
            decision = read_json(decision_path)
            accepted, reason = decision["accepted"], decision["reason"]
            if accepted != accepted_path.is_file():
                raise RuntimeError(f"Gate decision and checkpoint disagree for {domain}.")
        checkpoint = accepted_path if accepted else active
        if index:
            model, state = load(checkpoint, device)
        if accepted is False:
            test = evaluate(env, manifest, model, state, domains[:index], args)
            test[domain] = before
        else:
            test = evaluate(env, manifest, model, state, domains[:index + 1], args)
        record = {
            "index": index,
            "arrival_domain": domain,
            "accepted": accepted,
            "reason": reason,
            "checkpoint_stage": state["stage"],
            "checkpoint_sha256": sha256(checkpoint),
            "resources": checkpoint_resources(model, state, checkpoint),
            "pre_update_test": before,
            "test": test,
        }
        write_json(pending / "stage.json", record)
        pending.rename(stage)
    return record, stage / "accepted.pt" if accepted else active


def publish_summary(directory, domains, reports):
    matrix = [{domain: evaluation["metrics"]["score"] for domain, evaluation in stage["test"].items()}
              for stage in reports]
    summaries = []
    for index, stage in enumerate(reports):
        domain = domains[index]
        before = stage["pre_update_test"]
        summaries.append({
            "index": index,
            "arrival_domain": domain,
            "accepted": stage["accepted"],
            "reason": stage["reason"],
            "checkpoint_stage": stage["checkpoint_stage"],
            "stage_report": (stage_directory(Path(), index, domain) / "stage.json").as_posix(),
            "current_domain_gain": matrix[index][domain] - before["metrics"]["score"] if before else None,
            **score_summary(matrix[:index + 1], domains),
        })
    write_score_matrix(directory / "scores.csv", domains, matrix)
    write_json(directory / "report.json", {
        "score": "c3drive_route_progress_with_collision_and_lane_penalties",
        "domains": domains,
        "completed": len(reports) == len(domains),
        "score_matrix": [{domain: row.get(domain) for domain in domains} for row in matrix],
        "stages": summaries,
    })


def run_benchmark(args, manifest, connect):
    domains = ["clean", *args.domains]
    if len(set(domains)) != len(domains):
        raise ValueError("Each benchmark arrival domain must appear once.")
    directory = args.output.resolve()
    initial = args.checkpoint.resolve()
    with run_lock(directory):
        config_path = directory / "config.json"
        if not args.resume and any(p.name != ".benchmark.lock" for p in directory.iterdir()):
            raise FileExistsError("Choose a new benchmark output directory or use --resume.")
        config = configuration(args, manifest, domains)
        if config_path.exists():
            if read_json(config_path) != config:
                raise ValueError("Benchmark inputs or settings changed. Use a new output directory.")
        else:
            if any(p.name != ".benchmark.lock" for p in directory.iterdir()):
                raise ValueError("The output directory has files but no benchmark config.")
            _, initial_state = load(initial, "cpu")
            teacher, _ = load(args.teacher, "cpu")
            if (initial_state.get("stage") != 0 or initial_state.get("domain_slots") != {"clean": -1}
                    or "profiles" not in initial_state):
                raise ValueError("Use a clean stage-0 checkpoint produced by calibrate.")
            if teacher.spec["slots"] or initial_state["spec"]["slots"] < 2:
                raise ValueError("Use a teacher and a calibrated student with at least two adapter slots.")
            if initial_state["memory_budget"] < len(domains):
                raise ValueError("The initial replay budget must allow at least one item per scheduled domain.")
            write_json(config_path, config)
        reports, active = completed_stages(directory, domains, initial)
        if len(reports) < len(domains):
            env = connect(args, manifest["town"])
            for index in range(len(reports), len(domains)):
                print(f"Benchmark stage {index}: {domains[index]}", flush=True)
                record, active = run_stage(directory, index, domains, active, args, env, manifest)
                reports.append(record)
                publish_summary(directory, domains, reports)
        else:
            publish_summary(directory, domains, reports)
    print(directory / "report.json", flush=True)
