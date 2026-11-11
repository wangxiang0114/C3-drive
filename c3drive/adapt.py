import copy
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from .carla_env import Policy
from .data import EpisodeData, check_split_overlap, episodes, move, observations
from .detector import Detector, calibrate_threshold, fit_profile
from .model import behavior_loss
from .train import device_for, load, save, seed_all


def initialize(args):
    if args.output.exists():
        raise FileExistsError("Choose a new calibrated checkpoint output path.")
    model, state = load(args.checkpoint, device_for(args.device))
    if state["spec"]["slots"] < 2:
        raise ValueError("Adaptation requires a student with at least two adapter slots.")
    if "profiles" in state or state["domain_slots"] != {"clean": -1}:
        raise ValueError("Initialize the detector from the base student checkpoint.")
    paths = {split: episodes(args.data, "clean", split) for split in ("train", "validation", "fit", "calibrate", "threshold")}
    check_split_overlap([("clean", split, files) for split, files in paths.items()])
    profiles = {"clean": profile(args, model, "clean")}
    threshold = threshold_for(args, model, profiles, state["domain_slots"])
    memory = {"clean": candidate_features(model, model, paths["train"], max(1, args.memory_budget // 2),
                                           args.batch_size)}
    save(args.output, model, profiles=profiles, threshold=threshold, domain_slots=state["domain_slots"],
         block_size=args.block_size, alpha=args.alpha, memory=memory, memory_budget=args.memory_budget,
         stage=0)
    print(f"Detector threshold: {threshold:.6f}", flush=True)


def profile(args, model, domain):
    return fit_profile(model, episodes(args.data, domain, "fit"), episodes(args.data, domain, "calibrate"),
                       args.block_size, args.batch_size)


def threshold_for(args, model, profiles, mapping):
    return calibrate_threshold(model, profiles, mapping, episodes(args.data, "clean", "threshold"),
                               args.block_size, args.batch_size, args.alpha)


def policy_for(model, state, callback=None):
    detector = None
    if "profiles" in state:
        detector = Detector(state["profiles"], state["domain_slots"], state["block_size"], state["threshold"])
    return Policy(model, detector, callback)


def rollouts(env, manifest, model, state, domain, split, args):
    if args.limit and args.limit > len(manifest["routes"][split]):
        raise ValueError(f"Only {len(manifest['routes'][split])} {split} routes available.")
    return [env.run(route, domain, args.ticks, args.traffic, policy_for(model, state))
            for route in manifest["routes"][split][:args.limit or None]]


def metrics(rows):
    return {
        "score": float(np.mean([r["score"] for r in rows])),
        "completion": float(np.mean([r["completion"] for r in rows])),
        "collision_rate": float(np.mean([r["collisions"] > 0 for r in rows])),
        "success_rate": float(np.mean([r["success"] for r in rows])),
    }


def passes_regression(before, after, args):
    return (after["score"] >= before["score"] - args.max_score_drop
            and after["completion"] >= before["completion"] - args.max_completion_drop
            and after["collision_rate"] <= before["collision_rate"] + args.max_collision_increase)


@torch.no_grad()
def candidate_features(model, teacher, paths, limit, batch_size):
    data = EpisodeData(paths)
    device = next(model.parameters()).device
    values = {}
    indices = np.linspace(0, len(data) - 1, min(limit, len(data)), dtype=int)
    for start in range(0, len(indices), batch_size):
        samples = [data[i] for i in indices[start:start + batch_size]]
        batch = move({key: torch.stack([item[key] for item in samples]) for key in samples[0]}, device)
        feature = model.encode(*observations(batch))
        soft, _ = teacher(*observations(batch, clean=True))
        for key, tensor in dict(feature=feature, teacher=soft, target=batch["target"],
                                severity=batch["severity"], critical=batch["critical"]).items():
            values.setdefault(key, []).append(tensor.cpu())
    return {key: torch.cat(parts) for key, parts in values.items()}


def concat(groups):
    return {key: torch.cat([group[key] for group in groups]) for key in groups[0]}


def subset(group, indices):
    return {key: values[indices] for key, values in group.items()}


def pareto_coreset(group, risks, budget):
    if len(risks) <= budget:
        return group
    values = np.column_stack((group["critical"], group["severity"], risks))
    features = group["feature"].numpy()
    features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    dominates = (values[:, None, :] >= values[None, :, :]).all(axis=2)
    dominates &= (values[:, None, :] > values[None, :, :]).any(axis=2)
    remaining = np.arange(len(values))
    selected = []
    while len(selected) < budget:
        front = remaining[~dominates[np.ix_(remaining, remaining)].any(axis=0)]
        space = budget - len(selected)
        if len(front) <= space:
            selected.extend(front.tolist())
            remaining = remaining[~np.isin(remaining, front)]
            continue
        center = features[selected] if selected else features[front].mean(axis=0, keepdims=True)
        minimum = np.full(len(front), np.inf)
        for vector in center:
            minimum = np.minimum(minimum, ((features[front] - vector) ** 2).sum(axis=1))
        for _ in range(space):
            position = int(minimum.argmax())
            index = front[position]
            selected.append(int(index))
            minimum = np.minimum(minimum, ((features[front] - features[index]) ** 2).sum(axis=1))
            minimum[position] = -1
    return subset(group, selected)


def optimize_adapter(model, slot, groups, args, replay=()):
    device = next(model.parameters()).device
    adapter = model.adapters[slot]
    model.requires_grad_(False)
    adapter.requires_grad_(True)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    for _ in range(args.steps):
        losses = []
        for group, label, weight in [*groups, *replay]:
            indices = torch.randint(len(group["feature"]), (args.batch_size,))
            features = group["feature"][indices].to(device)
            target = group[label][indices].to(device)
            losses.append(weight * behavior_loss(model.plan(features, slot), target).mean())
        optimizer.zero_grad(set_to_none=True)
        torch.stack(losses).sum().backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
        optimizer.step()
    return model.eval()


@torch.no_grad()
def predictions(model, group, slot):
    return model.plan(group["feature"].to(next(model.parameters()).device), slot).cpu()


def consolidation_pairs(model, mapping, memory):
    occupied = sorted({slot for slot in mapping.values() if slot >= 0})
    by_slot = {slot: [d for d in memory if mapping[d] == slot] for slot in occupied}
    pairs = []
    for left, right in itertools.combinations(occupied, 2):
        domains = by_slot[left] + by_slot[right]
        if not by_slot[left] or not by_slot[right]:
            continue
        count = min(64, *(len(memory[d]["feature"]) for d in domains))
        probe = concat([subset(memory[d], slice(count)) for d in domains])
        distance = behavior_loss(predictions(model, probe, left), predictions(model, probe, right)).mean().item()
        pairs.append((distance, left, right, domains))
    return sorted(pairs)


def update_memory(previous, candidate, old_mapping, mapping, memory, new_domain, new_data, budget):
    domains = [*memory, new_domain]
    if len(domains) > budget:
        raise ValueError("Memory budget must support at least one item per admitted domain.")
    combined = []
    risks = []
    for index, domain in enumerate(domains):
        data = new_data if domain == new_domain else memory[domain]
        old = predictions(previous, data, old_mapping.get(domain, -1))
        new = predictions(candidate, data, mapping[domain])
        target = data["teacher"] if domain == new_domain else data["target"]
        risks.append((behavior_loss(new, target) - behavior_loss(old, target)).clamp_min(0).numpy())
        combined.append(dict(**data, domain=torch.full((len(old),), index, dtype=torch.int64)))
    pool = concat(combined)
    risk = np.concatenate(risks)
    if len(risk) <= budget:
        selected = pool
    else:
        # Reserve one Pareto representative per domain so future consolidation has a probe.
        anchors = []
        for index in range(len(domains)):
            candidates = torch.where(pool["domain"] == index)[0]
            one = dict(subset(pool, candidates), row=candidates)
            anchors.append(int(pareto_coreset(one, risk[candidates.numpy()], 1)["row"][0]))
        mask = ~np.isin(np.arange(len(risk)), anchors)
        chosen = subset(pool, anchors)
        if budget > len(anchors):
            chosen = concat([chosen, pareto_coreset(subset(pool, mask), risk[mask], budget - len(anchors))])
        selected = chosen
    membership = selected.pop("domain")
    return {domain: subset(selected, membership == index) for index, domain in enumerate(domains)}


def continual(args, env, manifest):
    seed_all(args.seed)
    device = device_for(args.device)
    previous, state = load(args.checkpoint, device)
    teacher, _ = load(args.teacher, device)
    if teacher.spec["slots"] != 0:
        raise ValueError("--teacher must point to a teacher checkpoint.")
    mapping = state["domain_slots"]
    domain = args.domain
    if Path(args.output).resolve() == Path(args.checkpoint).resolve() or Path(args.output).exists():
        raise ValueError("Choose a new output path for the candidate checkpoint.")
    if args.report.resolve() in {args.output.resolve(), args.checkpoint.resolve(), args.teacher.resolve()}:
        raise ValueError("Choose a report path separate from the checkpoints.")
    if domain in mapping:
        raise ValueError(f"{domain} already belongs to the accepted deployment domains.")
    if "profiles" not in state:
        raise ValueError("Initialize the detector before running a continual update.")
    if len(mapping) + 1 > state["memory_budget"]:
        raise ValueError("The replay budget needs at least one item for each admitted domain.")
    domain_paths = {split: episodes(args.data, domain, split) for split in ("train", "fit", "calibrate")}
    check_split_overlap([(domain, split, paths) for split, paths in domain_paths.items()])
    args.block_size, args.alpha = state["block_size"], state["alpha"]
    decision = {"domain": domain, "accepted": False, "attempts": []}

    def report(reason):
        decision["reason"] = reason
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(decision, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(decision, indent=2), flush=True)

    triggered = False

    def record_trigger():
        nonlocal triggered
        triggered = True

    for route in manifest["routes"]["stream"][:args.limit or None]:
        policy = policy_for(previous, state, record_trigger)
        env.run(route, domain, args.ticks, args.traffic, policy)
        if triggered:
            break
    if not triggered:
        report("no_drift_trigger")
        return
    memory = state["memory"]
    new_data = candidate_features(previous, teacher, domain_paths["train"], args.candidates, args.batch_size)
    baseline = {d: metrics(rollouts(env, manifest, previous, state, d, "gate", args)) for d in [*mapping, domain]}
    new_profile = profile(args, previous, domain)
    free = sorted(set(range(previous.spec["slots"])) - set(mapping.values()))
    options = [(0.0, None, free[0], [])] if free else consolidation_pairs(previous, mapping, memory)
    for distance, left, slot, merged in options:
        candidate = copy.deepcopy(previous)
        proposal = dict(mapping)
        attempt = {"merge_domains": merged, "behavior_distance": distance}
        decision["attempts"].append(attempt)
        if merged:
            groups = []
            for old_domain in merged:
                old_data = memory[old_domain]
                group = dict(old_data, anchor=predictions(previous, old_data, mapping[old_domain]))
                groups.append((group, "anchor", 1.0 / len(merged)))
                proposal[old_domain] = left
            optimize_adapter(candidate, left, groups, args)
            merged_state = dict(state, domain_slots=proposal)
            regression = {d: metrics(rollouts(env, manifest, candidate, merged_state, d, "gate", args))
                          for d in merged}
            attempt["merge_regression"] = regression
            if not all(passes_regression(baseline[d], regression[d], args) for d in merged):
                continue
        with torch.no_grad():
            candidate.adapters[slot].layers[-1].weight.zero_()
            candidate.adapters[slot].layers[-1].bias.zero_()
        replay = []
        for old_domain, data in memory.items():
            anchor = predictions(previous, data, mapping[old_domain])
            replay.extend([(data, "target", 0.2 / len(memory)),
                           (dict(data, anchor=anchor), "anchor", 0.3 / len(memory))])
        optimize_adapter(candidate, slot, [(new_data, "target", 1.0), (new_data, "teacher", 0.5)], args, replay)
        proposal[domain] = slot
        profiles = dict(state["profiles"], **{domain: new_profile})
        threshold = threshold_for(args, candidate, profiles, proposal)
        proposed_state = dict(state, profiles=profiles, domain_slots=proposal, threshold=threshold)
        after = {d: metrics(rollouts(env, manifest, candidate, proposed_state, d, "gate", args))
                 for d in [*mapping, domain]}
        attempt["before"], attempt["after"] = baseline, after
        improved = after[domain]["score"] >= baseline[domain]["score"] + args.min_gain
        if not improved or not all(passes_regression(baseline[d], after[d], args) for d in [*mapping, domain]):
            continue
        new_memory = update_memory(previous, candidate, mapping, proposal, memory, domain, new_data, state["memory_budget"])
        save(args.output, candidate, profiles=profiles, threshold=threshold, domain_slots=proposal,
             block_size=args.block_size, alpha=args.alpha, memory=new_memory, memory_budget=state["memory_budget"],
             stage=state["stage"] + 1)
        decision["accepted"] = True
        report("passed_gate")
        return
    report("failed_gate_or_no_merge")
