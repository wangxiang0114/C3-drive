"""Rebuild the sequential Driving Score table from local Bench2Drive reports."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from .paper_eval import audit_report, digest, route_ids
from .reporting import score_summary, write_json


def resolve(base, path):
    return (base / path).resolve()


def load_manifest(path):
    manifest = json.loads(path.read_bytes())
    domains, seeds, methods = (manifest[field] for field in ("domains", "seeds", "methods"))
    for field, values, count, kind in (
        ("domains", domains, 5, str), ("seeds", seeds, 5, int), ("methods", methods, None, str),
    ):
        if (not isinstance(values, list) or not values or
                any(type(value) is not kind or value == "" for value in values) or
                len(set(values)) != len(values) or (count is not None and len(values) != count)):
            raise ValueError(f"{field} must list {'exactly ' + str(count) if count else 'one or more'} distinct {kind.__name__} values.")
    if set(domains) != {"clean", "camera-drop", "occlusion", "gps", "speed"} or domains[0] != "clean":
        raise ValueError("List clean first, followed by the four manuscript perturbation domains.")
    if type(manifest["repetitions"]) is not int or manifest["repetitions"] < 1:
        raise ValueError("repetitions must be a positive integer.")
    return manifest


def route_order(manifest, route_file):
    official = route_ids(route_file)
    test_routes = manifest["test_routes"]
    if set(test_routes) != set(manifest["domains"]):
        raise ValueError("test_routes must assign a subset to each evaluation domain.")
    expected = {}
    for domain, subset in test_routes.items():
        if not isinstance(subset, list) or not subset or any(type(route) is not str for route in subset):
            raise ValueError(f"{domain}: supply nonempty route IDs as strings.")
        selected = set(subset)
        if len(selected) != len(subset) or not selected.issubset(official):
            raise ValueError(f"{domain}: the route subset includes duplicate or unknown IDs.")
        # The tagged evaluator preserves XML order even when --routes-subset is supplied.
        expected[domain] = [f"RouteScenario_{route}_rep{rep}" for route in official if route in selected
                            for rep in range(manifest["repetitions"])]
    return expected


def expected_cells(manifest):
    domains = manifest["domains"]
    for method in manifest["methods"]:
        for seed in manifest["seeds"]:
            for stage, domain in enumerate(domains):
                for tested in domains[:stage + 1]:
                    yield method, seed, stage, tested, "deployed"
                if stage:
                    yield method, seed, stage, domain, "before_update"


def collect_reports(manifest, directory, route_file, output):
    expected = set(expected_cells(manifest))
    ids = route_order(manifest, route_file)
    cells, sources, hash_by_seed = {}, [], {}
    for item in manifest["reports"]:
        key = tuple(item[field] for field in ("method", "seed", "stage", "domain", "policy"))
        method, seed, stage, domain, policy = key
        if type(seed) is not int or type(stage) is not int or key not in expected:
            raise ValueError(f"Unexpected report cell {key!r}.")
        if key in cells:
            raise ValueError(f"Duplicate report cell {key!r}.")
        path = resolve(directory, item["file"])
        if path == output:
            raise ValueError("Choose an output path different from the report inputs.")
        raw = path.read_bytes()
        sha = digest(raw)
        if hash_by_seed.setdefault(sha, seed) != seed:
            raise ValueError(f"{path}: the same report payload cannot count as two seeds.")
        try:
            scores = audit_report(json.loads(raw), ids[domain])
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise ValueError(f"{path}: {error}") from error
        cells[key] = scores["score_composed"]
        sources.append({"method": method, "seed": seed, "stage": stage, "domain": domain, "policy": policy,
                        "file": item["file"], "sha256": sha, "official_scores": scores})
    missing = sorted(expected - cells.keys())
    if missing:
        raise ValueError(f"Missing {len(missing)} report cells. First missing cell: {missing[0]!r}.")
    return cells, sources


def summarize(manifest, cells):
    domains = manifest["domains"]
    methods = {}
    for method in manifest["methods"]:
        per_seed = {}
        for seed in manifest["seeds"]:
            matrix = [{d: cells[method, seed, stage, d, "deployed"] for d in domains[:stage + 1]}
                      for stage in range(len(domains))]
            gains = {d: matrix[stage][d] - cells[method, seed, stage, d, "before_update"]
                     for stage, d in enumerate(domains) if stage}
            final = score_summary(matrix, domains)
            per_seed[seed] = {
                "stage_domain_ds": matrix,
                "adaptation_gain_by_domain": gains,
                "metrics": {
                    "final_current_domain_ds": final["current_domain_score"],
                    "mean_new_domain_ds": mean(matrix[stage][d] for stage, d in enumerate(domains) if stage),
                    "mean_cross_domain_ds": final["mean_score"],
                    "adaptation_gain": mean(gains.values()),
                    "average_forgetting": final["average_forgetting"],
                    "backward_transfer": final["backward_transfer"],
                },
            }
        metric_names = next(iter(per_seed.values()))["metrics"]
        aggregate = {}
        for name in metric_names:
            values = [row["metrics"][name] for row in per_seed.values()]
            aggregate[name] = {"mean": mean(values), "sample_std_dev": stdev(values)}
        methods[method] = {"seeds": per_seed, "aggregate": aggregate}
    return methods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Local experiment report manifest")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest_path = args.manifest.resolve()
        output = args.output.resolve()
        manifest = load_manifest(manifest_path)
        directory = manifest_path.parent
        routes = resolve(directory, manifest["routes"])
        if output in (manifest_path, routes):
            raise ValueError("Choose an output path different from the manifest and route XML.")
        cells, sources = collect_reports(manifest, directory, routes, output)
        write_json(output, {"protocol": "Bench2Drive v0.0.4 sequential deployment",
                            "domains": manifest["domains"], "repetitions": manifest["repetitions"],
                            "methods": summarize(manifest, cells), "reports": sources})
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
