"""Check route coverage and scores in Bench2Drive v0.0.4 evaluator reports."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from xml.etree import ElementTree

from .reporting import write_json

ROUTE_SHA256 = "885debbc3d87f14351b521d8da0dc9d25fcee1f0d926ed1c92cc9ff62504ca45"
ROUTE_COUNT = 220
SCORE_RANGES = {"score_composed": 100, "score_route": 100, "score_penalty": 1}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def route_ids(path):
    payload = path.read_bytes()
    if digest(payload) != ROUTE_SHA256:
        raise ValueError(f"{path} does not match the tagged Bench2Drive v0.0.4 validation XML.")
    routes = [route.attrib["id"] for route in ElementTree.fromstring(payload).findall("route")]
    if len(routes) != ROUTE_COUNT or len(set(routes)) != ROUTE_COUNT:
        raise ValueError(f"{path} must contain {ROUTE_COUNT} unique routes.")
    return routes


def read_scores(values):
    scores = {}
    for name, upper in SCORE_RANGES.items():
        value = values[name]
        if type(value) not in (int, float) or not 0 <= value <= upper:
            raise ValueError(f"{name} must be between 0 and {upper}.")
        scores[name] = value
    return scores


def audit_report(report, expected_ids):
    count = len(expected_ids)
    if report.get("entry_status") != "Finished" or report.get("eligible") is not True:
        raise ValueError("Expected a finished, eligible evaluator report.")
    checkpoint = report["_checkpoint"]
    if checkpoint["progress"] != [count, count] or len(checkpoint["records"]) != count:
        raise ValueError(f"Expected {count} completed route records.")
    rows = []
    for index, (record, route_id) in enumerate(zip(checkpoint["records"], expected_ids)):
        if record["route_id"] != route_id or record["index"] != index:
            raise ValueError(f"Expected route {route_id} at evaluator index {index}.")
        scores = read_scores(record["scores"])
        composed = scores["score_route"] * scores["score_penalty"]
        # Both factors and the product are rounded independently to six decimals.
        if not math.isclose(scores["score_composed"], composed, rel_tol=0, abs_tol=0.000052):
            raise ValueError(f"Driving Score does not match route completion and infraction penalty for {route_id}.")
        rows.append(scores)
    official = read_scores(checkpoint["global_record"]["scores_mean"])
    for name, score in official.items():
        if not math.isclose(score, mean(row[name] for row in rows), rel_tol=0, abs_tol=0.00000051):
            raise ValueError(f"The evaluator's global {name} disagrees with the route records.")
    return official


def audit_files(routes, reports, repetitions):
    ids = route_ids(routes)
    expected_ids = [f"RouteScenario_{route}_rep{rep}" for route in ids for rep in range(repetitions)]
    runs = []
    fingerprints = set()
    for report in reports:
        payload = report.read_bytes()
        sha256 = digest(payload)
        if sha256 in fingerprints:
            raise ValueError(f"{report} duplicates a previously supplied evaluator report.")
        fingerprints.add(sha256)
        try:
            scores = audit_report(json.loads(payload), expected_ids)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise ValueError(f"{report}: {error}") from error
        runs.append({"file": str(report), "sha256": sha256, "scores": scores})

    result = {
        "benchmark": "Bench2Drive v0.0.4 validation",
        "routes_sha256": ROUTE_SHA256,
        "unique_routes": len(ids),
        "repetitions_per_route": repetitions,
        "reports": runs,
    }
    if len(runs) > 1:
        result["across_reports"] = {}
        for name in SCORE_RANGES:
            values = [run["scores"][name] for run in runs]
            result["across_reports"][name] = {"mean": mean(values), "sample_std_dev": stdev(values)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, required=True, help="Tagged validation XML")
    parser.add_argument("--result", type=Path, action="append", default=[],
                        help="Completed evaluator JSON. Repeat for independent runs of one setting.")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1.")
    if args.output.resolve() in {path.resolve() for path in (args.routes, *args.result)}:
        parser.error("--output must differ from the input files.")
    try:
        result = audit_files(args.routes, args.result, args.repetitions)
        write_json(args.output, result)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Verified {result['unique_routes']} tagged routes and {len(result['reports'])} evaluator reports.")


if __name__ == "__main__":
    main()
