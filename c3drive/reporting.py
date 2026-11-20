"""Experiment summaries and atomic report writes."""

import csv
import json
from contextlib import contextmanager
from pathlib import Path
from statistics import mean
from tempfile import NamedTemporaryFile


@contextmanager
def atomic_output(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            yield handle
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_output(path) as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def score_summary(matrix, domains):
    """Summarize a prefix of the lower triangular score matrix.

    A domain's diagonal entry is its first post-update evaluation. Forgetting
    compares its final score with the best score before the final stage.
    """
    final_stage = len(matrix) - 1
    final = matrix[-1]
    historical = domains[:final_stage]
    forgetting = {
        domain: max(row[domain] for row in matrix[j:final_stage]) - final[domain]
        for j, domain in enumerate(historical)
    }
    backward_transfer = {
        domain: final[domain] - matrix[j][domain] for j, domain in enumerate(historical)
    }
    return {
        "current_domain_score": final[domains[final_stage]],
        "mean_score": mean(final.values()),
        "worst_domain_score": min(final.values()),
        "forgetting_by_domain": forgetting,
        "average_forgetting": mean(forgetting.values()) if forgetting else None,
        "backward_transfer_by_domain": backward_transfer,
        "backward_transfer": mean(backward_transfer.values()) if backward_transfer else None,
    }


def write_score_matrix(path, domains, matrix):
    with atomic_output(path) as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "arrival_domain", *domains])
        for index, scores in enumerate(matrix):
            writer.writerow([index, domains[index], *(scores.get(domain) for domain in domains)])
