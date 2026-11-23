"""Run the synthetic manuscript figure workflow using only locally generated data."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory for all outputs")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Choose a new output directory.")
    output.mkdir(parents=True, exist_ok=False)

    base = output / "base"
    records = output / "records"
    quota_figures = output / "quota-figures"
    env = {
        **os.environ,
        "C3DRIVE_BASE_RUNS": str(base),
        "C3DRIVE_EXTENDED_RUNS": str(output / "extended"),
        "C3DRIVE_LONG_RUNS": str(output / "long-horizon"),
        "C3DRIVE_QUOTA_RUNS": str(output / "quota"),
    }
    steps = (
        ("engine.py", "--runs", "30", "--output", base),
        ("analyze.py", "--directory", base),
        ("improve.py",),
        ("long_boundary.py",),
        ("quota_revision.py",),
        ("build_plot_records.py", "--output", records),
        ("plot_quota.py", "--records", records, "--output", quota_figures),
        ("collect_plot_inputs.py", "--replay-arrays", quota_figures / "data" / "replay_plot_arrays.npz",
         "--output", records / "manuscript_plot_inputs.npz"),
        ("plot_manuscript.py", "--records", records, "--output", output / "manuscript-figures"),
        ("check_records.py", "--records", records),
    )
    scripts = Path(__file__).resolve().parent
    for filename, *arguments in steps:
        print(f"Running {filename}", flush=True)
        subprocess.run([sys.executable, str(scripts / filename), *map(str, arguments)], env=env, check=True)


if __name__ == "__main__":
    main()
