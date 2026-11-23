# Experiment workflows

These workflows run the public CARLA 0.9.16 reference implementation on self-collected routes and use the [local route score](../README.md#6-test-on-held-out-routes). See the [manuscript reproducibility guide](reproducibility.md) for the Bench2Drive evaluation setup, official result checking, and synthetic mechanism experiments.

## Sequential closed-loop benchmark

Generate a route manifest, collect clean recordings, train the teacher and student, and calibrate the stage-0 student using the [setup instructions](../README.md#1-installation). Collect training and detector calibration recordings for each domain in the arrival order. Use the same manifest and data directory throughout.

```powershell
foreach ($domain in @("camera-drop", "occlusion", "gps", "speed")) {
    python -m c3drive collect --domain $domain --splits train fit calibrate
}
python -m c3drive benchmark --checkpoint runs/c3-stage0.pt --teacher runs/teacher.pt --domains camera-drop occlusion gps speed --output runs/sequential-default
```

The benchmark evaluates the clean checkpoint on the `test` split first. Each following stage measures the incoming domain before adaptation, runs the existing `adapt` procedure on the `stream` and `gate` splits, and evaluates the resulting deployed policy on every encountered domain. A failed or untriggered update leaves the active checkpoint in place. For that stage's incoming domain, the same held-out measurement serves as the before and after score, so the reported gain is zero. The benchmark also records domains that were encountered but did not pass the deployment gate.

Use `--domains` to compare arrival orders. Start each comparison from the same stage-0 checkpoint and use a separate output directory. The `--limit` option selects the first N routes from each of the `stream`, `gate`, and `test` splits for a shorter run. Use a dedicated CARLA instance for each benchmark. To resume an interrupted run, repeat the command with `--resume`. Completed stages are reused. The command checks checkpoint and manifest hashes, the recording inventory, and settings before resuming.

| Output | Contents |
| --- | --- |
| `config.json` | Domain order, input paths and hashes, recording inventory, and run settings |
| `scores.csv` | Stage-by-domain matrix. Blank cells are future domains that have not yet been evaluated |
| `report.json` | Current-domain score and gain, cross-domain mean, worst-domain score, average forgetting, and backward transfer |
| `stage-*/stage.json` | Paired and per-route test results, active checkpoint hash, parameter counts, adapter occupancy, replay usage, and detector tensor sizes |
| `stage-*/decision.json` | Drift and gate decision for an attempted update, with any merge and candidate regression results |
| `stage-*/accepted.pt` | Checkpoint saved for an accepted update |

Forgetting compares each historical domain's final score with its best score before the final stage. Backward transfer compares it with its first post-update score. Both are reported in units of the repository route score. The resource report separates tensor payload bytes from the size of the serialized checkpoint.

## Offline drift evaluation

Collect test recordings under both clean and shifted conditions, then evaluate the calibrated detector without starting CARLA:

```powershell
python -m c3drive collect --domain clean --splits test
python -m c3drive collect --domain camera-drop --splits test
python -m c3drive detect-eval --checkpoint runs/c3-stage0.pt --domains camera-drop --output runs/detector.json
```

The report compares each selected shift with the clean recordings. It includes episode-level false-alarm and detection rates, AUROC based on peak sequential evidence, and the median first alarm among detected episodes. The trace records evidence and the selected adapter slot for each full block. First-alarm timing is measured from the start of each recording. Use this workflow for recordings collected under a fixed perturbation regime. Controlled shift-onset experiments require separate onset annotations. Episodes with no complete block are listed separately.

Run the software tests without CARLA using `python -m unittest discover -s tests -v`.
