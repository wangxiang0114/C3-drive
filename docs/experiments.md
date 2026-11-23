# Experiment workflows

Train from local NPZ episodes and evaluate on held-out recordings or CARLA routes. [Bench2Drive report tools and plotting commands](reproducibility.md)

## Train, validate, and test

Install the project's Python dependencies. For offline testing, place local episodes in `<data>/clean/train/`, `<data>/clean/validation/`, and `<data>/clean/test/`. Each directory contains NPZ episodes in the [repository format](dataset.md). If you need to collect episodes using an existing CARLA installation, follow the [collection instructions](../README.md#2-generate-routes-and-collect-data) and include `test` for offline evaluation.

```powershell
python -m c3drive train-test --data "D:\C3Drive\episodes" --device cuda --output runs/offline-training
```

The workflow checks file contents and route separation before training. It trains the teacher on the clean training split, distills the student from the selected teacher, and selects both checkpoints by validation planning loss. The held-out test split is used for the final student evaluation. Offline reports include planning loss, average and final displacement errors in metres, and speed mean absolute error in metres per second. Use `--device cpu` for CPU training and adjust `--batch-size` to available memory.

For closed-loop testing, start CARLA and provide the route manifest used to produce your training and validation episodes. The workflow checks those episodes against the manifest before training. Use `--test-mode both` to also evaluate the recorded test episodes. Each offline test domain needs its own `<data>/<domain>/test/` directory.

```powershell
python -m c3drive train-test --data "D:\C3Drive\episodes" --manifest "D:\C3Drive\routes.json" --test-mode closed-loop --domains clean camera-drop --output runs/closed-loop-training
```

Use `--calibrate` when you want a checkpoint ready for continual adaptation. It requires the separate `clean/fit`, `clean/calibrate`, and `clean/threshold` splits. The workflow tests the calibrated checkpoint when this flag is supplied. Use the [sequential benchmark](#sequential-closed-loop-benchmark) to train and validate adaptation candidates under deployment shifts.

The output directory contains `teacher.pt`, `student.pt`, both validation reports, and the selected test reports. `workflow.json` ties these files to the tested checkpoint. To reevaluate an existing checkpoint without retraining, use the individual commands.

```powershell
python -m c3drive validate --data "D:\C3Drive\episodes" --checkpoint runs/offline-training/student.pt --output runs/validation-repeat.json
python -m c3drive test --mode offline --domains clean --data "D:\C3Drive\episodes" --checkpoint runs/offline-training/student.pt --output runs/test-repeat.json
```

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
