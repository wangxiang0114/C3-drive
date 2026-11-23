# Manuscript reproducibility

The EAAI manuscript describes a CARLA 0.9.15 experiment using Bench2Drive v0.0.4. This repository publishes a trainable CARLA reference implementation and tools for checking official Bench2Drive evaluator output. Use the workflow that matches the result you want to verify.

| Item | EAAI manuscript | Public artifacts |
| --- | --- | --- |
| Simulator | CARLA 0.9.15 | The self-collected reference workflow runs on 0.9.16. The official evaluation setup below uses 0.9.15. |
| Training and route splits | Bench2Drive v0.0.4 has 1,100 training routes and 220 validation routes. The manuscript uses base-route partitioning for calibration and separate gates. | The upstream dataset and all 220 published validation routes are available. The experiment's base-route assignment manifest is unavailable. The reference implementation collects Town10HD_Opt routes with BasicAgent, 24 per split by default. |
| Continual adaptation | Four adapter slots, 400 replay segments, top-two routing, batch size 32, and 5,000 steps per stage | The reference workflow retains defaults of two slots and 256 replay examples. |
| Student model and training | Frozen student backbone and planner during adaptation, router and adapter updates, two RTX 4090 GPUs with mixed precision and distributed training | The reference workflow includes a CNN–GRU student. The benchmark student, its original training code, and its checkpoints are unavailable. The manuscript mentions TransFuser in related work. |
| Driving Score | Bench2Drive/Leaderboard Driving Score with five paired seeds | The checker validates complete official JSON reports. The reference implementation computes a local proxy score. No original leaderboard files for the manuscript's DS tables are included. |

## Official route and result verification

Set up [Bench2Drive at tag 0.0.4](https://github.com/Thinklab-SJTU/Bench2Drive/tree/0.0.4) with CARLA 0.9.15 using the upstream instructions. Preserve the tagged route file bytes. The command below validates the route file by SHA-256 and verifies all 220 route IDs. It uses only Python's standard library and can run separately from the evaluator.

```bash
python -m c3drive.paper_eval --routes "/path/to/local/bench2drive_0.0.4_val.xml" --output runs/route-audit.json
```

Use [the official v0.0.4 dataset](https://huggingface.co/datasets/rethinklab/Bench2Drive-V0.0.4) for the 1,100 training routes and the [upstream evaluator](https://github.com/Thinklab-SJTU/Bench2Drive/blob/0.0.4/leaderboard/leaderboard/leaderboard_evaluator.py) for CARLA results. For an actual run, provide the agent implementation and its checkpoint through the evaluator's agent arguments. Archive the evaluator revision, simulator build, perturbation code and seeds, route subset and split files, the checkpoint hash, and the unmodified output JSON together.

After evaluation on the complete validation XML, pass each unmodified JSON report for the **same method, deployment stage, and evaluation domain**. Repeat the result argument for the five seeds and set the number of route repetitions to the evaluator's setting.

```bash
python -m c3drive.paper_eval --routes runs/bench2drive_0.0.4_val.xml \
  --result runs/seed1.json --result runs/seed2.json --result runs/seed3.json \
  --result runs/seed4.json --result runs/seed5.json --output runs/ds-audit.json
```

The checker verifies full route coverage, status, result order, report duplication by file hash, and agreement between per-route scores and global means. Multiple reports produce an across-report mean and sample standard deviation. It neither infers the original held-out partition nor verifies that an agent, perturbation implementation, or seed produced a supplied JSON. An audit of the complete upstream validation set cannot substitute for the manuscript's split-specific experiments. See the [Leaderboard score definition](https://leaderboard.carla.org/evaluation_v2_0/).

### Sequential Driving Score table

Use the local manifest to rebuild the Table 4 metrics from finished evaluator reports on the actual held-out route subsets. Paths are relative to the manifest or absolute. The manifest has these fields.

| Field | Value |
| --- | --- |
| `routes` | Path to the unchanged tagged validation XML |
| `domains` | `["clean", "camera-drop", "occlusion", "gps", "speed"]` for the default order |
| `seeds` | Five distinct integer seeds from the experiment |
| `methods` | Distinct method labels used in the reports |
| `repetitions` | Official evaluator repetitions per route |
| `test_routes` | Object mapping every domain to its original held-out route IDs as strings |
| `reports` | Array of objects with `method`, `seed`, `stage`, `domain`, `policy`, and `file` |

For each method and seed, use stage 0 for the clean baseline and stages 1–4 for the four updates. Supply one `deployed` report per domain already seen at each stage. Supply a `before_update` report on the incoming domain at each update, evaluated on that same domain's held-out routes. This gives 15 deployed and four before-update reports per method and seed. For example, the first pre-update report has `stage` 1, `domain` `camera-drop`, and `policy` `before_update`.

```bash
python -m c3drive.paper_table --manifest "/path/to/local/report-manifest.json" --output runs/table4.json
```

The output includes the per-seed stage-by-domain matrices, gains, and the five-seed mean and sample standard deviation for each metric. Current-domain DS is reported both as the final incoming-domain score and as the average of the four post-update incoming-domain scores. Adaptation gain averages the four paired before/after deltas. Final cross-domain DS, forgetting, and backward transfer use the manuscript's stage-matrix definitions. The manifest must come from the experiment owner to establish the true split, method, and seed provenance.

## Plotting

The plotting command reads local input files and writes PNG, PDF, and SVG. Install the plotting dependencies in a Python 3.12 environment.

```bash
python -m pip install -r experiments/mechanism_sim/requirements.txt
python experiments/mechanism_sim/plot_manuscript.py --records "/path/to/local/plot-inputs" --output runs/figures
```

The input directory must contain `manuscript_plot_inputs.npz`, `all_pair_audit.json`, `canonical_gate_clause_audit.csv`, and `critical_event_reference.npz`. Keep this directory outside the repository.

## Manuscript evidence still needed

- The simulator- and dataset-specific package needs the actual teacher and student architectures, training and preprocessing code, exact per-stage configurations, checkpoints, and original route assignments. It also needs the perturbation and gate implementations, paired baseline runs, route-level JSONs, and the table-generation inputs.
