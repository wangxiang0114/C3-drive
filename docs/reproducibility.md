# Manuscript reproducibility

The EAAI manuscript describes a CARLA 0.9.15 experiment using Bench2Drive v0.0.4. This repository publishes a trainable CARLA reference implementation, synthetic mechanism plotting tools, and a checker for official Bench2Drive evaluator output. Use the workflow that matches the result you want to verify.

| Item | EAAI manuscript | Public artifacts |
| --- | --- | --- |
| Simulator | CARLA 0.9.15 | The self-collected reference workflow runs on 0.9.16. The official evaluation setup below uses 0.9.15. |
| Training and route splits | Bench2Drive v0.0.4 has 1,100 training routes and 220 validation routes. The manuscript uses base-route partitioning for calibration and separate gates. | The upstream dataset and all 220 published validation routes are available. The experiment's base-route assignment manifest is unavailable. The reference implementation collects Town10HD_Opt routes with BasicAgent, 24 per split by default. |
| Continual adaptation | Four adapter slots, 400 replay segments, top-two routing, batch size 32, and 5,000 steps per stage | The reference workflow retains defaults of two slots and 256 replay examples. The synthetic experiment separately uses four slots and 400 samples. |
| Student model and training | Frozen student backbone and planner during adaptation, router and adapter updates, two RTX 4090 GPUs with mixed precision and distributed training | The reference workflow includes a CNN–GRU student. The benchmark student, its original training code, and its checkpoints are unavailable. The manuscript mentions TransFuser in related work. |
| Driving Score | Bench2Drive/Leaderboard Driving Score with five paired seeds | The checker validates complete official JSON reports. The reference implementation computes a local proxy score. No original leaderboard files for the manuscript's DS tables are included. |

## Official route and result verification

Set up [Bench2Drive at tag 0.0.4](https://github.com/Thinklab-SJTU/Bench2Drive/tree/0.0.4) with CARLA 0.9.15 using the upstream instructions. Preserve the tagged route file bytes. The command below validates the route file by SHA-256 and verifies all 220 route IDs. It uses only Python's standard library and can run separately from the evaluator.

```bash
mkdir -p runs
curl --fail --location https://raw.githubusercontent.com/Thinklab-SJTU/Bench2Drive/0.0.4/leaderboard/data/bench2drive_0.0.4_val.xml -o runs/bench2drive_0.0.4_val.xml
python -m c3drive.paper_eval --routes runs/bench2drive_0.0.4_val.xml --output runs/route-audit.json
```

Use [the official v0.0.4 dataset](https://huggingface.co/datasets/rethinklab/Bench2Drive-V0.0.4) for the 1,100 training routes and the [upstream evaluator](https://github.com/Thinklab-SJTU/Bench2Drive/blob/0.0.4/leaderboard/leaderboard/leaderboard_evaluator.py) for CARLA results. For an actual run, provide the agent implementation and its checkpoint through the evaluator's agent arguments. Archive the evaluator revision, simulator build, perturbation code and seeds, route subset and split files, the checkpoint hash, and the unmodified output JSON together.

After evaluation on the complete validation XML, pass each unmodified JSON report for the **same method, deployment stage, and evaluation domain**. Repeat the result argument for the five seeds and set the number of route repetitions to the evaluator's setting.

```bash
python -m c3drive.paper_eval --routes runs/bench2drive_0.0.4_val.xml \
  --result runs/seed1.json --result runs/seed2.json --result runs/seed3.json \
  --result runs/seed4.json --result runs/seed5.json --output runs/ds-audit.json
```

The checker verifies full route coverage, status, result order, report duplication by file hash, and agreement between per-route scores and global means. Multiple reports produce an across-report mean and sample standard deviation. It neither infers the original held-out partition nor verifies that an agent, perturbation implementation, or seed produced a supplied JSON. An audit of the complete upstream validation set cannot substitute for the manuscript's split-specific experiments. See the [Leaderboard score definition](https://leaderboard.carla.org/evaluation_v2_0/).

## Synthetic mechanism experiments

The archived simulation runs on CPU without CARLA. It models lateral and speed control with seeded observation and dynamics shifts, a bounded adapter pool, replay selection, and closed-loop gates. Its Tracking Score has a different definition and scale from Driving Score. Use Python 3.12 and install the pinned numerical dependencies in a separate virtual environment.

```bash
python -m pip install -r experiments/mechanism_sim/requirements.txt
python experiments/mechanism_sim/plot_manuscript.py --records "/path/to/local/plot-data" --output runs/manuscript-figures
```

The script exports Figs. 7–10 to PNG, PDF, and SVG. Point the command at a local directory containing these files from the synthetic analysis.

| File | Contents |
| --- | --- |
| `manuscript_plot_inputs.npz` | Run-level score, resource, and detector arrays in the format written by `collect_plot_inputs.py` |
| `all_pair_audit.json` | Paired historical losses in `functional` and `uniform_expectation` arrays |
| `canonical_gate_clause_audit.csv` | Per-proposal `run`, `passed`, `audit_history_drop`, and `audit_current_gain` columns |
| `critical_event_reference.npz` | Reference replay `rows` (domain, severity, risk, critical-event flag, Pareto rank) and Boolean `selected` mask |

Keep these inputs outside the repository. Use the tested plotting dependencies and Times New Roman when comparing PNGs with the original artwork. The renderer falls back to DejaVu Serif when Times New Roman is unavailable.

To rerun the synthetic simulator separately and analyze the new logs:

```bash
python experiments/mechanism_sim/engine.py --runs 30 --output runs/mechanism-sim
python experiments/mechanism_sim/analyze.py --directory runs/mechanism-sim
```

Use `--runs 1` and a new output directory for a quick run. The engine writes stage-by-domain scores, detector traces, update decisions, and checkpoint artifacts. The analyzer renders the archived version of the mechanism figures. These are earlier versions of the figure panels in the manuscript. The archive's detector uses 16 frames per block, while the manuscript's CARLA setup specifies 20 ticks per block.

Run `check_records.py --records /path/to/local/plot-data` to check the local `all_pair_validation_metrics.csv` and `canonical_gate_clause_audit.csv` files. It recomputes the comparison on 72 pairs from 12 checkpoints in Fig. 8(b) and the counterfactual on 110 proposals from 12 seeds in Fig. 8(f) and Table 10. The gate counterfactual evaluates the same proposals under forced commit or the recorded acceptance decisions, keeping the previous policy's score on rejected proposals.

After completing the 30 runs and the base analyzer, generate the synthetic variants, perturbation-condition experiments, capacity sweep, and replay quota experiments. Each of the following experiment scripts writes its own run directory under `runs/` and reports progress. Run them in the listed order.

```bash
python experiments/mechanism_sim/improve.py
python experiments/mechanism_sim/long_boundary.py
python experiments/mechanism_sim/quota_revision.py
python experiments/mechanism_sim/plot_quota.py --records "/path/to/local/plot-data" --output runs/mechanism-figures
python experiments/mechanism_sim/collect_plot_inputs.py --replay-arrays runs/mechanism-figures/data/replay_plot_arrays.npz --output runs/rerun-figure-inputs.npz
python experiments/mechanism_sim/plot_manuscript.py --records "/path/to/local/plot-data" --inputs runs/rerun-figure-inputs.npz --output runs/rerun-manuscript-figures
```

The extended scripts run 132 order and shift-condition jobs, 84 long-horizon jobs, and 48 quota jobs. The plotting script combines their outputs with the local audit inputs supplied via the records option and generates the mechanism panels corresponding to Figs. 7–10. The plotting commands first build replay arrays, then gather inputs and render the manuscript layout. This route reuses the supplied pair and gate audits and reference replay records. The quota plot also reads the local protocol diagnostics and both same-pool retention CSVs. A rerun on another platform or numerical stack may change individual gate decisions and derived figures. Use the original local plotting inputs when comparing with the archived manuscript figures. Runtime depends on CPU and available storage. Set `C3DRIVE_BASE_RUNS`, `C3DRIVE_EXTENDED_RUNS`, `C3DRIVE_LONG_RUNS`, or `C3DRIVE_QUOTA_RUNS` to reuse the corresponding archived run directory.

The archived engine and plotting script preserve the simulation logic and adapt the paths for distribution. The original engine SHA-256 was `36cbc2def6f6bb1fbe67ee10c4b18c580389b21010dcc7a6e84063d7d7dfcfeb`. The output configuration records the distributed engine SHA-256.

## Manuscript evidence still needed

- The simulator- and dataset-specific package needs the actual teacher and student architectures, training and preprocessing code, exact per-stage configurations, checkpoints, and original route assignments. It also needs the perturbation and gate implementations, paired baseline runs, route-level JSONs, and the table-generation inputs. The released repository cannot substantiate the reported 80.4 DS, 77.8 DS, 3.4-point average forgetting, or 82.5% consolidation result.
