# Bench2Drive report tools and plotting

## Official route and result verification

Use the validation XML from Bench2Drive v0.0.4. The command checks the file hash and route IDs. [Bench2Drive v0.0.4 setup](https://github.com/Thinklab-SJTU/Bench2Drive/tree/0.0.4)

```bash
python -m c3drive.paper_eval --routes "/path/to/local/bench2drive_0.0.4_val.xml" --output runs/route-audit.json
```

Pass evaluator JSONs for the same method, deployment stage, and domain. Set `--repetitions` to match the evaluator run.

```bash
python -m c3drive.paper_eval --routes runs/bench2drive_0.0.4_val.xml \
  --result runs/seed1.json --result runs/seed2.json --result runs/seed3.json \
  --result runs/seed4.json --result runs/seed5.json --output runs/ds-audit.json
```

The command checks route coverage and score consistency, then reports the mean and sample standard deviation across the supplied reports. [Leaderboard scoring](https://leaderboard.carla.org/evaluation_v2_0/)

### Sequential Driving Score table

Create a JSON manifest for the evaluator reports. Use absolute paths or paths relative to the manifest.

| Field | Value |
| --- | --- |
| `routes` | Path to the unchanged tagged validation XML |
| `domains` | `["clean", "camera-drop", "occlusion", "gps", "speed"]` for the default order |
| `seeds` | Five distinct integer seeds from the experiment |
| `methods` | Distinct method labels used in the reports |
| `repetitions` | Official evaluator repetitions per route |
| `test_routes` | Held-out route IDs as strings, grouped by domain |
| `reports` | Array of objects with `method`, `seed`, `stage`, `domain`, `policy`, and `file` |

For each method and seed, use stage 0 for the clean baseline and stages 1–4 for the four updates. Supply one `deployed` report per domain already seen at each stage. Supply a `before_update` report on the incoming domain at each update, evaluated on that same domain's held-out routes. This gives 15 deployed and four before-update reports per method and seed. For example, the first pre-update report has `stage` 1, `domain` `camera-drop`, and `policy` `before_update`.

```bash
python -m c3drive.paper_table --manifest "/path/to/local/report-manifest.json" --output runs/table4.json
```

The output contains each seed's stage-by-domain scores and paired adaptation gains, plus means and sample standard deviations across seeds.

## Plotting

The plotting command reads local input files and writes PNG, PDF, and SVG. Install the plotting dependencies in a Python 3.12 environment.

```bash
python -m pip install -r experiments/mechanism_sim/requirements.txt
python experiments/mechanism_sim/plot_manuscript.py --records "/path/to/local/plot-inputs" --output runs/figures
```

The input directory must contain `manuscript_plot_inputs.npz`, `all_pair_audit.json`, `canonical_gate_clause_audit.csv`, and `critical_event_reference.npz`. Keep this directory outside the repository.
