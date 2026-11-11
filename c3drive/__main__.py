import argparse
import json
import math
from pathlib import Path

from .data import DOMAINS, SPLITS


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def probability(value):
    number = float(value)
    if not 0 < number < 1:
        raise argparse.ArgumentTypeError("Expected a value between zero and one.")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive finite number.")
    return number


def simulation(parser, manifest=True):
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=positive, default=2000)
    parser.add_argument("--carla-root", help="CARLA installation containing PythonAPI/carla")
    if manifest:
        parser.add_argument("--manifest", type=Path, default=Path("data/routes.json"))
        parser.add_argument("--ticks", type=positive, default=900)
        parser.add_argument("--traffic", type=int, default=20)
        parser.add_argument("--limit", type=positive, help="Use this many routes per selected split")


def learning(parser):
    parser.add_argument("--data", type=Path, default=Path("data/episodes"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a CUDA device such as cuda:0")
    parser.add_argument("--batch-size", type=positive, default=16)
    parser.add_argument("--output", type=Path, required=True)


def evaluation(parser, split):
    simulation(parser)
    learning(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("offline", "closed-loop"), default="offline" if split == "validation" else "closed-loop")
    parser.add_argument("--domains", nargs="+", choices=DOMAINS)
    parser.set_defaults(split=split)


def command_parser():
    parser = argparse.ArgumentParser(description="Train and evaluate C3-Drive from scratch.")
    sub = parser.add_subparsers(dest="command", required=True)

    routes = sub.add_parser("routes", help="Generate disjoint CARLA route splits")
    simulation(routes, manifest=False)
    routes.add_argument("--town", default="Town10HD_Opt")
    routes.add_argument("--routes-per-split", type=positive, default=24)
    routes.add_argument("--seed", type=int, default=41)
    routes.add_argument("--output", type=Path, default=Path("data/routes.json"))

    collect = sub.add_parser("collect", help="Record expert sensor histories and trajectory labels")
    simulation(collect)
    collect.add_argument("--domain", choices=DOMAINS, default="clean")
    collect.add_argument("--splits", nargs="+", choices=(*SPLITS[:5], "test"), required=True)
    collect.add_argument("--data", type=Path, default=Path("data/episodes"))

    audit = sub.add_parser("inspect-data", help="Validate episode arrays and check route split overlap")
    audit.add_argument("--data", type=Path, default=Path("data/episodes"))
    audit.add_argument("--domains", nargs="+", choices=DOMAINS, default=["clean"])
    audit.add_argument("--splits", nargs="+", choices=SPLITS, default=["train", "validation"])
    audit.add_argument("--output", type=Path, required=True)

    for command, count in (("train-teacher", 30), ("train-student", 40)):
        train = sub.add_parser(command)
        learning(train)
        train.add_argument("--epochs", type=positive, default=count)
        train.add_argument("--lr", type=positive_float, default=3e-4)
        train.add_argument("--seed", type=int, default=41)
        train.add_argument("--resume", action="store_true", help="Resume from the matching .latest checkpoint")
        if command == "train-student":
            train.add_argument("--slots", type=positive, default=2)
            train.add_argument("--teacher", type=Path, required=True)

    init = sub.add_parser("calibrate", help="Fit clean-domain statistics and an episode-level drift threshold")
    learning(init)
    init.add_argument("--checkpoint", type=Path, required=True)
    init.add_argument("--alpha", type=probability, default=0.05)
    init.add_argument("--block-size", type=positive, default=16)
    init.add_argument("--memory-budget", type=positive, default=256)

    update = sub.add_parser("adapt", help="Detect a shift, train a candidate and apply the closed-loop gate")
    simulation(update)
    learning(update)
    update.add_argument("--checkpoint", type=Path, required=True)
    update.add_argument("--teacher", type=Path, required=True)
    update.add_argument("--domain", choices=DOMAINS[1:-1], required=True)
    update.add_argument("--report", type=Path, required=True)
    update.add_argument("--steps", type=positive, default=300)
    update.add_argument("--candidates", type=positive, default=1024)
    update.add_argument("--lr", type=positive_float, default=1e-3)
    update.add_argument("--seed", type=int, default=41)
    update.add_argument("--min-gain", type=float, default=0.02)
    update.add_argument("--max-score-drop", type=float, default=0.03)
    update.add_argument("--max-completion-drop", type=float, default=0.03)
    update.add_argument("--max-collision-increase", type=float, default=0.05)

    evaluation(sub.add_parser("validate", help="Evaluate a checkpoint on validation data or routes"), "validation")
    evaluation(sub.add_parser("test", help="Evaluate a frozen checkpoint on the held-out test split"), "test")
    return parser


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def collect(env, manifest, args):
    for split in args.splits:
        written = 0
        for index, route in enumerate(manifest["routes"][split][:args.limit or None]):
            path = args.data / args.domain / split / f"{index:04d}.npz"
            if path.exists():
                import numpy as np

                with np.load(path) as archive:
                    metadata = json.loads(str(archive["metadata"]))
                if any(metadata[key] != value for key, value in {
                    "route": route, "domain": args.domain, "town": manifest["town"],
                    "requested_ticks": args.ticks, "requested_traffic": args.traffic,
                }.items()):
                    raise ValueError(f"{path} uses a different recording configuration. Choose another --data directory.")
                print(f"Using {path}", flush=True)
                written += 1
                continue
            result = env.run(route, args.domain, args.ticks, args.traffic, record_path=path)
            saved = path.is_file()
            written += saved
            print(f"{args.domain}/{split}/{index:04d} ticks={result['ticks']} saved={saved}", flush=True)
        if not written:
            raise RuntimeError(f"No episodes with a complete target horizon in {args.domain}/{split}.")


def run_evaluation(args, env=None, manifest=None):
    from .adapt import metrics, policy_for, rollouts
    from .data import episodes
    from .evaluate import open_loop
    from .train import device_for, load, loader

    if args.output.resolve() == args.checkpoint.resolve():
        raise ValueError("Choose a report path different from the checkpoint.")
    model, state = load(args.checkpoint, device_for(args.device))
    domains = args.domains or (list(DOMAINS) if args.split == "test" and env is not None else ["clean"])
    report = {"mode": args.mode, "split": args.split, "checkpoint": str(args.checkpoint),
              "checkpoint_epoch": state.get("epoch"), "checkpoint_stage": state.get("stage"),
              "results": {}}
    if env is not None:
        report.update(town=manifest["town"], manifest=str(args.manifest), ticks=args.ticks, traffic=args.traffic)
    for domain in dict.fromkeys(domains):
        if env is None:
            paths = episodes(args.data, domain, args.split)[:args.limit or None]
            factory = lambda: policy_for(model, state).detector
            results = open_loop(model, (loader([path], args.batch_size) for path in paths), factory)
            report["results"][domain] = {"metrics": results, "episodes": [str(path) for path in paths]}
        else:
            rows = rollouts(env, manifest, model, state, domain, args.split, args)
            results = metrics(rows)
            report["results"][domain] = {"metrics": results, "routes": rows}
        print(domain, results, flush=True)
    write_json(args.output, report)


def main():
    parser = command_parser()
    args = parser.parse_args()
    if getattr(args, "traffic", 0) < 0:
        parser.error("--traffic must be nonnegative.")
    if args.command == "inspect-data":
        from .data import inspect_dataset

        report = inspect_dataset(args.data, args.domains, args.splits)
        write_json(args.output, report)
        print(args.output)
        return
    if args.command in ("train-teacher", "train-student"):
        from .train import fit

        fit(args, getattr(args, "teacher", None))
        return
    if args.command == "calibrate":
        from .adapt import initialize

        initialize(args)
        return
    if args.command in ("validate", "test") and args.mode == "offline":
        run_evaluation(args)
        return
    from .carla_env import connect, read_manifest

    if args.command == "routes":
        if args.output.exists():
            parser.error("Choose a new route manifest output path.")
        env = connect(args, args.town)
        write_json(args.output, env.make_routes(args.routes_per_split, args.seed))
        print(args.output)
        return
    manifest = read_manifest(args.manifest)
    if args.limit and any(args.limit > len(routes) for routes in manifest["routes"].values()):
        parser.error("--limit exceeds the number of routes in the manifest.")
    env = connect(args, manifest["town"])
    if args.command == "collect":
        collect(env, manifest, args)
        return
    if args.command == "adapt":
        from .adapt import continual

        continual(args, env, manifest)
        return
    run_evaluation(args, env, manifest)


if __name__ == "__main__":
    main()
