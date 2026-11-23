"""Train, validate and test the released planner on a local dataset."""

from argparse import Namespace

from .data import inspect_groups
from .reporting import write_json


def train_test(args, evaluate):
    from .adapt import initialize
    from .carla_env import connect, read_manifest
    from .train import fit

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new workflow output directory: {output}")
    if args.calibrate and args.slots < 2:
        raise ValueError("--calibrate requires --slots of at least 2.")

    domains = list(dict.fromkeys(args.domains))
    run_offline = args.test_mode in ("offline", "both")
    run_closed_loop = args.test_mode in ("closed-loop", "both")
    manifest = read_manifest(args.manifest) if run_closed_loop else None
    if manifest is not None and args.limit and args.limit > len(manifest["routes"]["test"]):
        raise ValueError("--limit exceeds the number of held-out routes in the manifest.")

    groups = [("clean", split) for split in ("train", "validation")]
    if args.calibrate:
        groups.extend(("clean", split) for split in ("fit", "calibrate", "threshold"))
    if run_offline:
        groups.extend((domain, "test") for domain in domains)
    audit = inspect_groups(args.data, groups, manifest)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "dataset.json", audit)

    teacher = output / "teacher.pt"
    student = output / "student.pt"
    teacher_validation = output / "validation-teacher.json"
    student_validation = output / "validation-student.json"

    def options(**changes):
        return Namespace(**(vars(args) | changes))

    fit(options(output=teacher, epochs=args.teacher_epochs, resume=False))
    evaluate(options(output=teacher_validation, checkpoint=teacher, split="validation", mode="offline",
                     domains=["clean"], limit=None))
    fit(options(output=student, epochs=args.student_epochs, resume=False), teacher)
    evaluate(options(output=student_validation, checkpoint=student, split="validation", mode="offline",
                     domains=["clean"], limit=None))

    selected = student
    if args.calibrate:
        selected = output / "student-calibrated.pt"
        initialize(options(output=selected, checkpoint=student))

    tests = {}
    if run_offline:
        report = output / "test-offline.json"
        evaluate(options(output=report, checkpoint=selected, split="test", mode="offline", domains=domains))
        tests["offline"] = report.name
    if manifest is not None:
        report = output / "test-closed-loop.json"
        env = connect(args, manifest["town"])
        evaluate(options(output=report, checkpoint=selected, split="test", mode="closed-loop", domains=domains),
                 env, manifest)
        tests["closed-loop"] = report.name

    write_json(output / "workflow.json", {
        "data": str(args.data.resolve()),
        "route_manifest": str(args.manifest.resolve()) if manifest is not None else None,
        "dataset_audit": "dataset.json",
        "training": {
            "seed": args.seed, "device": args.device, "learning_rate": args.lr, "batch_size": args.batch_size,
            "teacher_epochs": args.teacher_epochs, "student_epochs": args.student_epochs, "adapter_slots": args.slots,
        },
        "checkpoints": {"teacher": teacher.name, "student": student.name, "tested": selected.name},
        "validation": {"teacher": teacher_validation.name, "student": student_validation.name},
        "test_domains": domains,
        "tests": tests,
    })
    print(output / "workflow.json", flush=True)
