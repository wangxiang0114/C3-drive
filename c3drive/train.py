import random
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .data import EpisodeData, check_split_overlap, episodes, move, observations
from .evaluate import open_loop
from .model import Driver, behavior_loss


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_for(name):
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this PyTorch installation. Install a CUDA-enabled build or use --device cpu.")
    return device


class EpisodeBatches(Sampler):
    def __init__(self, ends, batch_size, shuffle):
        self.ranges = list(zip([0, *ends[:-1]], ends))
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        order = self.ranges.copy()
        if self.shuffle:
            random.shuffle(order)
        for start, end in order:
            indices = list(range(start, end))
            if self.shuffle:
                random.shuffle(indices)
            for offset in range(0, len(indices), self.batch_size):
                yield indices[offset:offset + self.batch_size]

    def __len__(self):
        return sum((end - start + self.batch_size - 1) // self.batch_size for start, end in self.ranges)


def loader(paths, size, shuffle=False):
    data = EpisodeData(paths)
    return DataLoader(data, batch_sampler=EpisodeBatches(data.ends, size, shuffle))


def save(path, model, **fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        tmp = Path(handle.name)
    try:
        torch.save(dict(spec=model.spec, weights=model.state_dict(), **fields), tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def load(path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = Driver(**state["spec"]).to(device)
    model.load_state_dict(state["weights"])
    return model.eval(), state


def fit(args, teacher_path=None):
    seed_all(args.seed)
    device = device_for(args.device)
    is_student = teacher_path is not None
    role = "student" if is_student else "teacher"
    latest = args.output.with_name(f"{args.output.stem}.latest{args.output.suffix}")
    if teacher_path and teacher_path.resolve() in (args.output.resolve(), latest.resolve()):
        raise ValueError("Teacher and student checkpoint paths must be different.")
    if not args.resume and (args.output.exists() or latest.exists()):
        raise FileExistsError("The run already has checkpoints. Use --resume or choose a new --output.")
    train_paths = episodes(args.data, "clean", "train")
    validation_paths = episodes(args.data, "clean", "validation")
    check_split_overlap([("clean", "train", train_paths), ("clean", "validation", validation_paths)])
    source_paths = [*train_paths, *validation_paths, *([teacher_path] if is_student else [])]
    sources = [(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in source_paths]
    if is_student:
        model = Driver(slots=args.slots).to(device)
        teacher, _ = load(teacher_path, device)
        if teacher.spec["slots"] != 0:
            raise ValueError("--teacher must point to a teacher checkpoint.")
        bridge = torch.nn.Linear(model.spec["dim"], teacher.spec["dim"]).to(device)
        parameters = list(model.parameters()) + list(bridge.parameters())
    else:
        model = Driver(width=32, dim=128, slots=0).to(device)
        bridge = None
        parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    train = loader(train_paths, args.batch_size, shuffle=True)
    validation = loader(validation_paths, args.batch_size)
    settings = dict(role=role, batch_size=args.batch_size, lr=args.lr, seed=args.seed, sources=sources,
                    device_type=device.type)
    best = float("inf")
    first_epoch = 1
    if args.resume:
        if not args.output.is_file() or not latest.is_file():
            raise FileNotFoundError(f"Resume requires {args.output} and {latest}.")
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if state["settings"] != settings or state["spec"] != model.spec:
            raise ValueError("Resume requires the original data, teacher, model and training settings.")
        model.load_state_dict(state["weights"])
        optimizer.load_state_dict(state["optimizer"])
        if bridge is not None:
            bridge.load_state_dict(state["bridge"])
        best = state["best"]
        first_epoch = state["epoch"] + 1
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device)
    if first_epoch > args.epochs:
        print(f"Checkpoint has already completed {first_epoch - 1} epochs.", flush=True)
    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        loss_sum = torch.zeros((), device=device)
        samples = 0
        for batch in train:
            batch = move(batch, device)
            prediction, features = model(*observations(batch))
            loss = behavior_loss(prediction, batch["target"]).mean()
            if is_student:
                with torch.no_grad():
                    teacher_plan, teacher_features = teacher(*observations(batch, clean=True))
                loss = loss + 0.5 * behavior_loss(prediction, teacher_plan).mean()
                loss = loss + 0.1 * torch.nn.functional.mse_loss(bridge(features), teacher_features)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            samples += len(prediction)
            loss_sum += loss.detach() * len(prediction)
        metrics = open_loop(model, [validation])
        score = metrics["planning_loss"]
        if not np.isfinite(score):
            raise FloatingPointError("Non-finite validation loss. Check the data and learning rate.")
        print(f"epoch={epoch} train_loss={loss_sum.item() / samples:.6f} validation={metrics}", flush=True)
        fields = dict(epoch=epoch, role=role, validation_loss=score, validation_metrics=metrics, domain_slots={"clean": -1})
        if score < best:
            best = score
            save(args.output, model, **fields)
        save(latest, model, **fields, best=best, settings=settings, optimizer=optimizer.state_dict(),
             bridge=bridge.state_dict() if bridge is not None else None,
             python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
             cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
