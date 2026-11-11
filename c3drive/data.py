import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch.utils.data import Dataset

from .model import HISTORY, HORIZON, IMAGE_HEIGHT, IMAGE_WIDTH, LIDAR_SIZE, STATE_SIZE

DOMAINS = ("clean", "camera-drop", "occlusion", "gps", "speed", "latency")
SPLITS = ("train", "validation", "fit", "calibrate", "threshold", "gate", "test", "stream")
INPUTS = ("image", "lidar", "state")
FIELDS = (*INPUTS, "clean_image", "clean_state", "target", "severity", "critical")
FRAME_FIELDS = (*INPUTS, "clean_image", "clean_state")
IMAGE_SHAPE = (9, IMAGE_HEIGHT, IMAGE_WIDTH)
SCHEMA = {
    "image": (IMAGE_SHAPE, np.uint8),
    "lidar": ((2, LIDAR_SIZE, LIDAR_SIZE), np.uint8),
    "state": ((STATE_SIZE,), np.float32),
    "clean_image": (IMAGE_SHAPE, np.uint8),
    "clean_state": ((STATE_SIZE,), np.float32),
    "target": ((HORIZON * 2 + 1,), np.float32),
    "severity": ((), np.float32),
    "critical": ((), np.uint8),
}


def sample_count(archive, path):
    value = archive["samples"]
    if value.ndim != 0 or value.dtype.kind not in "iu" or int(value) <= 0:
        raise ValueError(f"{path}: samples must be a positive integer scalar.")
    return int(value)


def read_arrays(archive, path, count):
    arrays = {}
    for name, (shape, dtype) in SCHEMA.items():
        value = archive[name]
        if value.shape != (count, *shape) or value.dtype != dtype:
            raise ValueError(f"{path}: {name} must have shape {(count, *shape)} and dtype {np.dtype(dtype)}.")
        if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"{path}: {name} contains non-finite values.")
        arrays[name] = value
    if ((arrays["severity"] < 0) | (arrays["severity"] > 1)).any() or (arrays["critical"] > 1).any():
        raise ValueError(f"{path}: severity must be in [0, 1] and critical must be binary.")
    return arrays


def check_route(archive, path, domain, split, route_splits):
    metadata = json.loads(str(archive["metadata"]))
    if metadata.get("domain") != domain:
        raise ValueError(f"{path}: metadata domain does not match its directory.")
    route = metadata["route"]
    identity = (metadata["town"], route["start"], route["end"])
    previous = route_splits.setdefault(identity, split)
    if previous != split:
        raise ValueError(f"{path}: route {identity} occurs in both {previous} and {split}.")
    return metadata


def inspect_dataset(root, domains, splits):
    report = {}
    route_splits = {}
    for domain in domains:
        for split in splits:
            files = []
            for path in episodes(root, domain, split):
                with np.load(path, allow_pickle=False) as archive:
                    count = sample_count(archive, path)
                    check_route(archive, path, domain, split, route_splits)
                    read_arrays(archive, path, count)
                files.append({"file": str(path.relative_to(root)), "samples": count})
            report[f"{domain}/{split}"] = {"episodes": len(files), "samples": sum(f["samples"] for f in files),
                                          "files": files}
    return report


def check_split_overlap(groups):
    route_splits = {}
    for domain, split, paths in groups:
        for path in paths:
            with np.load(path, allow_pickle=False) as archive:
                check_route(archive, path, domain, split, route_splits)


def episodes(root, domain, split):
    result = sorted((Path(root) / domain / split).glob("*.npz"))
    if not result:
        raise FileNotFoundError(f"No data for {domain}/{split} in {root}. Run collect first.")
    return result


class EpisodeData(Dataset):
    """A bounded cache of one recording per data-loader process."""

    def __init__(self, paths):
        self.paths = [Path(path) for path in paths]
        if not self.paths:
            raise ValueError("At least one episode is required.")
        self.ends = []
        total = 0
        for path in self.paths:
            with np.load(path, allow_pickle=False) as archive:
                missing = {*FIELDS, "samples", "metadata"} - set(archive.files)
                if missing:
                    raise ValueError(f"{path}: missing fields {sorted(missing)}.")
                total += sample_count(archive, path)
            self.ends.append(total)
        self.cached_path = None
        self.cache = {}

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = int(np.searchsorted(self.ends, index, side="right"))
        start = self.ends[episode - 1] if episode else 0
        offset = index - start
        path = self.paths[episode]
        if self.cached_path != path:
            with np.load(path, allow_pickle=False) as archive:
                self.cache = read_arrays(archive, path, self.ends[episode] - start)
            self.cached_path = path
        sample = {}
        for key, value in self.cache.items():
            if key in FRAME_FIELDS:
                frames = np.maximum(np.arange(offset - HISTORY + 1, offset + 1), 0)
                sample[key] = torch.from_numpy(value[frames])
            else:
                sample[key] = torch.as_tensor(value[offset])
        return sample


def write_episode(path, frames, targets, severity, critical, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(targets)
    arrays = {key: np.stack([frame[key] for frame in frames[:count]]) for key in FRAME_FIELDS}
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays, target=np.asarray(targets, np.float32),
                                severity=np.asarray(severity, np.float32), critical=np.asarray(critical, np.uint8),
                                samples=count, metadata=json.dumps(metadata))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def observations(batch, clean=False):
    if clean:
        return batch["clean_image"], batch["lidar"], batch["clean_state"]
    return tuple(batch[key] for key in INPUTS)
