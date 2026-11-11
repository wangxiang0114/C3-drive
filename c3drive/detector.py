import math

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from .data import observations
from .train import loader


def normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def distances(features, profile):
    delta = normalize(features) - profile["center"].numpy()
    return np.einsum("...i,ij,...j->...", delta, profile["precision"].numpy(), delta)


@torch.inference_mode()
def blocks(model, paths, block_size, batch_size):
    device = next(model.parameters()).device
    result = []
    for path in paths:
        features = [model.encode(*[value.to(device) for value in observations(batch)]).cpu().numpy()
                    for batch in loader([path], batch_size)]
        vectors = np.concatenate(features)
        count = len(vectors) // block_size
        if count:
            result.append(vectors[:count * block_size].reshape(count, block_size, -1).mean(axis=1))
    if not result:
        raise ValueError("No complete detector blocks. Increase recording length or reduce block size.")
    return result


def fit_profile(model, fit_paths, cal_paths, block_size, batch_size):
    fit = normalize(np.concatenate(blocks(model, fit_paths, block_size, batch_size)))
    if len(fit) < 2:
        raise ValueError("Detector covariance needs at least two blocks.")
    estimate = LedoitWolf().fit(fit)
    profile = {"center": torch.from_numpy(estimate.location_), "precision": torch.from_numpy(estimate.precision_)}
    scores = distances(np.concatenate(blocks(model, cal_paths, block_size, batch_size)), profile)
    profile["scores"] = torch.from_numpy(np.sort(scores))
    return profile


class Detector:
    def __init__(self, profiles, mapping, block_size, threshold):
        self.profiles = profiles
        self.mapping = mapping
        self.block_size = block_size
        self.threshold = threshold
        self.buffer = []
        self.evidence = 0.0
        self.peak = 0.0
        self.slot = -1

    @staticmethod
    def p_value(vector, profile):
        scores = profile["scores"].numpy()
        rank = np.searchsorted(scores, distances(vector, profile), side="left")
        return (1 + len(scores) - rank) / (1 + len(scores))

    def route_block(self, vector):
        values = {domain: self.p_value(vector, profile) for domain, profile in self.profiles.items()}
        best = max(values, key=values.get)
        self.slot = self.mapping[best]
        self.evidence = max(0.0, self.evidence - math.log(values[best]) - 1.0)
        self.peak = max(self.peak, self.evidence)
        return self.evidence > self.threshold

    def observe(self, vector):
        self.buffer.append(vector)
        if len(self.buffer) < self.block_size:
            return False
        vector = np.mean(self.buffer, axis=0)
        self.buffer.clear()
        return self.route_block(vector)


def calibrate_threshold(model, profiles, mapping, paths, block_size, batch_size, alpha):
    peaks = []
    for episode in blocks(model, paths, block_size, batch_size):
        detector = Detector(profiles, mapping, block_size, math.inf)
        for vector in episode:
            detector.route_block(vector)
        peaks.append(detector.peak)
    rank = math.ceil((len(peaks) + 1) * (1 - alpha))
    if rank > len(peaks):
        raise ValueError(f"Need at least {math.ceil(1 / alpha) - 1} independent threshold recordings for alpha={alpha}.")
    return sorted(peaks)[rank - 1]
