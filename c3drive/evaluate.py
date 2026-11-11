import torch

from .data import move, observations
from .model import HORIZON, POSITION_SCALE, SPEED_SCALE, behavior_loss


@torch.inference_mode()
def open_loop(model, batches, detector_factory=None):
    """Score held-out expert trajectories using sample-weighted metrics."""
    model.eval()
    device = next(model.parameters()).device
    totals = torch.zeros(4, dtype=torch.float64)
    samples = 0
    for batch_iter in batches:
        detector = detector_factory() if detector_factory else None
        for batch in batch_iter:
            batch = move(batch, device)
            features = model.encode(*observations(batch))
            if detector:
                slots = []
                for vector in features.cpu().numpy():
                    detector.observe(vector)
                    slots.append(detector.slot)
                plan = model.plan(features, torch.tensor(slots, device=device))
            else:
                plan = model.plan(features)
            target = batch["target"]
            displacement = (plan[:, :-1] - target[:, :-1]).reshape(-1, HORIZON, 2) * POSITION_SCALE
            distance = displacement.norm(dim=-1)
            speed_error = (plan[:, -1] - target[:, -1]).abs() * SPEED_SCALE
            errors = torch.stack((behavior_loss(plan, target), distance.mean(dim=1), distance[:, -1], speed_error))
            totals += errors.sum(dim=1, dtype=torch.float64).cpu()
            samples += len(plan)
    if not samples:
        raise ValueError("Evaluation needs at least one sample.")
    keys = ("planning_loss", "ade_m", "fde_m", "speed_mae_mps")
    return {"samples": samples, **dict(zip(keys, (totals / samples).tolist()))}
