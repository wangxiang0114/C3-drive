import torch
from torch import nn
from torch.nn import functional as F

HISTORY = 3
HORIZON = 5
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 112
LIDAR_SIZE = 48
STATE_SIZE = 8
POSITION_SCALE = 10.0
SPEED_SCALE = 10.0


def encoder(channels, width):
    return nn.Sequential(
        nn.Conv2d(channels, width, 5, stride=2, padding=2), nn.GroupNorm(4, width), nn.SiLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.GroupNorm(4, 2 * width), nn.SiLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=2, padding=1), nn.GroupNorm(4, 2 * width), nn.SiLU(),
        nn.AdaptiveAvgPool2d((2, 4)), nn.Flatten(),
    )


class Adapter(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dim, rank), nn.SiLU(), nn.Linear(rank, dim))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features):
        return features + self.layers(features)


class Driver(nn.Module):
    def __init__(self, width=16, dim=64, slots=4, rank=12):
        super().__init__()
        self.spec = dict(width=width, dim=dim, slots=slots, rank=rank)
        self.camera = encoder(9, width)
        self.lidar = encoder(2, width)
        self.fusion = nn.Sequential(nn.Linear(32 * width + STATE_SIZE, dim), nn.LayerNorm(dim), nn.SiLU())
        self.temporal = nn.GRU(dim, dim, batch_first=True)
        self.adapters = nn.ModuleList(Adapter(dim, rank) for _ in range(slots))
        self.planner = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 2 * HORIZON + 1))

    def encode(self, image, lidar, state):
        batch, frames = state.shape[:2]
        camera = self.camera(image.flatten(0, 1).float() / 255)
        points = self.lidar(lidar.flatten(0, 1).float() / 255)
        fused = self.fusion(torch.cat((camera, points, state.flatten(0, 1)), dim=1))
        _, hidden = self.temporal(fused.reshape(batch, frames, -1))
        return hidden[-1]

    def plan(self, features, slot=-1):
        if isinstance(slot, int):
            return self.planner(features if slot < 0 else self.adapters[slot](features))
        output = self.planner(features)
        for index in torch.unique(slot).tolist():
            if index >= 0:
                mask = slot == index
                output[mask] = self.planner(self.adapters[index](features[mask]))
        return output

    def forward(self, image, lidar, state, slot=-1):
        features = self.encode(image, lidar, state)
        return self.plan(features, slot), features


def behavior_loss(prediction, target):
    return F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
