"""Identical perception/action mapping for training and Academy inference."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Config:
    camera_topic: str = "/f1/camera/image_raw"
    cmd_topic: str = "/f1/cmd_vel"
    world: str = "default"
    dt: float = 0.1
    speed: float = 0.5
    max_w: float = 1.0
    max_steps: int = 1000
    lost_limit: int = 3
    timeout: float = 15.0

    def __post_init__(self):
        for name in ("dt", "speed", "max_w", "timeout"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.max_steps < 1 or self.lost_limit < 1:
            raise ValueError("Episode limits must be positive")

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def perceive(bgr):
    """Return three horizontal offsets and visibility flags, far to near.

    Offsets are image-space proxies, not metric cross-track errors. Select the
    largest red component in each strip to reduce sensitivity to small specks.
    """
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("Expected a uint8 HxWx3 BGR frame")
    height, width = bgr.shape[:2]
    if height < 20 or width < 20:
        raise ValueError("Camera frame is too small")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 90, 60), (12, 255, 255)) | cv2.inRange(
        hsv, (168, 90, 60), (179, 255, 255)
    )
    offsets = np.zeros(3, np.float32)
    visible = np.zeros(3, np.float32)
    half = max(2, int(height * 0.02))
    for i, fraction in enumerate((0.55, 0.70, 0.85)):
        row = int(height * fraction)
        strip = mask[max(0, row-half):min(height, row+half+1)]
        count, _, stats, centers = cv2.connectedComponentsWithStats(strip)
        if count > 1:
            component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            if stats[component, cv2.CC_STAT_AREA] >= max(5, int(strip.size * 0.002)):
                offsets[i] = 2 * centers[component, 0] / (width - 1) - 1
                visible[i] = 1
    return np.concatenate((offsets, visible)).astype(np.float32), mask


def observation(bgr, previous_action):
    features, _ = perceive(bgr)
    return np.append(features, np.float32(previous_action)).astype(np.float32)


def steering(action, config):
    value = np.asarray(action, dtype=np.float32)
    if value.shape != (1,) or not np.isfinite(value).all():
        raise ValueError("Action must have shape (1,) and a finite value")
    normalized = float(np.clip(value[0], -1, 1))
    return normalized, normalized * config.max_w


class PDController:
    """Camera feature baseline. Positive image error needs right (negative yaw)."""
    def __init__(self, config):
        self.config = config
        self.previous = None

    def __call__(self, obs):
        if not obs[5]:
            return np.array([0], np.float32)
        error = float(obs[2])
        if obs[3]:
            error = 0.4 * error + 0.6 * float(obs[0])
        derivative = 0 if self.previous is None else (error-self.previous)/self.config.dt
        self.previous = error
        w = -1.8 * error - 0.05 * derivative
        return np.array([np.clip(w/self.config.max_w, -1, 1)], np.float32)
