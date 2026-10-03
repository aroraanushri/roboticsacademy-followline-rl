"""Paste into the FollowLine Python editor after installing/mounting the pilot.

The Academy process uses its own Python interpreter. The source path below
lets it use shared perception and NumPy inference without installing torch.
"""
import sys
from pathlib import Path
import numpy as np
import HAL
import WebGUI
import Frequency

ROOT = Path("/tmp/followline-rl")
sys.path.insert(0, str(ROOT))
from followline_rl.core import Config, observation, steering, PDController
from followline_rl.policy import NumpyActor

# The bundled exported PPO actor is the default; False selects the PD baseline.
USE_TRAINED_POLICY = True
config = Config.load(ROOT / "runs/ppo-pilot-4/config.json") if USE_TRAINED_POLICY else Config()
policy = NumpyActor(ROOT / "runs/ppo-pilot-4/actor.npz") if USE_TRAINED_POLICY else PDController(config)
previous = 0.0
try:
    while True:
        image = HAL.getImage()
        if image is None:
            HAL.setV(0)
            HAL.setW(0)
            Frequency.tick(10)
            continue
        obs = observation(image, previous)
        WebGUI.showImage(image)
        if not obs[5]:
            # Stop on missing line during the demo; reset in the UI to restart.
            print("Red line lost. Reset the exercise before continuing.")
            break
        previous, w = steering(policy(obs), config)
        HAL.setV(config.speed)
        HAL.setW(w)
        Frequency.tick(1/config.dt)
finally:
    HAL.setV(0)
    HAL.setW(0)
