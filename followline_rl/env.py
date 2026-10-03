import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .core import Config, observation, steering


class FollowLineEnv(gym.Env):
    """Backend protocol: reset(), advance(v,w,dt), stop(), close().

    reset/advance return (BGR image, image simulation timestamp in seconds).
    Live backend pauses Gazebo before returning, including during PPO updates.
    """
    metadata = {"render_modes": []}

    def __init__(self, backend, config=None):
        self.config = config or Config()
        self.backend = backend
        self.action_space = spaces.Box(-1, 1, (1,), np.float32)
        self.observation_space = spaces.Box(
            np.array([-1,-1,-1,0,0,0,-1], np.float32),
            np.ones(7, np.float32), dtype=np.float32
        )
        self.done = True
        self.steps = self.lost = 0
        self.previous_action = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.done = True
        try:
            image, self.stamp = self.backend.reset()
            obs = observation(image, 0)
            if not obs[5]:
                raise RuntimeError("No red line at reset in the near strip. Check spawn pose and diagnostic image.")
        except BaseException:
            self.backend.stop()
            raise
        self.steps = self.lost = 0
        self.previous_action = 0.0
        self.done = False
        return obs, {"image_sim_time": self.stamp}

    def step(self, action):
        if self.done:
            raise gym.error.ResetNeeded("Call reset before stepping")
        try:
            normalized, w = steering(action, self.config)
            image, stamp = self.backend.advance(self.config.speed, w, self.config.dt)
            if not np.isfinite(stamp) or stamp <= self.stamp:
                raise RuntimeError("Camera timestamp did not advance; refusing stale observation")
            elapsed = stamp - self.stamp
            self.stamp = stamp
            obs = observation(image, normalized)
        except BaseException:
            self.done = True
            self.backend.stop()
            raise
        self.steps += 1
        self.lost = 0 if obs[5] else self.lost + 1
        terminated = self.lost >= self.config.lost_limit
        truncated = self.steps >= self.config.max_steps and not terminated
        reward = (1 - abs(float(obs[2]))) if obs[5] else -1.0
        reward -= 0.05 * abs(normalized - self.previous_action)
        if terminated:
            reward = -5.0
        self.previous_action = normalized
        self.done = terminated or truncated
        if self.done:
            self.backend.stop()
        info = {"image_sim_time": stamp, "elapsed_sim_time": elapsed,
                "image_error": float(obs[2]), "line_visible": bool(obs[5]),
                "steps": self.steps, "end_reason": "line_lost" if terminated else
                "time_limit" if truncated else "running"}
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.backend.close()
