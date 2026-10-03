"""Optional algorithm/export integration test on scripted frames, NOT Gazebo."""
import subprocess
import sys
import numpy as np
import pytest

sb3 = pytest.importorskip("stable_baselines3")
from followline_rl.core import Config
from followline_rl.env import FollowLineEnv
from followline_rl.policy import NumpyActor
from test_pilot import ScriptedBackend


def test_ppo_learn_save_export_equivalence(tmp_path):
    cfg = Config(max_steps=8)
    env = FollowLineEnv(ScriptedBackend(), cfg)
    model = sb3.PPO("MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1,
                    seed=42, device="cpu",
                    policy_kwargs={"net_arch": dict(pi=[64,64], vf=[64,64])})
    model.learn(total_timesteps=16)
    assert model.num_timesteps == 16
    model.save(str(tmp_path / "ppo"))
    cfg.save(tmp_path / "config.json")
    subprocess.run([sys.executable, "-m", "followline_rl.cli", "export",
                    "--config", str(tmp_path / "config.json"),
                    "--model", str(tmp_path / "ppo.zip"), "--out", str(tmp_path)], check=True)
    actor = NumpyActor(tmp_path / "actor.npz")
    env.observation_space.seed(9)
    for _ in range(20):
        obs = env.observation_space.sample()
        expected = model.predict(obs, deterministic=True)[0]
        np.testing.assert_allclose(actor(obs), expected, atol=1e-6, rtol=1e-5)
    env.close()
