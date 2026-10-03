import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from followline_rl.core import Config, PDController, perceive, observation, steering
from followline_rl.env import FollowLineEnv
from followline_rl.policy import NumpyActor


def frame(x=160, red=True):
    image = np.zeros((240, 320, 3), np.uint8)
    if red:
        image[:, max(0,x-4):min(320,x+5), 2] = 255
    return image


class ScriptedBackend:
    """Contract fixture only. This is NOT a robotics simulator or a learned policy."""
    def __init__(self, frames=None, stale=False):
        self.frames = frames or [frame()]
        self.stale = stale
        self.stopped = False
        self.calls = []

    def reset(self):
        self.stamp = self.index = 0
        self.stopped = False
        return frame(), 0.0

    def advance(self, v,w,dt):
        self.calls.append((v,w,dt))
        if not self.stale:
            self.stamp += dt
        image = self.frames[min(self.index, len(self.frames)-1)]
        self.index += 1
        return image, self.stamp

    def stop(self):
        self.stopped = True

    def close(self):
        self.stop()


def test_gym_contract():
    check_env(FollowLineEnv(ScriptedBackend()), skip_render_check=True)


@pytest.mark.parametrize("x,sign", [(50,-1),(160,0),(270,1)])
def test_red_detection(x,sign):
    obs, _ = perceive(frame(x))
    assert np.all(obs[3:] == 1)
    if sign:
        assert np.all(np.sign(obs[:3]) == sign)
    else:
        assert np.max(np.abs(obs[:3])) < 0.01


def test_bgr_blue_is_not_red():
    image = frame()[:,:,::-1].copy()
    assert not perceive(image)[0][3:].any()


def test_missing_line_not_centered():
    obs = observation(frame(red=False), 0)
    assert np.all(obs[3:6] == 0)


def test_loss_termination_and_stop():
    backend = ScriptedBackend([frame(red=False)])
    env = FollowLineEnv(backend)
    env.reset()
    for i in range(3):
        _, reward, done, truncated, _ = env.step(np.zeros(1, np.float32))
        assert done == (i==2)
        assert reward < 0 and not truncated
    assert backend.stopped
    with pytest.raises(gym.error.ResetNeeded):
        env.step([0])


def test_time_limit_is_truncation():
    env = FollowLineEnv(ScriptedBackend(), Config(max_steps=1))
    env.reset()
    _, _, done, truncated, info = env.step([0])
    assert not done and truncated and info["end_reason"] == "time_limit"


def test_action_clip_and_previous_action():
    backend = ScriptedBackend()
    env = FollowLineEnv(backend)
    env.reset()
    obs, *_ = env.step([20])
    assert obs[-1] == 1 and backend.calls[-1][1] == 1


@pytest.mark.parametrize("action", [[np.nan], [0,1]])
def test_bad_actions_stop(action):
    backend = ScriptedBackend()
    env = FollowLineEnv(backend)
    env.reset()
    with pytest.raises(ValueError):
        env.step(action)
    assert backend.stopped


def test_stale_camera_stops():
    backend = ScriptedBackend(stale=True)
    env = FollowLineEnv(backend)
    env.reset()
    with pytest.raises(RuntimeError, match="timestamp"):
        env.step([0])
    assert backend.stopped


def test_reset_requires_line():
    backend = ScriptedBackend()
    backend.reset = lambda: (frame(red=False), 0)
    with pytest.raises(RuntimeError, match="No red line"):
        FollowLineEnv(backend).reset()
    assert backend.stopped


def test_pd_turn_sign():
    controller = PDController(Config())
    assert controller(observation(frame(250), 0))[0] < 0


def test_config_roundtrip(tmp_path):
    p = tmp_path / "config.json"
    cfg = Config(speed=0.2)
    cfg.save(p)
    assert Config.load(p) == cfg


def test_numpy_actor(tmp_path):
    p = tmp_path / "actor.npz"
    np.savez(p, w0=np.zeros((64,7)), b0=np.zeros(64),
             w1=np.zeros((64,64)), b1=np.zeros(64), w2=np.zeros((1,64)), b2=np.array([2.0]))
    np.testing.assert_array_equal(NumpyActor(p)(np.zeros(7)), [1])
