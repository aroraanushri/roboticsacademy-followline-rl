# RoboticsAcademy FollowLine RL pilot

A standalone Gymnasium adapter and Stable-Baselines3 PPO experiment for
JdeRobot RoboticsAcademy's FollowLine exercise. Includes a trained checkpoint,
NumPy actor export, editor runner, and recorded evaluation results.

Developed by Anushri Arora following
[JdeRobot discussion #404](https://github.com/orgs/JdeRobot/discussions/404).
This is an experimental contribution candidate, not an upstream integration or
a migration of RL-Studio. Reinforcement learning offers a useful robotics
learning path: defining observations, actions and rewards, training in simulation,
and deploying a policy through a robot interface. This pilot provides a small
practical example relevant to Physical AI learning.

## Evidence and scope

The bundled PPO checkpoint contains **2,048 training timesteps**. Three
subsequent deterministic evaluations each reached the 1,000-step time limit
without triggering the three-observation line-loss termination.

| Episode | Steps | Return | Mean absolute image error | Mean step simulation seconds |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1000 | 978.4411 | 0.021514 | 0.19305 |
| 1 | 1000 | 978.3148 | 0.021638 | 0.20005 |
| 2 | 1000 | 978.1881 | 0.021768 | 0.19645 |

These are saved reports from the contributor's live container, not a new
simulation run performed while packaging. The contributor reports recording
an editor demonstration; the video is supplied separately on GitHub and was
not independently reviewed during packaging. Lap completion, robustness across
seeds and generalization to other circuits have not been established.

## Design

- Observation: red-line horizontal offsets at 55%, 70%, and 85% of image height,
  three visibility flags, and previous normalized steering (seven features).
- Action: steering in [-1, 1], mapped to +/-1 rad/s; forward speed fixed at 0.5 m/s.
- Reward: line alignment minus a steering-change penalty; penalties on line loss.
- Termination: near line absent for three observations; truncation at 1,000 steps.
- PPO: CPU, two 64-unit Tanh actor layers, 256-step rollouts, batch size 64, seed 42.
- ROS camera/velocity topics: `/f1/camera/image_raw` and `/f1/cmd_vel`.
- Persistent Gazebo service requests; pose monitoring in a separate process.
- Robot-only pose reset for `f1` in world `default`; the simulation clock stays continuous.

**Circuit-specific reset:** `ros_backend.py` currently hardcodes the simple-circuit
start position `(53.462, -10.7341, 0.004068)` and yaw `-1.57`. Changing the
world flag alone does not make another circuit compatible. The reset sends zero
velocity, moves the robot, waits for a camera frame and checks reported pose;
it does not explicitly reset joint states or prove all dynamics are identical.

## Start RoboticsAcademy

Tested by the contributor inside an existing ROS 2 Humble RoboticsAcademy
container, with Gazebo reporting 8.15.0 and Python bindings `gz.transport13` /
`gz.msgs10`. The mutable image tag was `jderobot/robotics-academy:latest`;
a reproducible image digest was not captured. The checked Academy source
reference was `66a7eab9a3f2dc8f520cf24c1a62b08a6e0bd0ec`;
it is not a verified identity for the running image.

Use the [RoboticsAcademy repository](https://github.com/JdeRobot/RoboticsAcademy)
for its container setup. In the contributor's existing host workspace:

```bash
cd ~/followline-workspace
sudo docker compose start
```

Open `http://localhost:7164/academy`, select FollowLine / Simple Circuit, and
connect. Wait for the camera and simulator. For external training/evaluation,
paste `academy/keepalive.py` into the Python editor and run it. It does not send
velocity commands. Stop other controllers.

## Install this project in the simulator container

On the host, from the parent of this freshly extracted repository:

```bash
sudo docker cp roboticsacademy-followline-rl developer-container:/tmp/followline-rl
sudo docker exec -it developer-container bash
```

This copy command assumes `/tmp/followline-rl` does not already exist. For an
existing installation, back it up before replacing files; Docker copying into
an existing directory can produce an extra nested folder. The contributor's
working container already has the project and virtual environment installed.

Inside a fresh container installation:

```bash
source /opt/ros/humble/setup.bash
apt-get update
apt-get install -y python3.10-venv
cd /tmp/followline-rl
python3 -m venv --system-site-packages /tmp/followline-venv
source /tmp/followline-venv/bin/activate
python -m pip install -e '.[test]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Use system OpenCV, ROS messages, `cv_bridge`, and Gazebo Python bindings from
Academy. The package does not install these system dependencies. Keep NumPy
below 2 for this tested stack. Disabling pytest plugin autoload avoids the
unrelated ROS Humble `launch_testing` hook conflict seen with pytest 9.

## Validate, train, evaluate

Run inside the container with ROS sourced and the virtual environment active:

```bash
bash scripts/preflight.sh
python -m followline_rl.cli preflight --out runs/new-preflight
python -m followline_rl.cli smoke --config runs/new-preflight/config.json \
  --episodes 10 --out runs/new-smoke
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[train]'
python -m followline_rl.cli train --config runs/new-preflight/config.json \
  --steps 2048 --out runs/new-training
```

Preflight is read-only; resume the simulation first so a new subscriber gets
camera data. Smoke drives straight for up to ten steps per episode and checks
resets. Each `train` command creates a new model; it does not resume training.
Use a new output directory. Training rounds to complete 256-step rollouts.

Evaluate the included checkpoint or export it again:

```bash
python -m followline_rl.cli evaluate --config runs/ppo-pilot-4/config.json \
  --model runs/ppo-pilot-4/ppo.zip --episodes 3 --out runs/new-evaluation
python -m followline_rl.cli export --config runs/ppo-pilot-4/config.json \
  --model runs/ppo-pilot-4/ppo.zip --out runs/new-export
```

CLI control leaves Gazebo paused. Use the Academy controls to resume for normal
editor execution. Never run an external controller and an editor controller at
the same time; reset checks reject another velocity publisher.

## Run the exported policy in the editor

Stop external training/evaluation, reset and resume the exercise, then paste
`academy/run_policy.py` into the FollowLine Python editor and click Run.
It defaults to the included PPO actor at
`/tmp/followline-rl/runs/ppo-pilot-4/actor.npz`. Set
`USE_TRAINED_POLICY = False` only to select the PD baseline. No PyTorch is needed
for NumPy actor inference; the source and exported files must be accessible to
the editor's Python interpreter. Missing line stops the demo immediately.

The editor uses wall-clock pacing and HAL images without freshness timestamps.
Training uses camera simulation timestamps and pause/unpause requests. Requested
training intervals are 0.1 seconds, but evaluation measured about 0.19–0.20
seconds per observation. These timing differences need further evaluation.

## Results, limitations and next contributions

See `VALIDATION.md` for evidence and packaging checks, and `MODEL_CARD.md` for
checkpoint details. The 100-step baseline report is shorter than the PPO
1,000-step evaluations, so it is not a controlled comparison. Image error is a
normalized alignment proxy, not metric cross-track error or lap progress.

Useful follow-ups: configurable circuit reset poses, reliable timestamped
editor stepping, lap/progress measurement, matched baseline comparisons,
multiple training seeds, pinned container dependencies, and maintainer-guided
upstream packaging. This pilot does not establish sim-to-real transfer.

## License and acknowledgments

Original pilot code, documentation and bundled policy artifacts are released
under MIT (`LICENSE`). RoboticsAcademy, RoboticsInfrastructure, Gymnasium,
Stable-Baselines3, ROS, Gazebo and their assets retain their own licenses.
Robot/world assets are provided by the existing Academy installation and are
not redistributed here. This project uses the Academy HAL/WebGUI/Frequency
interfaces; it does not include copies of their implementation.
