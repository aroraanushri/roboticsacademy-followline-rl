# Included PPO pilot checkpoint

- Author: Anushri Arora; experiment: RoboticsAcademy FollowLine / Simple Circuit.
- Algorithm: Stable-Baselines3 PPO, MlpPolicy, deterministic evaluation.
- Budget: 2,048 timesteps verified from checkpoint metadata; seed 42 in training code.
- Actor: two 64-unit Tanh layers; seven camera-derived features; one steering output.
- Files: `runs/ppo-pilot-4/ppo.zip` (SB3 checkpoint), `actor.npz` (NumPy weights),
  `config.json` (environment settings), `monitor.csv` (training episode log).
- Intended use: educational simulation example in the tested Academy container.
- Evaluation: three 1,000-step episodes; reports in `runs/evaluation-pilot-4/`.
- Limitations: one circuit and checkpoint; no held-out track or multi-seed study;
  no lap detector, hardware validation, metric tracking error, or convergence claim.
- Provenance: contributor-supplied live-run files, preserved during packaging.
- License: MIT for these pilot artifacts; simulator assets remain upstream.

SB3 loading uses serialized model data; only load checkpoints from trusted sources.
The exported actor uses `numpy.load(..., allow_pickle=False)`.
