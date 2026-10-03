# Publish this contribution

Create an empty public repository on your GitHub account named
`roboticsacademy-followline-rl`. Do not initialize it with another README.
Extract this archive on the Ubuntu host, then:

```bash
cd ~/followline-workspace/roboticsacademy-followline-rl
git init -b main
git add .
git status --short
git commit -m "Add FollowLine Gymnasium PPO pilot and evaluation evidence"
git remote add origin https://github.com/aroraanushri/roboticsacademy-followline-rl.git
git push -u origin main
```

Use your configured GitHub authentication. If Git reports missing author
identity, configure your preferred name and verified GitHub email for this
repository before committing. Do not enter account passwords into scripts.

Attach the recorded video directly in discussion #404 using GitHub's attachment
control or by dragging it into the comment editor. Include the public repository
link. No public repository or discussion comment was created while preparing
this archive.

Suggested update (review before posting):

Hi! Following your suggestion, I built a standalone Gymnasium environment for
RoboticsAcademy's FollowLine exercise and trained a small PPO policy with
Stable-Baselines3 on the existing ROS 2 Humble setup.

The included checkpoint has 2,048 training timesteps. Three deterministic
evaluation episodes each reached the 1,000-step time limit without triggering
line-loss termination; mean absolute normalized image error was about 0.02164.
Robot-only resets were checked with a separate Gazebo pose-monitor process.
I have also recorded a demonstration of the exported NumPy actor in the
RoboticsAcademy editor and am attaching it here.

Code, setup instructions, checkpoint and raw results:
https://github.com/aroraanushri/roboticsacademy-followline-rl

This is a first pilot: lap completion and generalization are not measured yet.
Would you prefer the next contribution as a FollowLine tutorial/policy example,
or a reusable training adapter in another repository?
