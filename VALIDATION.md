# Validation record

## Contributor live-run evidence

- Container: ROS 2 Humble; Gazebo 8.15.0; transport13/msgs10 Python imports passed.
- Ten smoke episodes with pose worker: saved `smoke.json` and reset images.
- Baseline: one 100-step episode, return 99.4713, time-limit ending.
- PPO checkpoint metadata: 2,048 timesteps.
- Evaluation: three 1,000-step episodes, all `time_limit`; raw JSON included.
- Video: contributor reports recording editor demonstration on October 3, 2026;
  supplied separately, not reviewed during packaging.
- Earlier contributor test output: 16 passed, one optional SB3 test skipped.

## Packaging changes

Removed old implementation backups, generated caches/build metadata, and failed
or superseded run folders. Retained current source, PPO pilot 4, evaluation,
preflight, successful pose-worker smoke, and short baseline artifacts.
Updated runner defaults to the bundled actor and added waiting for a missing
initial camera frame. Removed an unused in-process pose callback; the live
worker/service implementation otherwise remains unchanged. These packaging
changes have not been revalidated in the live simulator.

Packaging checks and their exact outcomes are added below. Offline tests use
scripted frames; they do not validate ROS transport, Gazebo resets or real laps.

## Packaging checks (October 3, 2026)

- Uploaded ZIP CRC check passed.
- All retained Python files parsed successfully.
- Offline tests: **16 passed, 1 skipped** (0.23 s); Python 3.12 review runtime,
  NumPy 1.26.4, OpenCV 4.11.0, Gymnasium 1.3.0, pytest 9.1.1.
- Optional SB3 learn/export equivalence test skipped because SB3/PyTorch are
  absent from this review environment. No new live ROS/Gazebo test performed.
- Bundled actor loaded with pickle disabled and produced a finite output of
  approximately 0.00446026 for a zero observation; no SB3 equivalence claim.
- Preserved run artifacts are byte-identical to the supplied archive; hashes
  are listed in `ARTIFACT_SHA256.json`.
- Checkpoint system metadata: Python 3.10.12, SB3 2.9.0, PyTorch 2.14.1+cpu,
  NumPy 1.24.3, Gymnasium 1.3.0, cloudpickle 3.1.2; GPU disabled.
