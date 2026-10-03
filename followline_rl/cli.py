import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from .core import Config, PDController, perceive
from .env import FollowLineEnv


def main():
    parser = argparse.ArgumentParser(description="FollowLine pilot: inspect, reset smoke test, baseline, PPO")
    parser.add_argument("mode", choices=["preflight", "smoke", "baseline", "train", "evaluate", "export"])
    parser.add_argument("--config", help="Saved config.json (required with evaluate/export)")
    parser.add_argument("--camera", default="/f1/camera/image_raw")
    parser.add_argument("--cmd", default="/f1/cmd_vel")
    parser.add_argument("--world", default="default")
    parser.add_argument("--out", default="runs/pilot")
    parser.add_argument("--model", default="runs/pilot/ppo.zip")
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1 or args.episodes < 1:
        parser.error("steps/episodes must be positive")
    if args.mode in ("evaluate", "export") and not args.config:
        parser.error("evaluate/export require --config from the training run")
    cfg = Config.load(args.config) if args.config else Config(
        camera_topic=args.camera, cmd_topic=args.cmd, world=args.world)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.json")
    if args.mode == "export":
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")
        # MlpPolicy default actor: Linear(7,64), Tanh, Linear(64,64), Tanh, Linear(64,1).
        layers = [layer for layer in model.policy.mlp_extractor.policy_net if hasattr(layer, "weight")]
        if len(layers) != 2:
            raise RuntimeError("Unexpected network. Export supports this project's default PPO policy only.")
        arrays = {}
        for i, layer in enumerate(layers + [model.policy.action_net]):
            arrays[f"w{i}"] = layer.weight.detach().cpu().numpy()
            arrays[f"b{i}"] = layer.bias.detach().cpu().numpy()
        np.savez(out / "actor.npz", **arrays)
        print(f"Exported deterministic actor to {out / 'actor.npz'}")
        return
    from .ros_backend import RosBackend
    backend = RosBackend(cfg)
    env = FollowLineEnv(backend, cfg)
    try:
        if args.mode == "preflight":
            frame, report = backend.inspect()
            features, mask = perceive(frame)
            report["line_features_far_to_near"] = features.tolist()
            report["shape"] = list(frame.shape)
            cv2.imwrite(str(out / "camera.png"), frame)
            cv2.imwrite(str(out / "red_mask.png"), mask)
            (out / "preflight.json").write_text(json.dumps(report, indent=2)+"\n")
            print(json.dumps(report, indent=2))
            return
        if args.mode == "train":
            # Establish Gazebo communication before loading PyTorch.
            backend._control("pause: false")
            backend._control("pause: true")

            import torch
            torch.set_num_threads(1)

            from stable_baselines3 import PPO
            from stable_baselines3.common.monitor import Monitor
            monitor = Monitor(env, filename=str(out / "monitor.csv"))
            model = PPO("MlpPolicy", monitor, n_steps=256, batch_size=64,
                        learning_rate=3e-4, seed=args.seed, device="cpu", verbose=1,
                        policy_kwargs={"net_arch": dict(pi=[64,64], vf=[64,64])})
            try:
                model.learn(total_timesteps=args.steps)
            except KeyboardInterrupt:
                print("Interrupted; saving the current policy (may still be untrained).")
            finally:
                model.save(str(out / "ppo"))
            return
        model = None
        if args.mode == "evaluate":
            from stable_baselines3 import PPO
            model = PPO.load(args.model, device="cpu")
        results = []
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            if args.mode == "smoke":
                frame, _ = backend.inspect()
                cv2.imwrite(str(out / f"reset_{episode}.png"), frame)
            controller = PDController(cfg)
            total = 0.0
            errors, durations = [], []
            limit = min(10, args.steps) if args.mode == "smoke" else cfg.max_steps
            for _ in range(limit):
                action = np.zeros(1, np.float32) if args.mode == "smoke" else (
                    model.predict(obs, deterministic=True)[0] if model else controller(obs))
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                errors.append(abs(info["image_error"]) if info["line_visible"] else 1.0)
                durations.append(info["elapsed_sim_time"])
                if terminated or truncated:
                    break
            backend.stop()
            result = {"episode": episode, "steps": len(errors), "return": total,
                      "mean_abs_image_error": float(np.mean(errors)),
                      "mean_step_sim_seconds": float(np.mean(durations)),
                      "end_reason": info["end_reason"] if args.mode != "smoke" else
                      (info["end_reason"] if terminated else "smoke_limit")}
            results.append(result)
            print(json.dumps(result), flush=True)
            (out / f"{args.mode}.json").write_text(json.dumps(results, indent=2)+"\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
