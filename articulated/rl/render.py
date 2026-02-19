"""Render a trained RL policy on Reacher-v5.

Examples:
  python -m articulated.rl.render --checkpoint checkpoints/rl/baseline_ppo_tuned2.zip
  python -m articulated.rl.render --checkpoint checkpoints/rl/baseline_ppo_tuned2.zip --out logs/rl/demo.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Optional

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def load_model(algorithm: str, checkpoint: str):
    if algorithm == "ppo":
        return PPO.load(checkpoint, device="cpu")
    if algorithm == "sac":
        return SAC.load(checkpoint, device="cpu")
    raise ValueError(f"Unknown algorithm: {algorithm}")


def render_policy(
    checkpoint: str,
    algorithm: Literal["ppo", "sac"] = "ppo",
    episodes: int = 1,
    max_steps: int = 200,
    render_mode: Literal["human", "rgb_array"] = "rgb_array",
    out_path: Optional[str] = None,
    vecnorm_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> None:
    def make_env():
        return gym.make("Reacher-v5", render_mode=render_mode)

    env = make_env()
    vec_env = None
    if vecnorm_path:
        vec_env = DummyVecEnv([make_env])
        if seed is not None:
            vec_env.seed(seed)
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = load_model(algorithm, checkpoint)

    frames: list[np.ndarray] = []
    for ep in range(episodes):
        if vec_env is not None:
            obs = vec_env.reset()
        else:
            obs, _ = env.reset(seed=seed)
        done = False
        steps = 0
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            if vec_env is not None:
                obs, _, dones, _ = vec_env.step(action)
                done = bool(dones[0])
            else:
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            steps += 1
            if render_mode == "rgb_array":
                if vec_env is not None:
                    frames.append(vec_env.envs[0].render())
                else:
                    frames.append(env.render())

    if vec_env is not None:
        vec_env.close()
    env.close()

    if out_path and frames:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out, frames, fps=30)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render trained policy on Reacher-v5")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--algorithm", type=str, default="ppo")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--render-mode", type=str, default="rgb_array", choices=["human", "rgb_array"]
    )
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--vecnorm", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    render_policy(
        checkpoint=args.checkpoint,
        algorithm=args.algorithm,
        episodes=args.episodes,
        max_steps=args.max_steps,
        render_mode=args.render_mode,
        out_path=args.out,
        vecnorm_path=args.vecnorm,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
