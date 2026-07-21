#!/usr/bin/env python
"""Run a single training run: one env, one hard-context handling policy, one seed.

Example:
    python scripts/run_experiment.py --env cartpole --handling-policy downweight_revisit --seed 0

    # short smoke test
    python scripts/run_experiment.py --env cartpole --handling-policy downweight_revisit \
        --seed 0 --total-timesteps 4000 --eval-freq 1000 --n-envs 4 --n-steps 128
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.config import load_config  # noqa: E402
from src.training.train import run_training  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--env",
        required=True,
        choices=[
            "cartpole",
            "lunarlander",
            "pendulum",
            "acrobot",
            "mountaincar",
            "mountaincarcontinuous",
            "bipedalwalker",
        ],
    )
    parser.add_argument(
        "--handling-policy",
        required=True,
        choices=["ignore", "hard_exclude", "downweight_revisit", "replay_priority"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None, help="PPO rollout length per env")
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--n-contexts", type=int, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {}
    if args.total_timesteps is not None:
        overrides["total_timesteps"] = args.total_timesteps
    if args.n_envs is not None:
        overrides["n_envs"] = args.n_envs
    if args.eval_freq is not None:
        overrides["eval_freq"] = args.eval_freq
    if args.n_contexts is not None:
        overrides["n_contexts"] = args.n_contexts
    if args.output_root is not None:
        overrides["output_root"] = args.output_root
    if args.device is not None:
        overrides["device"] = args.device
    if args.n_steps is not None:
        overrides["ppo"] = {"n_steps": args.n_steps}

    config = load_config(
        env_name=args.env,
        handling_policy=args.handling_policy,
        seed=args.seed,
        **overrides,
    )

    print(f"Running {config.run_name} -> {config.output_dir}")
    output_dir = run_training(config)
    print(f"Done. Logs written to {output_dir}")


if __name__ == "__main__":
    main()
