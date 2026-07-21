#!/usr/bin/env python
"""Sweep the full experimental grid:

    {cartpole [, other envs]} x {ignore, hard_exclude, downweight_revisit, replay_priority} x seeds

CartPole is the main testbed (default). LunarLander is the proposal's
optional "second environment only if time permits" robustness check.
Pendulum/Acrobot/MountainCar/MountainCarContinuous/BipedalWalker are bonus
environments, not required by the proposal, added purely for broader
robustness-check coverage - all opt-in via --envs. Runs sequentially,
in-process (so failures in one run don't lose the others - each is wrapped in
a try/except and logged). Use --dry-run to just print the grid without
training anything.

Example:
    python scripts/run_all.py --seeds 0 1 2
    python scripts/run_all.py --envs cartpole lunarlander --seeds 0 1 2
    python scripts/run_all.py --envs cartpole pendulum acrobot mountaincar mountaincarcontinuous --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.config import load_config  # noqa: E402
from src.training.train import run_training  # noqa: E402

POLICIES = ["ignore", "hard_exclude", "downweight_revisit", "replay_priority"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--envs",
        nargs="+",
        default=["cartpole"],
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
    parser.add_argument("--handling-policies", nargs="+", default=POLICIES, choices=POLICIES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = [
        (env, policy, seed)
        for env in args.envs
        for policy in args.handling_policies
        for seed in args.seeds
    ]
    print(f"Grid has {len(grid)} runs.")

    overrides = {}
    if args.total_timesteps is not None:
        overrides["total_timesteps"] = args.total_timesteps
    if args.output_root is not None:
        overrides["output_root"] = args.output_root

    failures = []
    for i, (env, policy, seed) in enumerate(grid, start=1):
        print(f"[{i}/{len(grid)}] {env} / {policy} / seed={seed}")
        if args.dry_run:
            continue
        config = load_config(env_name=env, handling_policy=policy, seed=seed, **overrides)
        try:
            run_training(config)
        except Exception:
            print(f"  FAILED: {env}/{policy}/seed{seed}")
            traceback.print_exc()
            failures.append((env, policy, seed))

    if failures:
        print(f"\n{len(failures)} run(s) failed:")
        for f in failures:
            print(" ", f)
    else:
        print("\nAll runs completed successfully.")


if __name__ == "__main__":
    main()
