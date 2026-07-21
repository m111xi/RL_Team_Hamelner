#!/usr/bin/env python
"""One-stop analysis after training: writes the summary CSVs and every
standard plot (learning curves, final performance, hard-context recovery,
signal validity, curriculum weight evolution) for each environment found
under --results-root, into results/<env>/plots/.

Run this once after scripts/run_all.py finishes (or any time in between - it
only reads whatever runs already exist).

Example:
    python scripts/analyze_results.py --results-root results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.aggregate_results import aggregate, group_across_seeds  # noqa: E402
from src.analysis.plots import (  # noqa: E402
    plot_curriculum_weights,
    plot_final_performance_bars,
    plot_hard_context_recovery,
    plot_learning_curves,
    plot_signal_validity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=str, default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)

    df = aggregate(root)
    if df.empty:
        print(f"No runs found under {root}")
        return

    df.to_csv(root / "summary.csv", index=False)
    grouped = group_across_seeds(df)
    grouped.to_csv(root / "summary_grouped.csv", index=False)
    print(f"Wrote {len(df)} run summaries to {root / 'summary.csv'}")
    print(f"Wrote {len(grouped)} grouped summaries to {root / 'summary_grouped.csv'}")

    for env in sorted(df["env"].unique()):
        env_df = df[df["env"] == env]
        policies = sorted(env_df["handling_policy"].unique())
        seeds = sorted(env_df["seed"].unique())
        plots_dir = root / env / "plots"
        print(f"\n{env}: {len(env_df)} runs, policies={policies}, seeds={seeds}")

        plot_learning_curves(root, env, policies, seeds, plots_dir / "learning_curves.png")
        plot_final_performance_bars(df, env, plots_dir / "final_performance.png")
        plot_hard_context_recovery(root, env, policies, seeds, plots_dir / "hard_context_recovery.png")
        print(f"  wrote learning_curves.png, final_performance.png, hard_context_recovery.png")

        first_seed = seeds[0]
        for policy in policies:
            run_dir = root / env / policy / f"seed{first_seed}"
            if (run_dir / "curriculum_weights.csv").exists():
                plot_curriculum_weights(run_dir, plots_dir / f"weights_{policy}_seed{first_seed}.png")
        print(f"  wrote weights_<policy>_seed{first_seed}.png for each policy")

        if "ignore" in policies:
            for seed in seeds:
                run_dir = root / env / "ignore" / f"seed{seed}"
                if (run_dir / "signal_state.csv").exists():
                    plot_signal_validity(run_dir, plots_dir / f"signal_validity_seed{seed}.png")
            print(f"  wrote signal_validity_seed<seed>.png from the 'ignore' runs")
        else:
            print("  skipped signal_validity plot: no 'ignore' policy run found for this env")

    print(f"\nAll plots written under {root}/<env>/plots/")


if __name__ == "__main__":
    main()
