"""Plotting helpers for learning curves, final performance, curriculum weight
evolution, and hard-context recovery. Kept dependency-light (matplotlib +
pandas only).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.aggregate_results import compute_recovery_stats, load_run
from src.analysis.signal_validity import compute_signal_validity, load_signal_and_eval


def _bin_episode_returns(episodes: pd.DataFrame, n_bins: int = 50) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame(columns=["bin_center", "mean", "std"])
    bins = np.linspace(episodes["timestep"].min(), episodes["timestep"].max(), n_bins + 1)
    episodes = episodes.copy()
    episodes["bin"] = pd.cut(episodes["timestep"], bins=bins, include_lowest=True)
    grouped = episodes.groupby("bin", observed=True)["return"].agg(["mean", "std", "count"])
    grouped = grouped[grouped["count"] > 0]
    centers = [interval.mid for interval in grouped.index]
    return pd.DataFrame({"bin_center": centers, "mean": grouped["mean"].values, "std": grouped["std"].values})


def plot_learning_curves(
    results_root: str | Path,
    env: str,
    handling_policies: list[str],
    seeds: list[int],
    out_path: str | Path,
    n_bins: int = 50,
) -> None:
    root = Path(results_root)
    fig, ax = plt.subplots(figsize=(8, 5))

    for policy in handling_policies:
        all_binned = []
        for seed in seeds:
            run_dir = root / env / policy / f"seed{seed}"
            if not (run_dir / "episodes.csv").exists():
                continue
            episodes = pd.read_csv(run_dir / "episodes.csv")
            all_binned.append(_bin_episode_returns(episodes, n_bins=n_bins))

        if not all_binned:
            continue

        merged = pd.concat(all_binned).groupby("bin_center")["mean"].agg(["mean", "std"])
        ax.plot(merged.index, merged["mean"], label=policy)
        ax.fill_between(
            merged.index,
            merged["mean"] - merged["std"].fillna(0),
            merged["mean"] + merged["std"].fillna(0),
            alpha=0.2,
        )

    ax.set_xlabel("timestep")
    ax.set_ylabel("episodic return")
    ax.set_title(f"{env} - learning curves across seeds, by handling policy")
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_final_performance_bars(summary_df: pd.DataFrame, env: str, out_path: str | Path) -> None:
    subset = summary_df[summary_df["env"] == env]
    if subset.empty:
        return

    grouped = subset.groupby("handling_policy")[["final_return", "eval_return"]].agg(["mean", "std"])
    policies = list(grouped.index)
    metrics = ["final_return", "eval_return"]

    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.35
    x = np.arange(len(policies))
    for i, metric in enumerate(metrics):
        means = grouped[(metric, "mean")].values
        stds = grouped[(metric, "std")].fillna(0).values
        ax.bar(x + i * width, means, width, yerr=stds, label=metric, capsize=3)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(policies)
    ax.set_ylabel("mean return")
    ax.set_title(f"{env} - final performance by handling policy")
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_curriculum_weights(run_dir: str | Path, out_path: str | Path, top_k: int = 10) -> None:
    run_dir = Path(run_dir)
    weights = pd.read_csv(run_dir / "curriculum_weights.csv")
    if weights.empty:
        return

    pivot = weights.pivot_table(index="timestep", columns="context_id", values="weight")
    # keep only the contexts with the highest weight variance (most interesting to look at)
    variances = pivot.var().sort_values(ascending=False)
    keep_cols = variances.head(top_k).index
    pivot = pivot[keep_cols]

    fig, ax = plt.subplots(figsize=(8, 5))
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], label=f"ctx {col}", alpha=0.8)
    ax.set_xlabel("timestep")
    ax.set_ylabel("sampling weight")
    ax.set_title(f"curriculum weight evolution ({run_dir.parent.name}/{run_dir.name})")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hard_context_recovery(
    results_root: str | Path,
    env: str,
    handling_policies: list[str],
    seeds: list[int],
    out_path: str | Path,
) -> None:
    """Bar chart of what fraction of ever-flagged-hard contexts recovered by
    the end of training, per handling policy (averaged across seeds)."""
    root = Path(results_root)
    fractions_by_policy: dict[str, list[float]] = {}

    for policy in handling_policies:
        fractions = []
        for seed in seeds:
            run_dir = root / env / policy / f"seed{seed}"
            events_path = run_dir / "hard_events.csv"
            if not events_path.exists():
                continue
            events = pd.read_csv(events_path)
            stats = compute_recovery_stats(events)
            if stats["n_flagged"] > 0:
                fractions.append(stats["fraction_recovered"])
        if fractions:
            fractions_by_policy[policy] = fractions

    if not fractions_by_policy:
        return

    policies = list(fractions_by_policy.keys())
    means = [float(np.mean(fractions_by_policy[p])) for p in policies]
    stds = [float(np.std(fractions_by_policy[p])) for p in policies]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(policies))
    ax.bar(x, means, yerr=stds, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.set_ylabel("fraction of hard contexts recovered")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{env} - hard-context recovery by handling policy")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_signal_validity(run_dir: str | Path, out_path: str | Path) -> None:
    """Scatter of the error signal vs. final empirical return, per context -
    the proposal's sanity check for whether the fixed difficulty signal is a
    good proxy for real difficulty. Best run on an `ignore`-policy run (see
    `src/analysis/signal_validity.py` docstring for why)."""
    run_dir = Path(run_dir)
    merged = load_signal_and_eval(run_dir)
    if merged.empty:
        return
    stats = compute_signal_validity(run_dir)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(merged["final_empirical_return"], merged["ema_signal"], alpha=0.7)
    ax.set_xlabel("final empirical return (higher = easier)")
    ax.set_ylabel("EMA error signal")
    corr = stats["signal_return_spearman"]
    corr_label = f"{corr:.2f}" if corr == corr else "n/a"  # NaN check
    ax.set_title(
        f"Signal validity ({run_dir.parent.name}/{run_dir.name})\n"
        f"Spearman r = {corr_label} (negative = error tracks difficulty, as expected)"
    )
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


__all__ = [
    "plot_learning_curves",
    "plot_final_performance_bars",
    "plot_curriculum_weights",
    "plot_hard_context_recovery",
    "plot_signal_validity",
    "load_run",
]
