"""Collates per-run CSV logs under results/ into summary tables.

Usage:
    python -m src.analysis.aggregate_results --results-root results --out results/summary.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.analysis.signal_validity import compute_signal_validity


def load_run(run_dir: Path) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    episodes = pd.read_csv(run_dir / "episodes.csv") if (run_dir / "episodes.csv").exists() else pd.DataFrame()
    eval_df = pd.read_csv(run_dir / "eval.csv") if (run_dir / "eval.csv").exists() else pd.DataFrame()
    weights = (
        pd.read_csv(run_dir / "curriculum_weights.csv")
        if (run_dir / "curriculum_weights.csv").exists()
        else pd.DataFrame()
    )
    events = (
        pd.read_csv(run_dir / "hard_events.csv") if (run_dir / "hard_events.csv").exists() else pd.DataFrame()
    )
    return {"config": config, "episodes": episodes, "eval": eval_df, "weights": weights, "events": events}


def compute_recovery_stats(events: pd.DataFrame) -> dict:
    """Pairs each "flagged_hard" event with the next "recovered" event (if any)
    per context, to summarize how hard contexts behave over training."""
    stats = {
        "n_flagged": 0,
        "n_recovered": 0,
        "fraction_recovered": float("nan"),
        "mean_time_to_recovery": float("nan"),
    }
    if events.empty:
        return stats

    n_flagged = 0
    n_recovered = 0
    recovery_times: list[float] = []
    for _cid, group in events.sort_values("timestep").groupby("context_id"):
        pending_flag_ts = None
        for _, row in group.iterrows():
            if row["event"] == "flagged_hard":
                n_flagged += 1
                pending_flag_ts = row["timestep"]
            elif row["event"] == "recovered" and pending_flag_ts is not None:
                n_recovered += 1
                recovery_times.append(row["timestep"] - pending_flag_ts)
                pending_flag_ts = None

    stats["n_flagged"] = n_flagged
    stats["n_recovered"] = n_recovered
    stats["fraction_recovered"] = n_recovered / n_flagged if n_flagged else float("nan")
    stats["mean_time_to_recovery"] = float(pd.Series(recovery_times).mean()) if recovery_times else float("nan")
    return stats


def summarize_run(run_dir: Path) -> dict:
    data = load_run(run_dir)
    config = data["config"]
    episodes = data["episodes"]
    eval_df = data["eval"]

    summary = {
        "env": config["env_name"],
        "handling_policy": config["handling_policy"],
        "seed": config["seed"],
        "n_episodes": len(episodes),
        "final_return": float("nan"),
        "auc_return": float("nan"),
        "eval_return": float("nan"),
    }

    if len(episodes):
        cutoff = episodes["timestep"].quantile(0.9)
        summary["final_return"] = float(episodes.loc[episodes["timestep"] >= cutoff, "return"].mean())
        summary["auc_return"] = float(episodes["return"].mean())

    if len(eval_df):
        last_ts = eval_df["timestep"].max()
        summary["eval_return"] = float(eval_df.loc[eval_df["timestep"] == last_ts, "mean_return"].mean())

    summary.update(compute_recovery_stats(data["events"]))
    summary.update(compute_signal_validity(run_dir))
    return summary


def aggregate(results_root: str | Path = "results") -> pd.DataFrame:
    root = Path(results_root)
    run_dirs = sorted({p.parent for p in root.glob("*/*/*/config.json")})
    rows = [summarize_run(d) for d in run_dirs]
    return pd.DataFrame(rows)


def group_across_seeds(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    metrics = [c for c in df.columns if c not in ("env", "handling_policy", "seed")]
    grouped = df.groupby(["env", "handling_policy"])[metrics].agg(["mean", "std"])
    grouped.columns = ["_".join(c) for c in grouped.columns]
    return grouped.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--out", type=str, default="results/summary.csv")
    args = parser.parse_args()

    df = aggregate(args.results_root)
    if df.empty:
        print(f"No runs found under {args.results_root}")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} run summaries to {out_path}")

    grouped = group_across_seeds(df)
    grouped_path = out_path.with_name(out_path.stem + "_grouped.csv")
    grouped.to_csv(grouped_path, index=False)
    print(f"Wrote {len(grouped)} grouped summaries to {grouped_path}")


if __name__ == "__main__":
    main()
