"""Sanity-check analysis from the proposal's Section 5: does the fixed
error-based difficulty signal (`ema_signal`) actually correlate with a
context's real, empirical difficulty (its final achieved return)?

Best computed on an `ignore`-policy run: since `ignore` never reacts to the
signal, every context gets unbiased, uniform exposure, so its measured
signal/return values aren't distorted by the handling policy itself (e.g. a
rarely-sampled context under `hard_exclude` would have noisier, staler
estimates for reasons unrelated to true difficulty).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_signal_and_eval(run_dir: str | Path) -> pd.DataFrame:
    """Per-context table of the final EMA error signal vs. final empirical
    (deterministic eval) return, for one run."""
    run_dir = Path(run_dir)
    signal_path = run_dir / "signal_state.csv"
    eval_path = run_dir / "eval.csv"
    columns = ["context_id", "ema_signal", "ema_return", "final_empirical_return"]
    if not signal_path.exists() or not eval_path.exists():
        return pd.DataFrame(columns=columns)

    signal_df = pd.read_csv(signal_path)
    eval_df = pd.read_csv(eval_path)
    if signal_df.empty or eval_df.empty:
        return pd.DataFrame(columns=columns)

    last_signal_ts = signal_df["timestep"].max()
    final_signal = signal_df[signal_df["timestep"] == last_signal_ts][["context_id", "ema_signal", "ema_return"]]

    last_eval_ts = eval_df["timestep"].max()
    final_eval = eval_df[eval_df["timestep"] == last_eval_ts][["context_id", "mean_return"]].rename(
        columns={"mean_return": "final_empirical_return"}
    )

    return final_signal.merge(final_eval, on="context_id", how="inner")


def compute_signal_validity(run_dir: str | Path) -> dict:
    """Spearman correlation between the error signal and empirical return.

    A good difficulty proxy should correlate *negatively*: high error <->
    low (bad) empirical return. Returns NaN if there's not enough data.
    """
    merged = load_signal_and_eval(run_dir)
    if len(merged) < 3:
        return {"n_contexts": len(merged), "signal_return_spearman": float("nan")}

    corr = merged[["ema_signal", "final_empirical_return"]].corr(method="spearman").iloc[0, 1]
    return {"n_contexts": len(merged), "signal_return_spearman": float(corr)}
