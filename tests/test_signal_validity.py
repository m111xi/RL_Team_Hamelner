import csv

import numpy as np

from src.analysis.signal_validity import compute_signal_validity, load_signal_and_eval


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def test_missing_files_return_empty(tmp_path):
    merged = load_signal_and_eval(tmp_path)
    assert merged.empty
    stats = compute_signal_validity(tmp_path)
    assert stats["n_contexts"] == 0
    assert np.isnan(stats["signal_return_spearman"])


def test_only_last_timestep_is_used(tmp_path):
    # signal_state has two timesteps; only the later one should be used
    _write_csv(
        tmp_path / "signal_state.csv",
        ["timestep", "context_id", "ema_signal", "ema_return", "is_hard"],
        [
            [100, 0, 999.0, -999.0, True],  # earlier - should be ignored
            [200, 0, 1.0, 10.0, False],
            [200, 1, 5.0, 2.0, True],
            [200, 2, 3.0, 6.0, False],
        ],
    )
    _write_csv(
        tmp_path / "eval.csv",
        ["timestep", "context_id", "mean_return"],
        [
            [200, 0, 10.0],
            [200, 1, 2.0],
            [200, 2, 6.0],
        ],
    )
    merged = load_signal_and_eval(tmp_path)
    assert len(merged) == 3
    assert set(merged["context_id"]) == {0, 1, 2}
    assert 999.0 not in merged["ema_signal"].values


def test_negative_correlation_detected_when_high_error_means_low_return(tmp_path):
    rows_signal = []
    rows_eval = []
    for cid in range(10):
        error = float(cid)  # increasing error
        ret = float(100 - 10 * cid)  # decreasing return -> perfectly anti-correlated
        rows_signal.append([500, cid, error, ret, error > 5])
        rows_eval.append([500, cid, ret])

    _write_csv(
        tmp_path / "signal_state.csv", ["timestep", "context_id", "ema_signal", "ema_return", "is_hard"], rows_signal
    )
    _write_csv(tmp_path / "eval.csv", ["timestep", "context_id", "mean_return"], rows_eval)

    stats = compute_signal_validity(tmp_path)
    assert stats["n_contexts"] == 10
    assert stats["signal_return_spearman"] < -0.99  # near-perfect monotonic anti-correlation


def test_too_few_contexts_returns_nan(tmp_path):
    _write_csv(
        tmp_path / "signal_state.csv",
        ["timestep", "context_id", "ema_signal", "ema_return", "is_hard"],
        [[100, 0, 1.0, 5.0, False], [100, 1, 2.0, 4.0, False]],
    )
    _write_csv(tmp_path / "eval.csv", ["timestep", "context_id", "mean_return"], [[100, 0, 5.0], [100, 1, 4.0]])
    stats = compute_signal_validity(tmp_path)
    assert np.isnan(stats["signal_return_spearman"])
