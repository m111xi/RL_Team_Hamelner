"""Plain-CSV run logging (episode returns, curriculum weights, hard-context
events, periodic eval, raw signal state)."""
from __future__ import annotations

import csv
from pathlib import Path


class RunLogger:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._ep_file = open(self.output_dir / "episodes.csv", "w", newline="")
        self._ep_writer = csv.writer(self._ep_file)
        self._ep_writer.writerow(["timestep", "context_id", "return", "length"])

        self._weights_file = open(self.output_dir / "curriculum_weights.csv", "w", newline="")
        self._weights_writer = csv.writer(self._weights_file)
        self._weights_writer.writerow(["timestep", "context_id", "weight"])

        self._events_file = open(self.output_dir / "hard_events.csv", "w", newline="")
        self._events_writer = csv.writer(self._events_file)
        self._events_writer.writerow(["timestep", "context_id", "event"])

        self._eval_file = open(self.output_dir / "eval.csv", "w", newline="")
        self._eval_writer = csv.writer(self._eval_file)
        self._eval_writer.writerow(["timestep", "context_id", "mean_return"])

        self._signal_file = open(self.output_dir / "signal_state.csv", "w", newline="")
        self._signal_writer = csv.writer(self._signal_file)
        self._signal_writer.writerow(["timestep", "context_id", "ema_signal", "ema_return", "is_hard"])

    def log_episode(self, timestep: int, context_id: int, ep_return: float, length: int) -> None:
        self._ep_writer.writerow([timestep, context_id, ep_return, length])
        self._ep_file.flush()

    def log_curriculum_weights(self, timestep: int, controller) -> None:
        for cid, w in zip(controller.context_ids, controller.get_weights()):
            self._weights_writer.writerow([timestep, cid, w])
        self._weights_file.flush()

    def log_hard_event(self, timestep: int, context_id: int, event: str) -> None:
        self._events_writer.writerow([timestep, context_id, event])
        self._events_file.flush()

    def log_eval(self, timestep: int, context_id: int, mean_return: float) -> None:
        self._eval_writer.writerow([timestep, context_id, mean_return])
        self._eval_file.flush()

    def log_signal_state(self, timestep: int, controller) -> None:
        signal = controller.signal
        for cid, ema_signal, ema_return, is_hard in zip(
            signal.context_ids, signal.ema_signal, signal.ema_return, signal.is_hard
        ):
            self._signal_writer.writerow([timestep, cid, ema_signal, ema_return, bool(is_hard)])
        self._signal_file.flush()

    def close(self) -> None:
        self._ep_file.close()
        self._weights_file.close()
        self._events_file.close()
        self._eval_file.close()
        self._signal_file.close()
