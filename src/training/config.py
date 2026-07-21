"""Experiment configuration: YAML defaults per env + CLI overrides."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@dataclass
class PPOParams:
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5


@dataclass
class SignalParams:
    """Params for the one fixed difficulty signal, shared by every policy."""

    ema_alpha: float = 0.3
    hard_std_factor: float = 1.0
    patience: int = 3


@dataclass
class PolicyParams:
    """Params for the handling policies. Each policy only uses the subset it needs:
    hard_exclude -> floor; downweight_revisit -> downweight_factor, cooldown;
    replay_priority -> temperature, uniform_mix; ignore -> none."""

    floor: float = 0.02
    downweight_factor: float = 0.3
    cooldown: int = 8
    temperature: float = 1.0
    uniform_mix: float = 0.05


@dataclass
class ExperimentConfig:
    env_name: str
    handling_policy: str
    seed: int = 0
    total_timesteps: int = 200_000
    n_envs: int = 8
    n_contexts: int = 50
    eval_freq: int = 20_000
    eval_episodes_per_context: int = 3
    output_root: str = "results"
    device: str = "cpu"
    ppo: PPOParams = field(default_factory=PPOParams)
    signal_params: SignalParams = field(default_factory=SignalParams)
    policy_params: PolicyParams = field(default_factory=PolicyParams)

    @property
    def run_name(self) -> str:
        return f"{self.env_name}/{self.handling_policy}/seed{self.seed}"

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root) / self.env_name / self.handling_policy / f"seed{self.seed}"


def load_config(env_name: str, handling_policy: str, seed: int = 0, **overrides) -> ExperimentConfig:
    yaml_path = CONFIG_DIR / f"{env_name}.yaml"
    raw: dict = {}
    if yaml_path.exists():
        with open(yaml_path) as f:
            raw = yaml.safe_load(f) or {}

    ppo = PPOParams(**{**raw.get("ppo", {}), **overrides.pop("ppo", {})})
    signal = SignalParams(**{**raw.get("signal_params", {}), **overrides.pop("signal_params", {})})
    policy = PolicyParams(**{**raw.get("policy_params", {}), **overrides.pop("policy_params", {})})

    base = {k: v for k, v in raw.items() if k not in ("ppo", "signal_params", "policy_params")}
    base.update(overrides)

    return ExperimentConfig(
        env_name=env_name,
        handling_policy=handling_policy,
        seed=seed,
        ppo=ppo,
        signal_params=signal,
        policy_params=policy,
        **base,
    )
