"""Builds env + PPO model for one experiment config and runs training."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList

from src.curriculum.controller import CurriculumController
from src.envs.contexts import build_context_set
from src.envs.factory import make_vec_env
from src.training.callbacks import CurriculumUpdateCallback, PeriodicEvalCallback
from src.training.config import ExperimentConfig
from src.utils.logging import RunLogger


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_training(config: ExperimentConfig) -> Path:
    set_global_seed(config.seed)

    contexts = build_context_set(config.env_name, seed=config.seed, n_contexts=config.n_contexts)

    controller = CurriculumController(
        context_ids=list(contexts.keys()),
        handling_policy=config.handling_policy,
        ema_alpha=config.signal_params.ema_alpha,
        hard_std_factor=config.signal_params.hard_std_factor,
        patience=config.signal_params.patience,
        floor=config.policy_params.floor,
        downweight_factor=config.policy_params.downweight_factor,
        cooldown=config.policy_params.cooldown,
        temperature=config.policy_params.temperature,
        uniform_mix=config.policy_params.uniform_mix,
    )

    vec_env = make_vec_env(config.env_name, contexts, controller, config.n_envs, config.seed)

    output_dir = config.output_dir
    run_logger = RunLogger(output_dir)

    ppo_kwargs = dict(
        learning_rate=config.ppo.learning_rate,
        n_steps=config.ppo.n_steps,
        batch_size=config.ppo.batch_size,
        n_epochs=config.ppo.n_epochs,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
        clip_range=config.ppo.clip_range,
        ent_coef=config.ppo.ent_coef,
        vf_coef=config.ppo.vf_coef,
    )
    model = PPO(
        "MlpPolicy",
        vec_env,
        seed=config.seed,
        device=config.device,
        tensorboard_log=str(output_dir / "tb"),
        verbose=0,
        **ppo_kwargs,
    )

    curriculum_cb = CurriculumUpdateCallback(controller=controller, run_logger=run_logger)
    eval_cb = PeriodicEvalCallback(
        env_name=config.env_name,
        contexts=contexts,
        eval_freq=config.eval_freq,
        episodes_per_context=config.eval_episodes_per_context,
        run_logger=run_logger,
        seed=config.seed,
    )

    with open(output_dir / "config.json", "w") as f:
        json.dump(_config_to_dict(config), f, indent=2)

    try:
        model.learn(total_timesteps=config.total_timesteps, callback=CallbackList([curriculum_cb, eval_cb]))
        model.save(str(output_dir / "model.zip"))
    finally:
        run_logger.close()

    return output_dir


def _config_to_dict(config: ExperimentConfig) -> dict:
    return {
        "env_name": config.env_name,
        "handling_policy": config.handling_policy,
        "seed": config.seed,
        "total_timesteps": config.total_timesteps,
        "n_envs": config.n_envs,
        "n_contexts": config.n_contexts,
        "eval_freq": config.eval_freq,
        "eval_episodes_per_context": config.eval_episodes_per_context,
        "ppo": vars(config.ppo),
        "signal_params": vars(config.signal_params),
        "policy_params": vars(config.policy_params),
    }
