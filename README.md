# Automatic Curriculum Learning for Contextual Reinforcement Learning
Member: Maximilian Schwingel
Member: Connor Gröling
Member: Laurenz von Schilgen

Automatic curriculum learning for contextual reinforcement learning, studying difficulty balancing and generalization across CARL environments with PPO-based agents.

This repository contains the code, experiments, and results for the RL Exam project. The project investigates how automatic context curricula can improve training and generalization in contextual reinforcement learning settings, with a focus on balancing task difficulty across CARL benchmark environments.

Project Overview
Contextual reinforcement learning models tasks as contextual MDPs, where environment dynamics or rewards vary across contexts.
This project studies whether automatically selecting contexts during training can improve generalization compared to static or uniformly sampled curricula, especially on interpretable benchmark environments such as CARL.

The main focus is on PPO-based agents and curriculum strategies that adapt the training distribution over contexts according to signals such as performance, difficulty, or uncertainty.
The goal is to evaluate how these choices affect training efficiency, robustness, and transfer to unseen contexts.

Goals
Study automatic curriculum learning in contextual reinforcement learning.

Analyze difficulty balancing across multiple CARL environments.

Compare curriculum strategies for PPO-based agents.

Evaluate in-distribution and out-of-distribution generalization across contexts.

Scope
This repository is intended for the RL Exam and serves as the central codebase for implementation, experiments, analysis, and documentation.
