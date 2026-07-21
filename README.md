# Handling Too-Hard Contexts in Automatic Curriculum Learning

Member: Maximilian Schwingel
Member: Connor Gröling
Member: Laurenz von Schilgen

This repository contains the code, experiments, and results for the RL Exam project (see
[`RL_Project_Proposal_Team_Hamelner.pdf`](RL_Project_Proposal_Team_Hamelner.pdf)).

## Project Overview

Automatic curriculum learning adapts training to the agent's current ability instead of
following a fixed manual schedule. In contextual reinforcement learning this matters most
when some contexts are temporarily too hard: naive uniform sampling keeps hammering the
agent with contexts it currently can't solve. This project's question is not *how to
detect* difficult contexts, but **how a curriculum should react once a context is
identified as too hard**.

We fix a single difficulty signal (an error-based signal derived from PPO's own
value-prediction error) and compare four **handling policies** built on the same uniform
sampling backbone:

- **Ignore** — keep sampling uniformly, never react to the signal.
- **Hard exclude** — temporarily pause hard contexts until their score improves.
- **Down-weight + revisit** — reduce their sampling probability, but force a
  periodic revisit regardless of measured improvement.
- **Replay-style prioritization** — continuously adapt sampling probability to
  the raw signal magnitude (PLR-style), without ever gating on a hard/not-hard flag.

Everything else — agent, optimizer, training budget, seeds, and the difficulty signal
itself — is held fixed. The only manipulated factor is the handling policy, which isolates
its effect instead of comparing whole curriculum frameworks. Metrics: episodic return,
sample efficiency, robustness across seeds, and (the new, central one) **recovery of hard
contexts over time** — does a context that gets flagged hard ever get un-flagged, and how
long does that take, under each policy?

## How it works

- **Environments**: [CARL](https://github.com/automl/CARL) (`carl-bench`) provides
  contextual versions of classic control tasks. **CARLCartPole** (varying pole `length`
  and `masspole`) is the main testbed. **CARLLunarLander** (varying `GRAVITY_Y` and
  `MAIN_ENGINE_POWER`) is available as a second environment for a robustness check, used
  only if time permits, per the proposal.

  Beyond what the proposal asks for, five more environments are wired up for broader
  robustness-check coverage (all in `src/envs/contexts.py`'s `ENV_SPECS`, all usable via
  `--env <name>` exactly like CartPole/LunarLander):
  - **CARLPendulum** (`m`, `l`) — continuous control, dense reward, works natively
    everywhere without Box2D.
  - **CARLAcrobot** (`LINK_LENGTH_1`, `LINK_MASS_1`) — discrete control, sparse reward
    (-1/step until swing-up) - a different reward profile than CartPole/Pendulum.
  - **CARLMountainCar** / **CARLMountainCarContinuous** (`force`/`power`, plus
    `gravity`/`goal_position`) — notoriously hard for vanilla PPO without reward shaping;
    kept anyway for diversity, but don't be surprised if every handling policy struggles
    roughly equally here. Their configs use a higher `eval_freq` than the other envs,
    since episodes basically always run the full 200-step cap (no early termination
    without solving it), making periodic eval disproportionately expensive otherwise.
  - **CARLBipedalWalker** (`MOTORS_TORQUE`, `FRICTION`) — continuous 4D-action locomotion,
    richer/harder than LunarLander, same Box2D dependency.

  (**CARLVehicleRacing** exists in CARL too but was left out: it's pixel-observation-based
  (96×96×3 images), which would need a CNN policy instead of a plain MLP - out of scope for
  this project's size.)

  > LunarLander and BipedalWalker need a working `Box2D` physics binding. On Python ≤ 3.13,
  > `pip install Box2D` gets you a prebuilt wheel with no build tools required. On Python
  > 3.14+ (e.g. a fresh Ubuntu's default `python3`), that wheel doesn't exist yet, and
  > you'd need `carl-bench`'s `box2d` extra instead, which builds from source via SWIG —
  > this fails on native Windows but works under WSL. See **Setup** below.
  >
  > Separately: gymnasium 0.29.1's Acrobot/MountainCar implementations reference a
  > `np.float_` alias that NumPy 2.0 removed; `src/envs/__init__.py` restores it as a
  > small, well-known compatibility shim (not a bug in this project's own code).

- **Difficulty signal** (`src/curriculum/signal.py`, `DifficultySignal`): one fixed,
  error-based signal used everywhere — `|V(s) − return|` from PPO's own just-collected
  rollout buffer, aggregated per context and smoothed with an EMA. A context is classified
  **hard** once its EMA return sits `hard_std_factor` standard deviations below the
  population mean for `patience` consecutive rollouts, and **recovered** once it's back
  above that line for `patience` consecutive rollouts (symmetric, to avoid flapping). This
  measurement is identical across all four policies — only the reaction differs.

- **Handling policies** (`src/curriculum/policies.py`), all built on a uniform sampling
  backbone:
  - `IgnorePolicy` — always uniform.
  - `HardExcludePolicy` — uniform among not-hard contexts; hard contexts get a small fixed
    `floor` weight (not exactly zero — a context can only stop being "hard" if its return
    improves, which needs occasional sampling; whether that trickle is enough to ever
    recover is itself part of the result).
  - `DownweightRevisitPolicy` — hard contexts multiplied by `downweight_factor`; after
    `cooldown` rollouts, forced back to full weight for one round regardless of whether the
    signal says it recovered.
  - `ReplayPriorityPolicy` — blends the uniform backbone with a softmax over the
    continuous EMA signal; never gates on the binary hard/not-hard flag, reacts to
    magnitude only.

  `src/curriculum/controller.py`'s `CurriculumController` owns one `DifficultySignal` +
  one policy, and is the single shared object every parallel env's context selector reads
  sampling weights from.

- **Context visibility**: the agent is always given the context explicitly (concatenated
  into the observation, `src/envs/wrappers.py::FlattenContextObsWrapper`) — this project
  no longer varies context visibility as an independent axis (that was part of an earlier,
  broader draft of the proposal).

- **PPO**: via Stable-Baselines3 (`MlpPolicy`), one `DummyVecEnv` per run so every parallel
  env can share the same in-process `CurriculumController`.

## Setup

```bash
pip install -r requirements.txt
```

This is enough for `cartpole` and `pendulum`. For `lunarlander`, you additionally need
`Box2D` — try this first, it's a plain wheel install on Python ≤ 3.13:

```bash
pip install Box2D
```

If that fails to find a matching wheel for your Python version (3.14+), you'll need the
`box2d-py`-via-SWIG route instead, which in practice means WSL — see below.

### WSL setup (only needed for LunarLander on Python 3.14+, e.g. a fresh Ubuntu)

Native Windows can't build `box2d-py`. Under WSL2 + Ubuntu it builds cleanly:

1. Install WSL if you haven't: open PowerShell **as Administrator** and run
   `wsl --install -d Ubuntu`, then reboot when prompted and finish the one-time Ubuntu user
   setup.
2. Inside the Ubuntu terminal, install the system packages needed to build `box2d-py`
   (and `pygame`, one of its dependencies) from source:
   ```bash
   sudo apt update && sudo apt install -y build-essential swig python3-venv python3-pip \
       libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
       libfreetype6-dev libportmidi-dev libjpeg-dev pkg-config python3-dev
   ```
3. Copy (or `git clone`) the project into the Linux filesystem rather than working on it
   through `/mnt/c/...` — it's noticeably faster:
   ```bash
   rsync -a --exclude='.git' --exclude='results' /mnt/c/Users/<you>/path/to/RL_Team_Hamelner/ ~/RL_Team_Hamelner/
   cd ~/RL_Team_Hamelner
   ```
4. Create a venv and install torch's **CPU-only** build first (installing `torch` the
   normal way pulls several GB of CUDA/NVIDIA packages that this project doesn't need for
   CartPole/Pendulum/LunarLander-scale training), then the rest:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install --index-url https://download.pytorch.org/whl/cpu torch
   pip install -r requirements.txt
   pip install "carl-bench[box2d]"
   ```
5. Run everything (`pytest`, `scripts/run_experiment.py`, ...) from inside that WSL shell
   exactly as described below — `--env lunarlander` now works.

## Running experiments

Single run:

```bash
python scripts/run_experiment.py --env cartpole --handling-policy downweight_revisit --seed 0
```

Quick smoke test (few thousand steps, to check the pipeline runs end-to-end):

```bash
python scripts/run_experiment.py --env cartpole --handling-policy downweight_revisit \
    --seed 0 --total-timesteps 6000 --eval-freq 2000 --n-envs 4 --n-steps 128
```

Full experimental grid (`cartpole x {ignore, hard_exclude, downweight_revisit,
replay_priority} x seeds`; every other environment is opt-in via `--envs` - LunarLander
per the proposal's "second environment only if time permits", the rest as bonus robustness
checks):

```bash
python scripts/run_all.py --seeds 0 1 2
python scripts/run_all.py --envs cartpole lunarlander --seeds 0 1 2
python scripts/run_all.py --envs cartpole pendulum acrobot mountaincar mountaincarcontinuous bipedalwalker --seeds 0 1 2
# or just print the grid without training:
python scripts/run_all.py --dry-run
```

Per-env defaults (timesteps, PPO hyperparameters, signal/policy hyperparameters) live in
`configs/<env>.yaml` for each of `cartpole`, `lunarlander`, `pendulum`, `acrobot`,
`mountaincar`, `mountaincarcontinuous`, `bipedalwalker`; both `run_experiment.py` and
`run_all.py` accept CLI overrides on top of them.

Each run writes to `results/<env>/<handling_policy>/seed<seed>/`: `episodes.csv`
(per-episode return/length/context), `curriculum_weights.csv` (sampling weight per context
over time), `hard_events.csv` (every `flagged_hard`/`recovered` transition — the data
behind the recovery metric), `signal_state.csv` (the raw `ema_signal`/`ema_return`/
`is_hard` per context over time — the data behind the signal-validity sanity check),
`eval.csv` (periodic deterministic per-context return), `config.json` (the resolved
config), `model.zip`, and a `tb/` TensorBoard log.

## Analyzing results

One command, run any time (during or after training - it just reads whatever runs already
exist):

```bash
python scripts/analyze_results.py --results-root results
```

Writes `results/summary.csv` (one row per run: final/AUC/eval return,
`n_flagged`/`n_recovered`/`fraction_recovered`/`mean_time_to_recovery` from
`hard_events.csv`, and `signal_return_spearman` from `signal_state.csv` + `eval.csv`) and
`results/summary_grouped.csv` (mean/std across seeds, grouped by env/handling_policy) -
plus, for every environment found, a full set of PNGs under `results/<env>/plots/`:
`learning_curves.png`, `final_performance.png`, `hard_context_recovery.png` (fraction of
ever-flagged-hard contexts that recovered, per policy — the key comparison the proposal
asks for), `weights_<policy>_seed<seed>.png` (curriculum weight evolution, one per
policy), and — if an `ignore`-policy run is present — `signal_validity_seed<seed>.png`
(the proposal's Section 5 sanity check: does the fixed error signal actually correlate
with a context's real, empirical difficulty? Best read from `ignore`, since it never
reacts to the signal, so every context gets unbiased exposure).

For anything more custom, `src/analysis/plots.py` exposes each plotting function
individually (`plot_learning_curves`, `plot_final_performance_bars`,
`plot_curriculum_weights`, `plot_hard_context_recovery`, `plot_signal_validity`), and
`src/analysis/aggregate_results.py`/`signal_validity.py` expose the underlying data-loading
functions to build on top of.

While a run is in progress, you can also watch it live via TensorBoard (every run writes to
`results/<env>/<handling_policy>/seed<seed>/tb/`):

```bash
tensorboard --logdir results
```

## Tests

```bash
pytest
```

Unit tests cover the difficulty signal's hard/recovered classification, each of the four
handling policies individually, the controller, and the observation wrapper, without
needing CARL installed. Integration test suites exercise every environment end-to-end with
the real CARL classes; the LunarLander and BipedalWalker suites (the two that need Box2D)
skip automatically if `Box2D` isn't importable (see **Setup** above).

## Scope note

Full multi-seed training to convergence across the whole grid takes real wall-clock time
and is not run automatically — the pipeline above has been built and smoke-tested
end-to-end (all four policies produce sane, clearly differentiated weight dynamics: e.g.
`hard_exclude` suppresses a flagged context indefinitely while `downweight_revisit` shows
periodic resets back to full weight). Kick off `scripts/run_all.py` when you're ready to
produce the actual experiment results and figures for the report.

Measured (production PPO settings, CPU): ~14 min per CartPole run (300k steps) and ~21 min
per LunarLander run (500k steps) - roughly 35-50% of that is the periodic deterministic
eval, not training itself. The default grid (`cartpole x 4 policies x 3 seeds` = 12 runs)
is therefore ~2.8 hours; adding `lunarlander` roughly doubles it. `run_all.py` runs
sequentially and wraps each run in its own try/except, so one crashing run won't take the
rest of an overnight batch down with it - check the printed failure list at the end. If
running unattended overnight, make sure Windows won't sleep (Settings -> System -> Power,
or just keep the lid open / plugged in).
