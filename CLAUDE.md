# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HDMI (Humanoid Dynamic Motion Imitation) is a framework for training humanoid robots to perform whole-body interaction tasks from human demonstration videos. The codebase implements a two-stage PPO-ROA (Proximal Policy Optimization with Regularized Online Adaptation) training pipeline using MuJoCo physics simulation.

## Installation

**Standard installation:**
```bash
# Install mjlab (required dependency - fork of IsaacLab for MuJoCo)
git clone git@github.com:EGalahad/mjlab.git
cd mjlab
pip install -e .

# Install HDMI
cd ..
git clone https://github.com/EGalahad/hdmi
cd hdmi
git checkout mjlab
pip install -e .
```

**Alternative (uv):**
```bash
uv sync
```

## Training Commands

The codebase uses Hydra for configuration management. All configs are in `cfg/`.

**Teacher policy (Stage 1):**
```bash
python scripts/train.py algo=ppo_roa_train task=G1/hdmi/move_suitcase

# Or with uv:
uv run scripts/train.py algo=ppo_roa_train task=G1/hdmi/move_suitcase
```

**Student policy (Stage 2 - fine-tuning):**
```bash
python scripts/train.py algo=ppo_roa_finetune task=G1/hdmi/move_suitcase checkpoint_path=run:<teacher_wandb-run-path>
```

**Evaluation:**
```bash
python scripts/play.py algo=ppo_roa_train task=G1/hdmi/move_suitcase checkpoint_path=run:<wandb-run-path>

# Export policy for deployment:
python scripts/play.py algo=ppo_roa_train task=G1/hdmi/move_suitcase checkpoint_path=run:<wandb-run-path> export_policy=true
```

**Sequential training (both stages):**
```bash
python scripts/train_sequential.py task=G1/hdmi/move_suitcase
```

## Architecture

### Environment System (`active_adaptation/envs/`)

- **Base classes**: `base.py` defines the core `_Env` class using TorchRL's `EnvBase`
- **MDP components** (`envs/mdp/`):
  - `action.py`: Action managers (e.g., `JointPosition` for joint position control)
  - `observations/`: Observation functions organized by type (common, priv_body, amp)
  - `commands/`: Command managers, including HDMI-specific tracking commands
  - `rewards/`: Reward functions (feet contact rewards, tracking rewards)
  - `terminations.py`: Episode termination conditions
  - `randomizations.py`: Domain randomization utilities

- **Task hierarchy**:
  - `humanoid.py`: Base humanoid environment
  - `locomotion.py`: Locomotion-specific environments
  - Tasks configured via YAML files in `cfg/task/G1/hdmi/` and `cfg/task/G1/tracking/`

### Learning System (`active_adaptation/learning/`)

- **PPO variants** (`learning/ppo/`):
  - `ppo_roa.py`: Main PPO-ROA implementation for teacher-student training
  - `ppo_amp.py`: PPO with Adversarial Motion Priors
  - `critics.py`: Value function critics
  - `common.py`: Shared PPO utilities

- **Modules** (`learning/modules/`):
  - `rnn.py`: Recurrent modules for adaptation
  - `temporal.py`: Temporal processing modules
  - `distributions.py`: Custom probability distributions
  - `ensemble.py`: Ensemble networks
  - `evidential.py`: Evidential networks for uncertainty

- **Utils** (`learning/utils/`):
  - `gae.py`: Generalized Advantage Estimation
  - `valuenorm.py`: Value normalization
  - `vecnorm.py`: Vector normalization
  - `replay_buffer.py`: Experience replay buffers

### Motion Data System

- **Motion datasets**: Located in `data/motion/` (HDMI format)
- **Retargeting**: Scripts in `data/retarget/` convert motion capture data to HDMI format
- **Motion utilities**: `active_adaptation/utils/motion.py` provides `MotionDataset` and `MotionData` classes

### Key Concepts

**ROA (Regularized Online Adaptation)**: Two-encoder architecture where a "teacher" encoder sees privileged information (e.g., terrain, dynamics) and a "student" encoder learns from observation history. During training, the student is distilled from the teacher. During deployment, only the student is used.

**HDMI Command System**: `RobotTracking` and `RobotObjectTracking` in `envs/mdp/commands/hdmi/command.py` manage reference motion playback and object interaction tracking.

**Observation Groups**: Observations are organized into groups (defined in `envs/base.py::ObsGroup`) that can be computed on-demand and cached for efficiency.

## Configuration Structure

Hydra configs are composed from multiple layers:

- `cfg/train.yaml`: Main training config (imports task + algo)
- `cfg/task/`: Task-specific configs
  - `cfg/task/base/hdmi-base.yaml`: Base HDMI task config
  - `cfg/task/G1/hdmi/*.yaml`: Specific tasks (move_suitcase, open_door, etc.)
- `cfg/algo/`: Algorithm configs (though ppo_roa configs are defined in code via dataclasses)

Algorithm configs use `_target_` to specify the Python class (e.g., `active_adaptation.learning.ppo.ppo_roa.PPOROA`).

## WandB Integration

- Project tracked under `hdmi` project
- Checkpoints saved with WandB run metadata
- Load checkpoints via `checkpoint_path=run:<wandb-run-path>` format
- Configs and policy source code auto-saved to WandB runs

## Asset System

Robot assets defined in `active_adaptation/assets/` with metadata in `get_asset_meta()`. The G1 humanoid supports multiple configurations (e.g., `g1_29dof_rubberhand-feet_sphere-eef_box-body_capsule`).

## Important Notes

- Training uses distributed data parallel when multiple GPUs are available (`active_adaptation.is_distributed()`)
- The codebase compiles policies with `torch.compile` for performance
- Action delays and smoothing can be configured via `min_delay`, `max_delay`, and `alpha` parameters
- Object trajectories in HDMI tasks are manually annotated (see FAQ.md)
- Successful sim2real deployment code is in a separate repo: github.com/EGalahad/sim2real
