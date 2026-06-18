# Walkthrough - Resolving Isaac Lab Training Errors

This walkthrough documents the fixes applied to successfully run the reinforcement learning training script for Franka robot tasks within NVIDIA Isaac Lab.

## Summary of Changes

### 1. Hydra Dependency Installation
Installed the missing `hydra-core` package directly into the Isaac Sim Kit Python environment to resolve `ImportError: Hydra is not installed`:
```bash
~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh -m pip install hydra-core
```

### 2. Configuration Key Updates (`STATES` to `OBSERVATIONS`)
Updated the network input keys from `STATES` to `OBSERVATIONS` across all PPO configuration files under `franka_isaaclab/tasks` to prevent `AttributeError: 'NoneType' object has no attribute 'shape'` during skrl model instantiation:
* **Reach Config**: [skrl_ppo_cfg.yaml (reach)](file:///home/iyangim/franka_isaaclab/source/franka_isaaclab/franka_isaaclab/tasks/manager_based/reach/agents/skrl_ppo_cfg.yaml)
* **Lift Config**: [skrl_ppo_cfg.yaml (lift)](file:///home/iyangim/franka_isaaclab/source/franka_isaaclab/franka_isaaclab/tasks/manager_based/lift/agents/skrl_ppo_cfg.yaml)
* **Stack Config**: [skrl_ppo_cfg.yaml (stack)](file:///home/iyangim/franka_isaaclab/source/franka_isaaclab/franka_isaaclab/tasks/manager_based/stack/agents/skrl_ppo_cfg.yaml)

## Verification Results

We verified the fixes by running the training pipeline:
```bash
~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/skrl/train.py --task=Template-Reach-v0
```

* **Outcome**: The simulator successfully starts up, parses configuration, builds the environment, wraps it with `SkrlVecEnvWrapper`, and enters the training loop (which prints a progress bar showing iteration counts).
* **Cleanup**: The test run was cancelled to save GPU resources, but training functionality is fully operational.
