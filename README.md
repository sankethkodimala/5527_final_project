# CSCI 5527 Final Project - ViZDoom Reinforcement Learning

This repository contains the code for a CSCI 5527 final project comparing reinforcement learning agents across ViZDoom tasks. The project focuses on PPO, Recurrent PPO with LSTM memory, and DQN training/evaluation pipelines.

## Scenarios
- Predict Position
- Deadly Corridor
- Defend the Center
- Defend the Line
- My Way Home

## Algorithms
- PPO
- Recurrent PPO with LSTM
- DQN

## Setup

Create and activate an environment, then install the project dependencies:

```bash
conda create -n vizdoom-env python=3.10 -y
conda activate vizdoom-env
pip install -r requirements.txt
```

## Training Examples

Defend the Line PPO:

```bash
python train_ppo_defend_the_line.py --timesteps 1000000 --seed 5527
```

Defend the Line Recurrent PPO with LSTM:

```bash
python train_lstm_defend_the_line.py --timesteps 1000000 --checkpoint-freq 100000 --seed 5527
```

Defend the Line DQN:

```bash
python train_dqn_defend_the_line.py --timesteps 1000000 --seed 5527
```

Several MSI/SLURM scripts are included for longer cluster runs, including:

```bash
sbatch train_ppo_defend_the_line.sh
sbatch train_lstm_defend_the_line.sh
sbatch train_dqn_defend_the_line.sh
```

Additional scenario-specific training scripts live in the scenario folders and top-level shell scripts.

## Evaluation

Evaluation scripts are included with each scenario where applicable. For Defend the Line:

```bash
python defend_the_line/evaluate_ppo_defend_the_line.py
python defend_the_line/evaluate_lstm_defend_the_line.py
python defend_the_line/evaluate_dqn_defend_the_line.py
```

These scripts expect locally generated model checkpoints. Checkpoints are not committed to this repository.

## Generated Outputs

Training generates checkpoints, model zip files, TensorBoard output, SLURM logs, local ViZDoom files, and Python caches. These files are intentionally ignored by Git so the repository remains code-only and reproducible from source.

Ignored generated outputs include:

- `checkpoints/`
- `tensorboard_logs/`
- `logs/`, `*_logs/`, and `my_logs/`
- `*.zip`, `*.out`, `*.err`, `*.log`, and `*.pdf`
- `__pycache__/`, `_vizdoom/`, and `*.ini`
- local virtual environments such as `.venv/` and `venv/`
