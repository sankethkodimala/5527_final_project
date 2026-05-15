#!/bin/bash
#SBATCH --job-name=vizdoom_defend_center_dqn_shaped
#SBATCH -A csci5527
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=18:00:00
#SBATCH --output=defend_center_logs/dqn_shaped_%j.out
#SBATCH --error=defend_center_logs/dqn_shaped_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kodim003@umn.edu

PYTHON=/users/9/kodim003/.conda/envs/5527-project/bin/python

module load gcc/9.2.0
module load cuda/11.2

cd /projects/standard/csci5527/shared/VizDoom/5527_final_project

mkdir -p defend_center_logs

export SDL_VIDEODRIVER=offscreen
export DISPLAY=""

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:    $(date)"
echo "Run:      DQN shaped reward, 10M steps"

$PYTHON train_defend_center_dqn.py --timesteps 10000000 --checkpoint-freq 200000 --seed 5527 --reward-mode shaped

echo "End: $(date)"

