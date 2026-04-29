#!/bin/bash
#SBATCH --job-name=vizdoom_ppo
#SBATCH -A csci5527
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nadim016@umn.edu

PYTHON=/users/8/nadim016/.conda/envs/vizdoom-env/bin/python

module load gcc/9.2.0
module load cuda/11.2

cd /projects/standard/csci5527/shared/VizDoom/5527_final_project

mkdir -p logs

export SDL_VIDEODRIVER=offscreen
export DISPLAY=""

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:    $(date)"

$PYTHON train.py --timesteps 1000000 --checkpoint-freq 100000 --seed 5527

echo "End: $(date)"
