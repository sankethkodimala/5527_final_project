#!/bin/bash
#SBATCH --job-name=vizdoom_dqn_predict_position
#SBATCH -A csci5527
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=predict_position_logs/train_dqn_%j.out
#SBATCH --error=predict_position_logs/train_dqn_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=castl145@umn.edu

PYTHON=/users/4/castl145/.conda/envs/5527-ppo-new/bin/python

module load gcc/9.2.0
module load cuda/11.2

cd /projects/standard/csci5527/shared/VizDoom/5527_final_project

mkdir -p predict_position_dqn_logs

export SDL_VIDEODRIVER=offscreen
export DISPLAY=""

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:    $(date)"

$PYTHON train_dqn_predict_position.py \
    --timesteps 2000000 \
    --checkpoint-freq 100000 \
    --seed 5527 

echo "End: $(date)"
