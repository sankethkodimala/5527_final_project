#!/bin/bash
#SBATCH --job-name=vizdoom_lstm_defend_the_line
#SBATCH -A csci5527
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=defend_the_line_logs/train_lstm_%j.out
#SBATCH --error=defend_the_line_logs/train_lstm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nadim016@umn.edu

PYTHON=/users/8/nadim016/.conda/envs/vizdoom-env/bin/python

module load gcc/9.2.0
module load cuda/11.2

cd /projects/standard/csci5527/shared/VizDoom/5527_final_project

mkdir -p defend_the_line_logs

export SDL_VIDEODRIVER=offscreen
export DISPLAY=""

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:    $(date)"

# RESUME="--resume checkpoints/recurrent_ppo_VizdoomDefendLine-v1/best/best_model.zip"

$PYTHON train_lstm_defend_the_line.py \
    --timesteps 10000000 \
    --checkpoint-freq 100000 \
    --seed 5527 \
    $RESUME

echo "End: $(date)"
