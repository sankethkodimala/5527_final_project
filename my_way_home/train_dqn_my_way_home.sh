#!/bin/bash
#SBATCH --job-name=vizdoom_dqn_my_way_home
#SBATCH -A csci5527
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=my_way_home_dqn_logs/train_dqn_%j.out
#SBATCH --error=my_way_home_dqn_logs/train_dqn_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=speng026@umn.edu

PYTHON=/users/4/speng026/.conda/envs/5527/bin/python

module load gcc/9.2.0
module load cuda/11.2

cd /projects/standard/csci5527/shared/VizDoom/5527_final_project/my_way_home

mkdir -p my_way_home_dqn_logs

export SDL_VIDEODRIVER=offscreen
export DISPLAY=""

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:    $(date)"

$PYTHON train_dqn_my_way_home.py \
    --timesteps 10000000 \
    --checkpoint-freq 250000 \
    --seed 5527

echo "End: $(date)"
