#!/bin/bash
#SBATCH --job-name=vizdoom_my_way_home_standard_ppo
#SBATCH -A csci5527
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=11:00:00
#SBATCH --output=my_way_home_logs_short/v2/train_%j.out
#SBATCH --error=my_way_home_logs_short/v2/train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=speng026@umn.edu

PYTHON=/users/4/speng026/.conda/envs/5527/bin/python

module load gcc/9.2.0
module load cuda/11.2

cd /projects/standard/csci5527/shared/VizDoom/5527_final_project

mkdir -p my_way_home_logs_short/v2

export SDL_VIDEODRIVER=offscreen
export DISPLAY=""

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:    $(date)"

SCENARIO_ID=${SCENARIO_ID:-VizdoomMyWayHome-v1}
ENVIRONMENT_CLASS=${ENVIRONMENT_CLASS:-MyWayHomeDoomEnv}

$PYTHON train_mark.py --timesteps 15000000 --checkpoint-freq 750000 --seed 5527 --scenario-id "$SCENARIO_ID" --environment-class "$ENVIRONMENT_CLASS"

echo "End: $(date)"
