#!/bin/bash
#SBATCH --job-name=vizdoom_defend_center_ppo
#SBATCH -A csci5527
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=04:00:00
#SBATCH --output=defend_center_logs/train_%j.out
#SBATCH --error=defend_center_logs/train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=castl145@umn.edu

PYTHON=/users/9/kodim003/.conda/envs/5527-project/bin/python

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

SCENARIO_ID=${SCENARIO_ID:-VizdoomDeadlyCorridor-v1}
ENVIRONMENT_CLASS=${ENVIRONMENT_CLASS:-CorridorDoomEnv}

$PYTHON train.py --timesteps 1000000 --checkpoint-freq 100000 --seed 5527 --scenario-id "$SCENARIO_ID" --environment-class "$ENVIRONMENT_CLASS"

echo "End: $(date)"
