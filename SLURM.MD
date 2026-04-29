Connect to VPN if not on eduroam

SSH into:
`ssh {x500}@agate.msi.umn.edu`

Go to project
Project path: `/projects/standard/csci5527/shared/VizDoom/5527_final_project`

(First time or after env change) use conda to setup env:
```
conda create -n {name} python=3.10 -y
source activate {name}

module load gcc/9.2.0
module load cuda/11.2

pip install "torch==2.3.1" --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

In train.sh file before running:

Update PYTHON path in train.sh after conda env is setup
Change the #SBATCH's at top to change GPU, email that receives noti's, etc.

To submit/run:
`sbatch train.sh` submits a batch job
`squeue --me` views currently submitted jobs (PD = pending, in queue)