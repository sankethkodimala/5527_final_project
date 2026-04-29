Connect to VPN if not on eduroam

SSH into:
`ssh {NetID}@agate.msi.umn.edu`

(NetID is x500)

Go to project
Project path: `/projects/standard/csci5527/shared/VizDoom`

(First time or after env change) use conda to setup env:
```
conda create -n {name} python=3.8 -y
source activate {name}

module load python3
module load gcc/9.2.0
module load cuda/11.2

conda install pytorch=1.10.0 torchvision=0.11.1 cudatoolkit=11.3 -c pytorch -y
```

In train.sh file before running:

Update PYTHON path in train.sh after conda env is setup
Change the #SBATCH's at top to change GPU, email that receives noti's, etc.

To submit/run:

`sbatch train.sh` submits a batch job
`squeue -me` views currently running jobs