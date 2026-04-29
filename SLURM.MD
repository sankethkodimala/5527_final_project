`ssh {NetID}@agate.msi.umn.edu`

Project path: `/projects/standard/csci5527/shared/VizDoom`

Use conda to setup env:
```
conda create -n {name} python=3.8 -y
source activate {name}

module load python3
module load gcc/9.2.0
module load cuda/11.2

conda install pytorch=1.10.0 torchvision=0.11.1 cudatoolkit=11.3 -c pytorch -y
```