#!/bin/zsh
#SBATCH -n 1
#SBATCH -t 240
#SBATCH -A dlclim
#SBATCH --gres=gpu:2

module load python/anaconda3

date

##tid=$SLURM_ARRAY_TASK_ID

program="runhs.py"

python ./$program 

date

