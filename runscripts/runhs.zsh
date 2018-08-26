#!/bin/zsh
#SBATCH -n 1
#SBATCH -t 4-0
#SBATCH -A dlclim
#SBATCH --gres=gpu:2

## Makes use of the array task id, so you have to run like so:
##   sbatch -a 0 runhs.zsh

module load python/anaconda3

date

tid=$SLURM_ARRAY_TASK_ID

program="runhs.py"

python ./$program -g 25 -e 500 ./testdata/dnnclim.dat ./hsearch.baseconfig$tid $tid

date

