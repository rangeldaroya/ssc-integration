#!/bin/bash 
#SBATCH -o logs_dl2/output.%a.out 
#SBATCH --mail-type=ALL 
#SBATCH --partition=cpu-preempt
#SBATCH --nodes=1 
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=8G 
#SBATCH --time=1:00:00 
#SBATCH --job-name=ssc-dl
#SBATCH --array=4401-5000%20   # DONE: 0-3500, 5000 upwards
# 3501-7162

num=$((SLURM_ARRAY_TASK_ID ))
echo $num
# conda activate ann-ssc
python 01_download_data.py -i ${num}
# NOTE: might need to redo 3800-4000