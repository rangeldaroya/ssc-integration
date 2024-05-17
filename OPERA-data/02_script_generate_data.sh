#!/bin/bash 
#SBATCH -o logs_gendata_v2/output.%a.out 
#SBATCH --mail-type=ALL 
#SBATCH --partition=cpu-preempt
#SBATCH --nodes=1 
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=8G 
#SBATCH --time=4:00:00 
#SBATCH --job-name=ssc-generate
#SBATCH --array=4401-5000%20   # Done: 0-3000, 5000-6000, 6001-7162 # TODO: up to 4400
# 3501-7162

num=$((SLURM_ARRAY_TASK_ID ))
echo $num
# conda activate ann-ssc
python 02_generate_data.py -i ${num}
