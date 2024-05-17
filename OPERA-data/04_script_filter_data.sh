#!/bin/bash 
#SBATCH -o logs_filter/test.%j.out 
#SBATCH --mail-type=ALL 
#SBATCH --partition=cpu
#SBATCH --nodes=1 
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=8G 
#SBATCH --time=8:00:00 
#SBATCH --job-name=test-ssc-filter
# #SBATCH --array=6001-7500%50   # up to: train: 84572, val: 13213, test: 12444
# 3501-7162

num=$((SLURM_ARRAY_TASK_ID ))
echo $num
# conda activate ann-ssc
# python 04_preprocess_trainvaltest.py -i ${num} -split "train"
python 04_preprocess_trainvaltest_loop.py -split "test"
