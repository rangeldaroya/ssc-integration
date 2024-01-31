#!/bin/bash  
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=4G  # Requested Memory
#SBATCH -p cpu-preempt  # Partition
#SBATCH -G 0  # Number of GPUs
#SBATCH -o 01_slurm_feats/test-%j-%a.out  # %j = job ID
#SBATCH --job-name=test-integ-feats
#SBATCH --time=01:00:00 
#SBATCH --mail-type=ALL
#SBATCH --array=0-2488%200   # 100 jobs at a time (for train, until 18135. for val until 2488, for test until 2300)


echo "Running i=$((SLURM_ARRAY_TASK_ID))"
num=$((SLURM_ARRAY_TASK_ID))
# num=$((SLURM_ARRAY_TASK_ID + 10000))

# python 01_generate_feats_multitask.py --i ${num} --split "train"
# python 01_generate_feats_multitask.py --i ${num} --split "val"
python 01_generate_feats_multitask.py --i ${num} --split "test"