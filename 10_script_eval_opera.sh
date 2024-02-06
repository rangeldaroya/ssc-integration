#!/bin/bash  
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=4G  # Requested Memory
#SBATCH -p gypsum-1080ti  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -o 10_slurm_opera/segnet-eval-%j.out  # %j = job ID
#SBATCH --job-name=segnet-eval-opera
#SBATCH --time=04:00:00 
#SBATCH --mail-type=ALL


python 10_opera_eval_model.py --backbone "segnet"