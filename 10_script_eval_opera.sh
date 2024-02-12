#!/bin/bash  
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=4G  # Requested Memory
#SBATCH -p gypsum-1080ti  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -o 10_slurm_opera/deeplabv3p-eval-%j.out  # %j = job ID
#SBATCH --job-name=deeplabv3p-eval-opera
#SBATCH --time=48:00:00 
#SBATCH --mail-type=ALL


python 10_opera_eval_model.py --backbone "deeplabv3p" --batch_size 1