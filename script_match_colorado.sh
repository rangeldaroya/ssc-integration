#!/bin/bash  
#SBATCH -c 8  # Number of Cores per Task
#SBATCH --mem=16G  # Requested Memory
#SBATCH -p gypsum-1080ti  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -o slurm/colorado-%j.out  # %j = job ID
#SBATCH --job-name=colorado-opera
#SBATCH --time=24:00:00 
#SBATCH --mail-type=ALL


python match_hls_colorado.py