#!/bin/bash 
#SBATCH -o logs/delete.%j.out 
#SBATCH --mail-type=ALL 
#SBATCH --partition=cpu-preempt
#SBATCH --nodes=1 
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=16G 
#SBATCH --time=8:00:00 
#SBATCH --job-name=del-files

# conda activate ann-ssc
python 03_delete_files.py