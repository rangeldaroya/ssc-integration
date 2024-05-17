#!/bin/bash  
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=12G  # Requested Memory
#SBATCH -p gypsum-1080ti  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -o 01_slurm_feats/test-nosun-deeplabv3p-%j.out  # %j = job ID
#SBATCH --job-name=test-nosun-deeplabv3p-integ-feats
#SBATCH --time=96:00:00 
#SBATCH --mail-type=ALL
# #SBATCH --array=0-1500%100   # 100 jobs at a time (for train, until 18135. for val until 2488, for test until 2300)


echo "Running i=$((SLURM_ARRAY_TASK_ID))"
# num=$((SLURM_ARRAY_TASK_ID + 1500))
# num=$((SLURM_ARRAY_TASK_ID + 10000))

# python 01_generate_feats_multitask.py --i ${num} --split "train" --out_dir "opera_multitask_feats" --ckpt_path "ckpts_opera/alltasks_deeplabv3p_distrib.pth.tar"
python 01_generate_feats_multitask_loop.py \
    --split "test" \
    --out_dir "opera_multitask_feats_deeplabv3p_nosun" \
    --ckpt_path "ckpts_opera/alltasks_deeplabv3p_distrib_v2.pth.tar" \
    --tasks water_mask cloudshadow_mask cloud_mask snowice_mask 

# python 01_generate_feats_multitask_loop.py \
#     --split "train" \
#     --backbone "satlas_si_resnet50" \
#     --out_dir "opera_multitask_feats_satlas_si_resnet50" \
#     --ckpt_path "ckpts_opera/alltasks_satlas_si_resnet50_distrib.pth.tar" \
#     --tasks water_mask cloudshadow_mask cloud_mask snowice_mask sun_mask     

# python 01_generate_feats_multitask_loop.py \
#     --split "test" \
#     --backbone "mobilenetv3" \
#     --out_dir "opera_multitask_feats_mobilenetv3" \
#     --ckpt_path "ckpts_opera/alltasks_mobilenetv3_distrib_v2.pth.tar" \
#     --tasks water_mask cloudshadow_mask cloud_mask snowice_mask sun_mask   
# masks: water_mask cloudshadow_mask cloud_mask snowice_mask sun_mask    

# python 01_generate_feats_multitask.py --i ${num} --split "train" --out_dir "opera_multitask_feats" --ckpt_path "ckpts_opera/alltasks_deeplabv3p_distrib.pth.tar"
# python 01_generate_feats_multitask.py --i ${num} --split "val" --out_dir "opera_multitask_feats" --ckpt_path "ckpts_opera/alltasks_deeplabv3p_distrib.pth.tar"
# python 01_generate_feats_multitask.py --i ${num} --split "test" --out_dir "opera_multitask_feats" --ckpt_path "ckpts_opera/alltasks_deeplabv3p_distrib.pth.tar"