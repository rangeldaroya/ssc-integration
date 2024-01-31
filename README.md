* Environment requirements in `multitask_env.yml`
* Download the data and ckpts [here](https://drive.google.com/drive/folders/1A34eUqeUzXpGmE4fgtaWrkTnfMl0geMp?usp=sharing)
* `01_generate_feats_multitask.py` uses the model to generate features (one csv file per tile)
    * There are currently three data splits (train, test, val) to generate features for
    * For batch processing, refer to `01_script_gen_feats.sh`
* To merge all the features in one file, use `02_merge_new_feats.ipynb`


Some Notes:
* The features are currently generated over a 512 pixel by 512 pixel tile centered at the given lat/lon (each pixel is 30m by 30m on HLS)
* The code that generates features assumes that the necessary tiles are already downloaded and placed in `/scratch/workspace/rdaroya_umass_edu-water/hls_data/output`