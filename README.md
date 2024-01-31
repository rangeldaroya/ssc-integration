* Environment requirements in `multitask_env.yml`
* Download the data and ckpts [here](https://drive.google.com/drive/folders/1A34eUqeUzXpGmE4fgtaWrkTnfMl0geMp?usp=sharing)
* `01_generate_feats_multitask.py` uses the model to generate features (one csv file per tile)
    * There are currently three data splits (train, test, val) to generate features for
    * For batch processing, refer to `01_script_gen_feats.sh`
* To merge all the features in one file, use `02_merge_new_feats.ipynb`