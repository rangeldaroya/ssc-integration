import pandas as pd
import numpy as np
import os
import rasterio
import pickle
from loguru import logger
from tqdm import tqdm
import argparse
import time
import json


# TODO: save mapping from src_df to saved npy train/val/test idxs

OUT_DIR = "OPERA-data/data_downloads"
DATASET_DIR = "/scratch/workspace/rdaroya_umass_edu-water/hls_data/multitask_data_opera"
# train_df = pd.read_csv("../data/hls_train_df_new.csv")
# val_df = pd.read_csv("../data/hls_val_df_new.csv")
# test_df = pd.read_csv("../data/hls_test_df_new.csv")
# train_df = pd.read_csv("../data/hls_train_df_261k.csv")
# val_df = pd.read_csv("../data/hls_val_df_261k.csv")
# test_df = pd.read_csv("../data/hls_test_df_261k.csv")

nodata_val=-9999

parser = argparse.ArgumentParser(description='Generate dataset for FMask multitask training')
parser.add_argument('--split', default='train', type=str,
                    help='Split to process (train, val, test)')
parser.add_argument('--mode', default='generate', type=str,
                    help='Mode to run (generate, test)')


def filter_and_save_data(split="train"):
    # src_df = pd.read_csv(f"../data/hls_{split}_df_new.csv")
    src_df = pd.read_csv(f"../data/hls_{split}_df_261k.csv")

    valid_row_idxs = []
    # mapping = {}

    mapping_fp = os.path.join(DATASET_DIR, f"{split}_mapping.json")
    with open(mapping_fp, "r") as f:
        mapping = json.load(f)
    valid_row_idxs = list(mapping.keys())
    prev_row_ctr = mapping[str(len(valid_row_idxs)-1)]
    logger.debug(f"Loaded prev mapping: {len(mapping)} {len(valid_row_idxs)} {prev_row_ctr}")
    for row_ctr, (row_num, row) in tqdm(enumerate(src_df.iterrows()), total=len(src_df)):
        if row_ctr <= prev_row_ctr:
            logger.debug(f"Skipping {row_ctr}")
            continue
        
        # logger.debug(f"{row_ctr}, {row_num}")
        date_str, site_id, sat_type, tss_value = row["date"], row["SiteID"], row["satellite"], row["tss_value"]

        feat_bands = ["02", "03", "04", "15", "11", "12"]   # 13 is 8A (UPDATE: 15 is 8A)
        fp_dir = os.path.join(OUT_DIR, site_id, date_str, "cropped_imgs")
        if not os.path.exists(fp_dir):
            logger.error(f"{fp_dir} does not exist [row_ctr={row_ctr}, row_num={row_num}]")
            continue
        fp_dir_files = os.listdir(fp_dir)
        if len(fp_dir_files) == 0:
            logger.error(f"{fp_dir} does not contain files [row_ctr={row_ctr}, row_num={row_num}]")
            continue
        fn_samp = fp_dir_files[0]
        fn_prefix = fn_samp.split("_")[:-1]
        fn_prefix = "_".join(fn_prefix)

        feat_bands_data = []
        try:
            for band in feat_bands:
                fp = os.path.join(fp_dir, f"{fn_prefix}_{band}.tif")
                dataset = rasterio.open(fp)
                data = dataset.read(1)
                assert data.shape == (512,512)
                feat_bands_data.append(data)
        except rasterio.errors.RasterioIOError:
            logger.error(f"{fp} cannot be read [row_ctr={row_ctr}, row_num={row_num}]")
            continue
        features = np.stack(feat_bands_data, axis=-1)   # stack in 3rd axis
        if np.sum(np.where(features!=nodata_val, 1,0)) == 0:
            logger.error(f"{f} has all no data [{nodata_val}] features. Skipping.")
            continue
        # logger.debug(f"features: {features.shape}")

        fmask_fp = os.path.join(fp_dir, f"{fn_prefix}_16.tif")  # 14 is FMask (UPDATE: 16 is FMask)
        try:
            fmask_dataset = rasterio.open(fmask_fp)
        except rasterio.errors.RasterioIOError:
            logger.error(f"{fmask_fp} cannot be read [row_ctr={row_ctr}, row_num={row_num}]")
            continue

        fmask_data = fmask_dataset.read(1)
        assert np.max(fmask_data) <= 255
        assert np.min(fmask_data) >= 0
        fmask_data = fmask_data.astype(np.uint8)
        assert fmask_data.shape == (512,512)
        # For FMask bit information, see https://hls.gsfc.nasa.gov/wp-content/uploads/2019/01/HLS.v1.4.UserGuide_draft_ver3.1.pdf
        water_mask = ((fmask_data>>5) & 1)
        snowice_mask = ((fmask_data>>4) & 1)
        cloudshadow_mask = ((fmask_data>>3) & 1)
        cloud_mask = ((fmask_data>>1) & 1)
        cirrus_mask = (fmask_data & 1)

        sunmask_fp = os.path.join(fp_dir, f"{fn_prefix}_21.tif")  # 21 is sunmask
        try:
            sunmask_dataset = rasterio.open(sunmask_fp)
        except rasterio.errors.RasterioIOError:
            logger.error(f"{sunmask_fp} cannot be read [row_ctr={row_ctr}, row_num={row_num}]")
            continue
        sunmask_data = sunmask_dataset.read(1)
        assert np.nanmax(sunmask_data) <= 1
        assert np.nanmin(sunmask_data) >= 0
        sunmask_data = sunmask_data.astype(np.uint8)
        assert sunmask_data.shape == (512,512)

        valid_row_idxs.append(row_ctr)
        valid_row_ctr = len(valid_row_idxs) - 1
        mapping[valid_row_ctr] = row_ctr
        out_path = os.path.join(DATASET_DIR, split, f"{valid_row_ctr:06d}.npy")   # Save with same idx in filtered csv
        with open(out_path, "wb") as f:
            np.savez(
                f,
                site_id=site_id,
                date_str=date_str,
                features=features,
                water_mask=water_mask,
                snowice_mask=snowice_mask,
                cloudshadow_mask=cloudshadow_mask,
                sun_mask=sunmask_data,
                cloud_mask=cloud_mask,
                cirrus_mask=cirrus_mask,
                tss_value=tss_value,
            )
        # if row_ctr > 5:
        #     break

        mapping_fp = os.path.join(DATASET_DIR, f"{split}_mapping2.json")
        with open(mapping_fp, "w") as f:
            json.dump(mapping, f)
            
    filtered_src_df = src_df.iloc[valid_row_idxs]
    filtered_src_df["file_idx"] = list(range(len(filtered_src_df)))
    out_df_fp = f"../data/hls_{split}_df_filtered2.csv"
    filtered_src_df.to_csv(out_df_fp, index=False)
    logger.debug(f"Filtered {split} data to {filtered_src_df.shape} [fp={out_df_fp}]")

if __name__=="__main__":
    args = parser.parse_args()
    logger.debug(f"args: {args}")

    if args.mode == "generate":
        filter_and_save_data(split=args.split)
    elif args.mode == "test":
        # np.seterr(all='raise')
        # Check data
        split_df = pd.read_csv(f"../data/hls_{args.split}_df_filtered.csv")#[69834:]
        start_time = time.time()
        for row_ctr, row in split_df.iterrows():
            file_idx, actual_site_id, actual_date, actual_tss_value = row["file_idx"], row["SiteID"], row["date"], row["tss_value"]
            fp = os.path.join(DATASET_DIR, args.split, f"{file_idx:06d}.npy")
            if not os.path.exists(fp):
                continue
            with open(fp, "rb") as f:
                npzfile = np.load(f)
                features = npzfile["features"]
                water_mask = npzfile["water_mask"]
                cloudshadow_mask = npzfile["cloudshadow_mask"]
                cloud_mask = npzfile["cloud_mask"]
                cirrus_mask = npzfile["cirrus_mask"]
                sun_mask = npzfile["sun_mask"]
                site_id = npzfile["site_id"]
                date_str = npzfile["date_str"]
                tss_value = npzfile["tss_value"]
            # print(f"{file_idx}: {site_id}, {date_str}")
            # print(f"actual_site_id={actual_site_id} site_id={site_id}")
            print(f"fp:{fp}")
            feats_tmp = np.where(features==-9999, np.nanmax(features), features)
            feats_min = np.nanmin(feats_tmp)
            feats_tmp = np.where(features==-9999, feats_min, features)    # replace nodata values with minimum
            feats_tmp = np.where(np.isnan(feats_tmp), feats_min, feats_tmp)    # replace nan values with minimum
            feats_tmp = np.nan_to_num(feats_tmp, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            feats_tmp = (feats_tmp - np.nanmin(feats_tmp)) / np.maximum((np.nanmax(feats_tmp) - np.nanmin(feats_tmp)), 1)
            print(np.max(features), np.min(features), file_idx)
            print(np.max(feats_tmp), np.min(feats_tmp), file_idx)
            # assert actual_site_id == site_id
            # assert actual_date == date_str
            # assert actual_tss_value == tss_value
        dur = time.time()-start_time
        logger.debug(f"Processing all files took {dur} seconds for split {args.split}")
    else:
        raise NotImplementedError
            
"""
To read dataset


start_time = time.time()
with open(fp, "rb") as f:
    npzfile = np.load(f)
    features = npzfile["features"]
    water_mask = npzfile["water_mask"]
    cloudshadow_mask = npzfile["cloudshadow_mask"]
    cloud_mask = npzfile["cloud_mask"]
    cirrus_mask = npzfile["cirrus_mask"]
    dur = time.time()-start_time    # ~0.675
"""