#from original notebook:
# import requests
# import json

import numpy as np

import os
# from urllib.request import urlretrieve
# import earthaccess
# from earthaccess import Auth, DataCollections, DataGranules, Store

# from numpy.lib.stride_tricks import sliding_window_view

from tqdm import tqdm
from loguru import logger
# from datetime import datetime


import argparse


DATA_DIR = "/scratch/workspace/rdaroya_umass_edu-water/hls_data/multitask_data_opera_v2"
nodata_val = -9999

if __name__=="__main__":
    arg_parser = argparse.ArgumentParser(description="Append results of Confluence workflow execution to the SoS.")
    arg_parser.add_argument("-i",
                            type=int,
                            help="Index to specify input data to execute on, value of -235 indicates AWS selection")
    arg_parser.add_argument("-split",
                            type=str,
                            default="train",
                            help="train/val/test")

    args = arg_parser.parse_args()
    logger.debug(f"args: {args}")

    suffix = "108k"

    filepaths_fp = os.path.join(DATA_DIR, f"{args.split}_{suffix}.npy")

    with open(filepaths_fp, "rb") as f:
        npzfile = np.load(f, allow_pickle=True)
        fps = npzfile["fps"]
    
    start = 0
    if args.split == "train":
        # start = 15800
        end = 84572
    elif args.split == "val":
        # start = 8300
        end = 13213
    elif args.split == "test":
        # start = 8300
        end = 12444

    for index in range(start, end):
        fp = fps[index]
        # logger.debug(f"fp: {fp}")
        logger.debug(f"i: {index}. total: {len(fps)}")
        with open(fp, "rb") as f:
            npzfile = np.load(f, allow_pickle=True)
            features = npzfile["features"][:512, :512]
            assert features.shape == (512,512,6)
        if np.sum(np.where(features!=nodata_val, 1,0)) == 0:
            logger.error(f"[train] {fp} has all no data [{nodata_val}] features. Skipping.")
        else:
            with open(f"filtered_datanew/{args.split}/{index}.txt", "w") as f:
                f.write("1")
