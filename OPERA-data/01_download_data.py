#from original notebook:
import requests
import json

import numpy as np

import os
from urllib.request import urlretrieve
import earthaccess
from earthaccess import Auth, DataCollections, DataGranules, Store

from numpy.lib.stride_tricks import sliding_window_view

from tqdm import tqdm
from loguru import logger
from datetime import datetime


import argparse


opera_bands = ["B01_WTR", "B03_CONF", "B08_SHAD", "B09_CLOUD"]
landsat_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "Fmask"] # landsat (B,G,R,NIR,SWIR1,SWIR2)
sentinel_bands = ["B02", "B03", "B04", "B8A", "B11", "B12", "Fmask"]



if __name__=="__main__":
    arg_parser = argparse.ArgumentParser(description="Append results of Confluence workflow execution to the SoS.")
    arg_parser.add_argument("-i",
                            type=int,
                            help="Index to specify input data to execute on, value of -235 indicates AWS selection")
    args = arg_parser.parse_args()
    logger.debug(f"args: {args}")
    i = args.i
    logger.debug(f"Processing index {i}")
    fname = f"{i}.npy"
    fp = os.path.join("sampled_per_tile_id", fname)
    with open(fp, "rb") as f:
        npzfile = np.load(f, allow_pickle=True)
        tile_id = npzfile["tile_id"][()]
        links = npzfile["links"][()]
        space_group = npzfile["space_group"][()]
    logger.debug(f"Processing tiles from {space_group}")
    opera_downloads = []
    hls_downloads = []

    for group in links:
        opera_link = group["opera_link"]
        opera_sat = opera_link.split("_")[-5]
        for b in opera_bands:
            link = opera_link.replace("B01_WTR", b)
            opera_downloads.append(link)

        if opera_sat.startswith("L"):
            hls_bands = landsat_bands
        else:
            hls_bands = sentinel_bands
        
        hls_link = group["hls_link"]
        for b in hls_bands:
            link = hls_link.replace("B01", b)
            hls_downloads.append(link)

    logger.debug(f"Downloading {len(opera_downloads)} opera_downloads")
    logger.debug(f"Downloading {len(hls_downloads)} hls_downloads")
    earthaccess.download(opera_downloads, f"./data_downloads/{space_group}/opera")
    earthaccess.download(hls_downloads, f"./data_downloads/{space_group}/hls")
    logger.debug("Finished download")
