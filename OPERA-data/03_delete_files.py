import os
import numpy as np
from loguru import logger

start_idx = 4000
end_idx = 4400

opera_bands = ["B01_WTR", "B03_CONF", "B08_SHAD", "B09_CLOUD"]
landsat_bands = ["B02", "B03", "B04", "B05", "B06", "B07"]#, "Fmask"] # landsat (B,G,R,NIR,SWIR1,SWIR2)
sentinel_bands = ["B02", "B03", "B04", "B8A", "B11", "B12"]#, "Fmask"]
DATA_DIR = "data_downloads/"

for i in range(start_idx, end_idx):
    fname = f"{i}.npy"
    fp = os.path.join("sampled_per_tile_id", fname)
    with open(fp, "rb") as f:
        npzfile = np.load(f, allow_pickle=True)
        tile_id = npzfile["tile_id"][()]
        links = npzfile["links"][()]
        space_group = npzfile["space_group"][()]

    all_filenames = []
    for group_ctr,group in enumerate(links):
        opera_samp = group["opera_link"].split("/")[-1]
        opera_prefix = opera_samp.split("_")[:-2]
        opera_prefix = "_".join(opera_prefix)
        # logger.debug(f"opera_prefix: {opera_prefix}, opera_samp: {opera_samp}")
        opera_link = group["opera_link"]
        opera_sat = opera_link.split("_")[-5]

        fp_dir = os.path.join(DATA_DIR, space_group, "opera")
        opera_water_fp = os.path.join(fp_dir, f"{opera_prefix}_B01_WTR.tif")
        all_filenames.append(opera_water_fp)
        opera_conf_fp = os.path.join(fp_dir, f"{opera_prefix}_B03_CONF.tif")
        all_filenames.append(opera_conf_fp)
        opera_cloud_fp = os.path.join(fp_dir, f"{opera_prefix}_B09_CLOUD.tif")
        all_filenames.append(opera_cloud_fp)
        opera_sunmask_fp = os.path.join(fp_dir, f"{opera_prefix}_B08_SHAD.tif")
        all_filenames.append(opera_sunmask_fp)


        if opera_sat.startswith("L"):
            hls_feat_bands = landsat_bands
        else:
            hls_feat_bands = sentinel_bands

        # Process hls bands to get features
        hls_samp = group["hls_link"].split("/")[-1]
        hls_prefix = hls_samp.split(".")[:-2]
        hls_prefix = ".".join(hls_prefix)
        # logger.debug(f"hls_prefix: {hls_prefix}, hls_samp: {hls_samp}")
        fp_dir = os.path.join(DATA_DIR, space_group, "hls")
        feat_bands_data = []
        for band in hls_feat_bands:
            hls_fp = os.path.join(fp_dir, f"{hls_prefix}.{band}.tif")
            all_filenames.append(hls_fp)
        fmask_fp = os.path.join(DATA_DIR, space_group, "hls", f"{hls_prefix}.Fmask.tif")
        all_filenames.append(fmask_fp)

    for filename in all_filenames:
        if os.path.exists(filename):
            os.remove(filename)
            logger.debug(f"{i} {filename} removed")
        else:
            logger.error(f"{i} {filename} does not exist")