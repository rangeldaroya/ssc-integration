import pandas as pd
import numpy as np
import os
import rasterio as rio
import pickle
from loguru import logger
from tqdm import tqdm
import argparse
import time
import json
from pyproj import Transformer


DATA_DIR = "data_downloads/"
OUT_DATASET_DIR = "/scratch/workspace/rdaroya_umass_edu-water/hls_data/multitask_data_opera_v2"
nodata_val=-9999
opera_bands = ["B01_WTR", "B03_CONF", "B08_SHAD", "B09_CLOUD"]
landsat_bands = ["B02", "B03", "B04", "B05", "B06", "B07"]#, "Fmask"] # landsat (B,G,R,NIR,SWIR1,SWIR2)
sentinel_bands = ["B02", "B03", "B04", "B8A", "B11", "B12"]#, "Fmask"]


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

    samp_out_dir = os.path.join(OUT_DATASET_DIR, space_group)
    if not os.path.exists(samp_out_dir):
        os.makedirs(samp_out_dir, exist_ok=True)

    # TODO: randomly generate coordinates
    # crop_size = 512     # TODO: can change this to something bigger (to have random crops when training)
    # left_coor = 1574    # can be generated from 0 to (3659-512)
    # top_coor = 1574    # can be generated from 0 to (3659-512)
    
    crop_size = 512
    left_coor = np.random.randint(0, high=(3660-crop_size))
    top_coor = np.random.randint(0, high=(3660-crop_size))

    # center crop each tile (1574+512 = 2086) -- 1574:2086
    for group_ctr,group in enumerate(links):
        opera_samp = group["opera_link"].split("/")[-1]
        opera_prefix = opera_samp.split("_")[:-2]
        opera_prefix = "_".join(opera_prefix)
        logger.debug(f"opera_prefix: {opera_prefix}, opera_samp: {opera_samp}")
        opera_link = group["opera_link"]
        opera_sat = opera_link.split("_")[-5]


        # Process opera bands to get labels and other data (get high CONF water + sunmask, snowice, etc)
        # Get High Confidence water pixels
        fp_dir = os.path.join(DATA_DIR, space_group, "opera")
        opera_water_fp = os.path.join(fp_dir, f"{opera_prefix}_B01_WTR.tif")
        try:
            water_band = rio.open(opera_water_fp).read(1)[top_coor:top_coor+crop_size, left_coor:left_coor+crop_size]
        except rio.errors.RasterioIOError:
            logger.error(f"{opera_water_fp} cannot be read [space_group={space_group}]")
            continue
        assert water_band.shape==(crop_size,crop_size)  # get 1 (open water), 2 (partial water), 254 (ocean)
        water = np.where(water_band==1, 1, 0) + np.where(water_band==2, 1, 0) + np.where(water_band==254, 1, 0)   #1=open water; 2=partial surface water; 254=ocean masked
        water = np.where(water>0, 1, 0)
        # print(water_band.shape, water.shape, np.unique(water_band))

        opera_conf_fp = os.path.join(fp_dir, f"{opera_prefix}_B03_CONF.tif")
        try:
            conf_band = rio.open(opera_conf_fp).read(1)[top_coor:top_coor+crop_size, left_coor:left_coor+crop_size]
        except rio.errors.RasterioIOError:
            logger.error(f"{opera_conf_fp} cannot be read [space_group={space_group}]")
            continue
        assert conf_band.shape==(crop_size,crop_size)   # get 1 (open water high conf), 3 (partial surface water conservative), 254 (ocean)
        high_conf = np.where(conf_band==1, 1, 0) + np.where(conf_band==3, 1, 0) + np.where(conf_band==254, 1, 0)
        high_conf = np.where(high_conf>0, 1, 0)
        # print(high_conf.shape, np.unique(conf_band))

        high_conf_water = high_conf * water
        assert high_conf_water.shape == (crop_size,crop_size)


        # Get cloud mask
        opera_cloud_fp = os.path.join(fp_dir, f"{opera_prefix}_B09_CLOUD.tif")
        try:
            cloud_band = rio.open(opera_cloud_fp).read(1)[top_coor:top_coor+crop_size, left_coor:left_coor+crop_size]
        except rio.errors.RasterioIOError:
            logger.error(f"{opera_cloud_fp} cannot be read [space_group={space_group}]")
            continue
        assert cloud_band.shape==(crop_size,crop_size)
        cloud_data = np.where(cloud_band==4, 1, 0)

        # Get cloudshadow mask
        cloudshadow_data = np.where(cloud_band==1, 1, 0)

        # Get snow/ice mask
        snowice_data = np.where(cloud_band==2, 1, 0)


        # Get sunmask
        opera_sunmask_fp = os.path.join(fp_dir, f"{opera_prefix}_B08_SHAD.tif")
        try:
            sunmask_band = rio.open(opera_sunmask_fp).read(1)[top_coor:top_coor+crop_size, left_coor:left_coor+crop_size]
        except rio.errors.RasterioIOError:
            logger.error(f"{opera_sunmask_fp} cannot be read [space_group={space_group}]")
            continue
        assert sunmask_band.shape==(crop_size,crop_size)
        sunmask_data = sunmask_band


        if opera_sat.startswith("L"):
            hls_feat_bands = landsat_bands
        else:
            hls_feat_bands = sentinel_bands

        # Process hls bands to get features
        hls_samp = group["hls_link"].split("/")[-1]
        hls_prefix = hls_samp.split(".")[:-2]
        hls_prefix = ".".join(hls_prefix)
        logger.debug(f"hls_prefix: {hls_prefix}, hls_samp: {hls_samp}")
        fp_dir = os.path.join(DATA_DIR, space_group, "hls")
        feat_bands_data = []
        try:
            for band in hls_feat_bands:
                hls_fp = os.path.join(fp_dir, f"{hls_prefix}.{band}.tif")
                dataset = rio.open(hls_fp)
                data = dataset.read(1)[top_coor:top_coor+crop_size, left_coor:left_coor+crop_size]
                assert data.shape == (crop_size,crop_size)
                feat_bands_data.append(data)
                # print(hls_fp, data.shape)
        except rio.errors.RasterioIOError:
            logger.error(f"{hls_fp} cannot be read [space_group={space_group}]")
            continue
        features = np.stack(feat_bands_data, axis=-1)   # stack in 3rd axis
        assert features.shape == (crop_size,crop_size,6)
        if np.sum(np.where(features!=nodata_val, 1,0)) == 0:
            logger.error(f"{hls_fp} has all no data [{nodata_val}] features. Skipping.")
            continue

        # Process hls bands to get Fmask (for comparison later -- no need to postprocess; just save crop)
        fmask_fp = os.path.join(DATA_DIR, space_group, "hls", f"{hls_prefix}.Fmask.tif")  # 14 is FMask (UPDATE: 16 is FMask)
        try:
            fmask_dataset = rio.open(fmask_fp)
        except rio.errors.RasterioIOError:
            logger.error(f"{fmask_fp} cannot be read [space_group={space_group}]")
            continue
        fmask_data = fmask_dataset.read(1)
        assert np.max(fmask_data) <= 255
        assert np.min(fmask_data) >= 0
        fmask_data = fmask_data.astype(np.uint8)[top_coor:top_coor+crop_size, left_coor:left_coor+crop_size]
        assert fmask_data.shape == (crop_size,crop_size)

        # Generate lat/lon of middle of tiles
        ul = [dataset.bounds.left, dataset.bounds.top]
        ur = [dataset.bounds.right, dataset.bounds.top]
        lr = [dataset.bounds.right, dataset.bounds.bottom]
        ll = [dataset.bounds.left, dataset.bounds.bottom]
        transformer = Transformer.from_crs(dataset.crs, "EPSG:4326")    # lat/lon is EPSG 4326
        lats, lons = transformer.transform([ul[0], ur[0], lr[0], ll[0], ul[0]], [ul[1], ur[1], lr[1], ll[1], ul[1]])
        lat_c = (lats[0]+lats[2])/2
        lon_c = (lons[1]+lons[0])/2
        print(f"lat_c: {lat_c}, lon_c: {lon_c}")

        # TODO: save outputs
        out_path = os.path.join(OUT_DATASET_DIR, space_group, f"{i}_{tile_id}_{group_ctr:02d}.npy")   # Save with same idx in filtered csv
        with open(out_path, "wb") as f:
            np.savez(
                f,
                opera_sat=opera_sat,
                space_group=space_group,
                tile_id=tile_id,
                date_str=group["datetime"],
                features=features,
                fmask_data=fmask_data,

                mid_latlon=[lat_c, lon_c],

                water_mask=high_conf_water,
                snowice_mask=snowice_data,
                cloudshadow_mask=cloudshadow_data,
                sun_mask=sunmask_data,
                cloud_mask=cloud_data,
                # cirrus_mask=cirrus_mask,
                left_coor=left_coor,
                top_coor=top_coor,
                crop_size=crop_size,
            )
        logger.debug(f"Saved features to {out_path}")