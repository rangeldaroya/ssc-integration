from datetime import datetime
import rasterio
import numpy as np
import pandas as pd
import os
from loguru import logger
from glob import glob

from matplotlib import pyplot as plt


# Find all relevant opera tiles
ctr = 0
opera_files_with_match = []
opera_hls_mapping = {}
for fp_opera in os.listdir("colorado-opera"):
    if not fp_opera.endswith("WTR.tif"):
        continue
    opera_sat = fp_opera.split("_")[-5]
    opera_tile_id = fp_opera.split("_")[-8]
    opera_datetime = fp_opera.split("_")[-7]
    opera_year = opera_datetime[:4]
    opera_date = opera_datetime.split("T")[0]
    datetime_object = datetime.strptime(opera_date, '%Y%m%d')
    opera_day_of_year = datetime_object.timetuple().tm_yday

    hls_date = f"{opera_year}{opera_day_of_year}"
    
    if opera_sat == "L8": # landsat
        sat_dir = "colorado-landsat"
    else:
        sat_dir = "colorado-sentinel"
    sat_files = glob(os.path.join(sat_dir, f"*.{opera_tile_id}.{hls_date}*"))
    if len(sat_files)>0:
        opera_hls_mapping[fp_opera] = sat_files
        opera_files_with_match.append(fp_opera)
        ctr += 1
        logger.debug(f"[{len(sat_files)}]. opera_sat: {opera_sat}, opera_tile_id: {opera_tile_id}, opera_year: {opera_year}, opera_day_of_year: {opera_day_of_year}. hls_date: {hls_date}")
        # break
logger.debug(f"Found {ctr} matched opera files")


print(len(opera_files_with_match))
ctr = 0
for fp in opera_files_with_match:
    dataset = rasterio.open(os.path.join("colorado-opera", fp))
    data = dataset.read(1)
    data_vals = np.unique(data)
    if 1 in data_vals:
        ctr += 1
        logger.debug(f"vals: {data_vals}")
logger.debug(f"Found {ctr} tiles with water pixels")



output_chunks = []
input_chunks = []
fmask_chunks = []
satellite_types = []
for opera_fn, hls_fps in opera_hls_mapping.items():
    output_dataset = rasterio.open(os.path.join("colorado-opera", opera_fn))
    output_data = output_dataset.read()

    samp_fp = hls_fps[0]
    if "landsat" in samp_fp:
        sat_type = "L"
        bands = ["B02", "B03", "B04", "B05", "B06", "B07"]  # landsat (B,G,R,NIR,SWIR1,SWIR2)
    else:
        sat_type = "S"
        bands = ["B02", "B03", "B04", "B8A", "B11", "B12"]  # sentinel
    # print(samp_fp)
    prefix = ".".join(samp_fp.split(".")[:-2])
    input_bands = []
    for b in bands:
        band_path = f"{prefix}.{b}.tif"
        band_data = rasterio.open(band_path).read()
        input_bands.append(band_data)
    input_data = np.concatenate(input_bands, axis=0)
    fmask_path = f"{prefix}.Fmask.tif"
    fmask_data = rasterio.open(fmask_path).read()

    # print(input_data.shape, output_data.shape)
    
    _, s1,s2 = output_data.shape
    chunk_size = 512
    for i in range(s1//chunk_size):
        for j in range(s2//chunk_size):
            input_chunk = input_data[:, i*chunk_size : (i+1)*chunk_size, j*chunk_size : (j+1)*chunk_size,]
            output_chunk = output_data[:, i*chunk_size : (i+1)*chunk_size, j*chunk_size : (j+1)*chunk_size,]
            fmask_chunk = fmask_data[:, i*chunk_size : (i+1)*chunk_size, j*chunk_size : (j+1)*chunk_size,]
            non_nodata = np.where(output_chunk==255, 0, 1)
            if np.sum(non_nodata) < 1:  # means there's no data in the whole chunk
                continue
            unique_vals = np.unique(output_chunk)
            if (1 not in unique_vals) and (2 not in unique_vals) and (254 not in unique_vals):
                continue
            input_chunks.append(input_chunk)
            output_chunks.append(output_chunk)
            fmask_chunks.append(fmask_chunk)
            satellite_types.append(sat_type)
            break
        break
    break
print(input_chunks)
num_pos_samples = 0
ctr = 0
for inp, outp, fmask, sat_type in zip(input_chunks, output_chunks, fmask_chunks, satellite_types):
    out_path = os.path.join("colorado_data", f"{ctr:06d}.npy")
    with open(out_path, "wb") as f:
        np.savez(
            f,
            input=inp,
            output=outp,
            fmask=fmask,
            satellite=sat_type,
        )
    ctr += 1
    if np.sum(outp)>0:
        num_pos_samples += 1
print(num_pos_samples)