import torch
import torchvision.transforms as transforms
import pandas as pd 
import numpy as np 
import os
from loguru import logger
import rasterio
from matplotlib import pyplot as plt
import argparse

from models.get_model import get_model


parser = argparse.ArgumentParser(description='Generate dataset for FMask multitask training')
parser.add_argument('--i', default=0, type=int,
                    help='Index to process from csv file')
parser.add_argument('--data_dir', default='/scratch/workspace/rdaroya_umass_edu-water/hls_data/output', type=str,
                    help='Directory of HLS data (sources of bands of size 512x512 pixels, centered at the lat/lon of SSC)')
parser.add_argument('--out_dir', default='integrated_feats', type=str,
                    help='Directory where produced features will be saved')
parser.add_argument('--split', default='train', type=str,
                    help='Split to process (Choices: train, val, test)')
parser.add_argument('--ckpt_path', default='ckpts/deeplabv3p_distrib.pth.tar', type=str,
                    help='Checkpoint path of multitask model to use')
parser.add_argument('--backbone', default='deeplabv3p', type=str,
                    help='Backbone of model being used')
parser.add_argument('--is_distrib', default=1, type=int,
                    help='Flag to indicate if model was trained in distributed way (1=distributed, 0=not)')


def get_band_fp_data(data_dir, site_id, date_str):
    fp_dir = os.path.join(data_dir, site_id, date_str, "cropped_imgs") 
    fp_dir_files = os.listdir(fp_dir)
    fn_samp = fp_dir_files[0]
    fn_prefix = fn_samp.split("_")[:-1]
    fn_prefix = "_".join(fn_prefix)
    return fp_dir, fn_prefix

def get_features(band, fp_dir, fn_prefix, final_water_mask, nodata_val=-9999):
    if band == "8a":
        fp = os.path.join(fp_dir, f"{fn_prefix}_15.tif")    # band 15 in cropped_imgs is band8a
    else:
        fp = os.path.join(fp_dir, f"{fn_prefix}_{band:02d}.tif")

    if os.path.exists(fp):
        dataset = rasterio.open(fp)
        data = dataset.read(1)
        assert data.shape == (512,512)

        data = np.where(data==nodata_val, np.nan, data) # replace nodata vals with nan
        mask = np.where(final_water_mask==1, 1, np.nan) # change masked out values to nan
        
        masked_feat = data * mask
        band_min = np.nanmin(masked_feat)
        band_mean = np.nanmean(masked_feat)
        band_max = np.nanmax(masked_feat)
        band_std = np.nanstd(masked_feat)
        band_median = np.nanmedian(masked_feat)
        tmp = np.where(masked_feat==0, np.nan, masked_feat)   # treat nan values as zero
        tmp = np.where(np.isnan(tmp), 0, tmp)
        band_count = np.count_nonzero(tmp)
        # band_count = np.count_nonzero(masked_feat)
    else:
        logger.warning(f"{fp} does not exist")
        band_min = None
        band_mean = None
        band_max = None
        band_std = None
        band_median = None
        band_count = None
    return band_min, band_mean, band_max, band_std, band_median, band_count

def get_input_data(fp_dir, fn_prefix):   # uses cropped tiles (512 x 512) as input to the model (center of image is ssc sample)
    split = "test"
    feat_bands = ["02", "03", "04", "15", "11", "12"]   # 15 is 8A
    
    feat_bands_data = []
    for band in feat_bands:
        fp = os.path.join(fp_dir, f"{fn_prefix}_{band}.tif")
        dataset = rasterio.open(fp)
        data = dataset.read(1)
        assert data.shape == (512,512)
        feat_bands_data.append(data)
    features = np.stack(feat_bands_data, axis=-1)   # stack in 3rd axis

    return features

def prepare_inp_data(features, nodata_val=-9999):
    feats_tmp = np.where(features==nodata_val, np.nanmax(features), features)
    feats_min = np.nanmin(feats_tmp)
    feats_tmp = np.where(features==nodata_val, feats_min, features)    # replace nodata values with minimum
    feats_tmp = np.where(np.isnan(feats_tmp), feats_min, feats_tmp)    # replace nan values with minimum
    feats_tmp = np.nan_to_num(feats_tmp, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = feats_tmp.astype(np.float32)
    image = data_transforms(image)
    # Normalize data
    if (torch.max(image)-torch.min(image)):
        image = image - torch.min(image)
        image = image / torch.maximum(torch.max(image),torch.tensor(1))
    else:
        image = np.zeros_like(image)
    image = image.type(torch.FloatTensor)
    return image

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
def get_masks(inp_data, ckpt_path, backbone, is_distrib=True):
    opt = Namespace(
        ckpt_path=ckpt_path,
        backbone=f'{backbone}',
        head=f'{backbone}_head',
        method='vanilla', 
        tasks=['water_mask', 'cloudshadow_mask', 'cloud_mask', 'snowice_mask', 'sun_mask'], 
    )
    num_inp_feats = 6   # number of channels in input
    tasks_outputs = {
        "water_mask": 1,
        "cloudshadow_mask": 1,
        "cloud_mask": 1,
        "snowice_mask": 1,
        "sun_mask": 1,
    }
    model = get_model(opt, tasks_outputs=tasks_outputs, num_inp_feats=num_inp_feats)

    logger.debug(f"Loading weights from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    if is_distrib:
        new_ckpt = {k.split("module.")[-1]:v for k,v in checkpoint["state_dict"].items()}
        checkpoint["state_dict"] = new_ckpt
    tmp = model.load_state_dict(checkpoint["state_dict"], strict=True)
    logger.debug(f"After loading ckpt: {tmp}")
    logger.debug(f"Checkpoint epoch: {checkpoint['epoch']}. best_perf: {checkpoint['best_performance']}")
    if backbone == "deeplabv3p":
        optim_threshes = {     # for DeepLabv3+
            "water_mask": 0.2,
            "cloudshadow_mask": 0.2,
            "cloud_mask": 0.3,
            "snowice_mask": 0.2,
            'sun_mask': 0.3,    # for 261k dataset
        }
    elif backbone == "mobilenetv3":
        optim_threshes = {     # for MobileNet
            "water_mask": 0.2,
            "cloudshadow_mask": 0.2,
            "cloud_mask": 0.3,
            "snowice_mask": 0.2,
            'sun_mask': 0.3,
        }
    elif backbone == "segnet":
        optim_threshes = {     # for SegNet
            "water_mask": 0.2,
            "cloudshadow_mask": 0.2,
            "cloud_mask": 0.3,
            "snowice_mask": 0.2,
            'sun_mask': 0.5,
        }
    model.eval()
    inp_data = prepare_inp_data(inp_data)   # size: (6,512,512)
    inp_data = torch.unsqueeze(inp_data, dim=0) # to have batch size of 1
    with torch.no_grad():
        test_pred, feat = model(inp_data, feat=True)
        
    masks = {}
    for t in tasks_outputs.keys():
        pred_img = test_pred[t][0,:,:].detach().cpu().numpy()
        thresh = optim_threshes[t]
        masks[t] = (pred_img > thresh).astype(int).squeeze()

    return masks

ID_cols = ['SiteID', 'lat', 'lon', 'date',
       'cloud_cover', 'tss_value', 'relative_day', "MGRS", "LorS"]


if __name__=="__main__":
    args = parser.parse_args()
    logger.debug(f"args: {args}")

    split_csv = f"data/{args.split}_ssc.csv"
    split_csv = pd.read_csv(split_csv)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
    out_split_dir = os.path.join(args.out_dir, args.split)
    if not os.path.exists(out_split_dir):
        os.makedirs(out_split_dir, exist_ok=True)


    split_data = split_csv[ID_cols]
    row = split_data.iloc[args.i]

    new_data = row[ID_cols]
    site_id, date_str = row["SiteID"], row["date"]
    fp_dir, fn_prefix = get_band_fp_data(args.data_dir, site_id, date_str)   # get directory of tile, and the prefix for cropped imgs
    inp_data = get_input_data(fp_dir, fn_prefix)    # size: (512,512,6)
    masks = get_masks(inp_data, ckpt_path=args.ckpt_path, backbone=args.backbone, is_distrib=(args.is_distrib==1))

    water_mask = masks["water_mask"]
    cloudshadow_mask = masks["cloudshadow_mask"]
    cloud_mask = masks["cloud_mask"]
    snowice_mask = masks["snowice_mask"]
    sun_mask = masks["sun_mask"]

    final_water_mask = water_mask * (1-cloudshadow_mask) * (1-cloud_mask) * (1-snowice_mask) * sun_mask
    # final_water_mask = water_mask * sun_mask
    
    for band in list(range(1, 13))+["8a"]:
        band_min, band_mean, band_max, band_std, band_median, band_count = get_features(
            band, fp_dir, fn_prefix, final_water_mask
        )

        new_data[f"b{band}_min"] = band_min
        new_data[f"b{band}_mean"] = band_mean
        new_data[f"b{band}_max"] = band_max
        new_data[f"b{band}_std"] = band_std
        new_data[f"b{band}_median"] = band_median
        new_data[f"b{band}_count"] = band_count

    df = pd.DataFrame(new_data.values.reshape(1,-1), columns=list(new_data.keys()))
    out_fp = os.path.join(out_split_dir, f"{args.i:06d}.csv")
    df.to_csv(out_fp, index=False)