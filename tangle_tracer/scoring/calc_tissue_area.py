import os
import pandas as pd
import zarr
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def get_tissue_area_all_imgs(img_path, downsample_rate=64):
    valid_wsis = [wsi for wsi in os.listdir(img_path) if '.sync' not in wsi]
    wsi_df = pd.DataFrame({'WSI': valid_wsis})
    wsi_df[f'tissue_area_downsample{downsample_rate}'] = None

    for wsi in tqdm(valid_wsis):
        img = zarr.open(Path(img_path, wsi), mode='r')
        downsampled_img = img[::downsample_rate, ::downsample_rate]

        # Use numpy operations to calculate area of non-black region
        area = np.count_nonzero(downsampled_img)

        wsi_df.loc[wsi_df['WSI'] == wsi, f'tissue_area_downsample{downsample_rate}'] = area

    return wsi_df

def main():
    parser = argparse.ArgumentParser(description='Calculate tissue area for all images in a directory.')
    parser.add_argument('img_path', type=str, help='Path to directory containing WSIs.')
    parser.add_argument('--downsample_rate', type=int, default=64, help='Downsampling rate to apply to WSIs.')
    parser.add_argument('--output_path', type=str, default='./tissue_area.csv', help='Path and name of output file.')
    args = parser.parse_args()
    os.makedirs(Path(args.output_path).parent, exist_ok=True)

    wsi_df = get_tissue_area_all_imgs(args.img_path, args.downsample_rate)
    
    # Save results to CSV file
    wsi_df.to_csv(args.output_path, index=False)

if __name__ == '__main__':
    main()
