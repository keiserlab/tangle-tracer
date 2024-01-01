import os
import argparse
from pathlib import Path
import pandas as pd
import zarr
from tqdm import tqdm
import cv2
import numpy as np
from skimage import measure

def count_blobs(mask, size_threshold=20):
    #are there too many labels? check how many to see if this can be optimized
    labels = measure.label(mask, connectivity=2, background=0)
    new_mask = np.zeros(mask.shape, dtype='uint8')
    sizes = []
    
    # loop over the unique components
    for label in tqdm(np.unique(labels)):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels 
        labelMask = np.zeros(mask.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > size_threshold:
            sizes.append(numPixels)
            new_mask = cv2.add(new_mask, labelMask)

    num_labels = len(np.unique(labels))
    return sizes, new_mask, num_labels

def score_all_heatmaps(heatmap_path, downsample_rate=64, size_threshold=1):
    valid_wsis = [wsi for wsi in os.listdir(heatmap_path) if ".sync" not in wsi]
    wsi_df = pd.DataFrame(pd.Series(valid_wsis, name="WSI"))
    wsi_df[f"CNN_NFT_count_downsample{downsample_rate}"] = None
    wsi_df[f"CNN_NFT_labels_downsample{downsample_rate}"] = None
    for wsi in tqdm(valid_wsis):
        heatmap = zarr.open(Path(heatmap_path, wsi), mode="r")
        downsampled_heatmap = (heatmap[::downsample_rate, ::downsample_rate] > 0.5) * 1.0
        sizes, mask, num_labels = count_blobs(downsampled_heatmap, size_threshold=size_threshold)
        wsi_df.loc[wsi_df["WSI"] == wsi, f"CNN_NFT_count_downsample{downsample_rate}"] = len(sizes)
        wsi_df.loc[wsi_df["WSI"] == wsi, f"CNN_NFT_labels_downsample{downsample_rate}"] = num_labels

    return wsi_df


def main(args):
    heatmap_path = args.heatmap_path
    downsample_rate = args.downsample_rate
    size_threshold = args.size_threshold
    output_file = args.output_file
    os.makedirs(Path(output_file).parent, exist_ok=True)

    score_df = score_all_heatmaps(heatmap_path, downsample_rate, size_threshold)
    score_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score all heatmaps in a directory")
    parser.add_argument("heatmap_path", type=str, help="path to directory containing heatmaps")
    parser.add_argument("--downsample_rate", type=int, default=64, help="downsample rate (default=64)")
    parser.add_argument("--size_threshold", type=int, default=1, help="size threshold for blob detection (default=1)")
    parser.add_argument("--output_file", type=str, default="/srv/home/sghandian/data/nft/outputs/0fnaalid/nft_scores.csv", help="output file path")
    args = parser.parse_args()

    main(args)
