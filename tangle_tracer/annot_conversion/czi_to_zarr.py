import os, gc
from pathlib import Path

import numpy as np
import cv2
import zarr
from numcodecs import Blosc
import czifile
# from aicspylibczi import CziFile

from tqdm import tqdm
from matplotlib import pyplot as plt

    

def czi_to_numpy(czi_file, scale = 1.0):
    #pass in a czi file
    np_img = np.squeeze(czi_file)
    #check if there is an extra dimension (multiple views)
    if scale < 1.0:
        scaled_dims = tuple([int(dim * scale) if dim > 5 else int(dim) for dim in np_img.shape][:2][::-1])
        np_img = cv2.resize(np_img, dsize = scaled_dims)

    return np_img

def np_to_zarr(np_img):
    z = zarr.array(np_img, chunks = (5000, 5000), dtype = 'uint8')
    return z    

def zarr_to_np(zarr_img):
    np_img = zarr_img[:]
    return np_img

def czi_to_zarr(czi_file, scale = 1.0):
    np_img = czi_to_numpy(czi_file, scale)
    z = np_to_zarr(np_img)
    return z

def load_czi(cziPath):
    return np.squeeze(czifile.imread(cziPath))

def get_section(img, coords, tile_size = 256, display = False):
    """
    Returns an orthogonal slice of a zarr or numpy image given 
    the coordinates of the top left point, and either 1 or 2 dimensional
    tile_size indicating the width and height respectively.

    Input:
        img (zarr or np.array) - Image from which to slice
        coords (tuple or array) - Width and Height, can be a single int,
                                  in which case it slices a square image.
        display (bool) - Flag for whether to display returned slice.
    """
    x, y = coords
    if type(tile_size) == int:
        tile_size = (tile_size, tile_size)
    try:
        img_slice = img.oindex[slice(x, x + tile_size[0]), slice(y, y + tile_size[1])]
    except AttributeError:
        img_slice = img[slice(x, x + tile_size[0]), slice(y, y + tile_size[1])]
    if display:
        plt.figure(figsize=(10, 10))
        plt.imshow(img_slice)
        plt.show()
    
    return img_slice

def save_to_zarr(filepaths, output_dir, scale = 1.0, chunks = (5000, 5000)):
    COMPRESSOR = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
    SAVE_ARRAY_KWARGS = dict(compressor=COMPRESSOR, chunks = chunks, write_empty_chunks = False) 
    
    #change output_dir depending on input scale
    output_dir = Path(output_dir, f"scale_{scale}/")

    pbar = tqdm(total=len(filepaths), desc = 'Converting czi to zarr')
    
    #NOTE: we may want to alter this function to load tiles instead of entire CZI imgs
    #due to memory concerns
    for czi_path in filepaths:
        #load in czi file
        czi_img = load_czi(Path(czi_path))
        filename = Path(czi_path).stem
        print(f"Shape of {filename}: {czi_img.shape}")

        #check if there are multiple scenes (less than 5)
        np_imgs = []
        if czi_img.shape[0] <= 5:
            for idx in range(czi_img.shape[0]):
                scene = np.squeeze(czi_img[idx, :])
                #split into separate arrays
                np_imgs.append(scene)
                del scene
        else: #if img has one scene
            np_imgs.append(czi_img)

        del czi_img
        gc.collect()

        np_imgs = [czi_to_numpy(np_img, scale) for np_img in np_imgs]

        #MOST of the time this will be a single scene
        for i, scene in enumerate(np_imgs):
            #apply a different naming scheme to multi-scene case
            if len(np_imgs) > 1:
                wsi_name = filename.stem + f'_s{i}'
                filename = Path(wsi_name).with_suffix('.zarr')
            else:
                filename = Path(filename).with_suffix('.zarr')

            zarr_output_name = Path(output_dir, filename)
            print(f'Saving zarr array: {filename}', f"shape of numpy array: {scene.shape}", zarr_output_name)
            zarr.save_array(
                zarr_output_name, 
                scene, 
                **SAVE_ARRAY_KWARGS)
        
            del scene
            gc.collect()

        del np_imgs
        gc.collect()
        pbar.update()



def main(args):
    READ_PATH = args.read_path
    WRITE_PATH = args.write_path
    filepaths = [Path(READ_PATH, filepath) for filepath in os.listdir(READ_PATH)]
    save_to_zarr(filepaths, WRITE_PATH, scale = args.scale)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # add program level args
    parser.add_argument("--read_path", type=str, default = '/scratch/sghandian/nft/raw_data/wsis/Batch_02_CZI Files')
    parser.add_argument("--write_path", type=str, default = '/scratch/sghandian/nft/raw_data/wsis/zarr')
    parser.add_argument("--scale", type=float, default = 1.0)   
    args = parser.parse_args()

    main(args)
    
    """
    Example Usage:

    # To convert all WSIs at given read_path to zarr, run the following. This takes ~10 minutes/WSI and significant RAM utilization as the entire WSI is loaded into memory. 
    python annot_conversion/czi_to_zarr.py --read_path='/scratch/sghandian/nft/raw_data/wsis/Batch_02_CZI Files' \
          --write_path='/scratch/sghandian/nft/raw_data/wsis/zarr'
   
   """