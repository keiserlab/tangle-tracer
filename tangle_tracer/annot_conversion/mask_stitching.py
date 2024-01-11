import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import shutil

import zarr
from numcodecs import Blosc
#set this for multiple processing
Blosc.use_threads = False

#distributed processing of masks
import dask
from dask.distributed import Client

from WSIAnnotation import load_annot


#slicing array properly using just original coords and ROI shape
def unpadding(array, coords: list, 
              rectangle_shape:tuple = (10680, 21236, 3),
              tile_size: tuple = (400, 400)):
    """
    :param array: numpy array
    :param coords: local coordinates of the upper left corner of bbox in ROI
    :return: padded array
    """
    xmax, ymax = coords[0] + tile_size[0], coords[1] + tile_size[1]

    a = np.abs(np.min([0, coords[0]]))
    b = np.abs(np.min([0, coords[1]]))
    
    pad_height = np.min([rectangle_shape[0] - xmax, tile_size[0]])
    pad_width =  np.min([rectangle_shape[1] - ymax, tile_size[1]])

    return array[slice(a, pad_height), slice(b, pad_width)]          


def stitch_wsi(wsiAnnot):
    """
    Takes in a wsiAnnot object and stitches masks of annotations into respective regions.
    Returns rectangle segmentation masks.
    """
    all_masks = {}
    for rect_idx, rect in wsiAnnot.img_regions.items():
        #create an empty mask
        base = np.zeros(rect['img'][:, :, 0].shape)
        for annot_name, annot in rect['bboxes'].items():
            xmin, ymin = annot['xmin'], annot['ymin']
            xmax, ymax = annot['xmax'], annot['ymax']
            #assign relevant region of empty mask to annot mask
            annot_mask = wsiAnnot.masks[annot_name][:,:,-1]
            try:
                #applies the UNION of base and annot_mask rather than overwriting
                base[slice(ymin, ymax), slice(xmin, xmax)] += annot_mask
            #unlikely to encounter these issues, but unpadding to make them tile_size
            except ValueError:
                #handle OOI errors
                sliced_mask = unpadding(annot_mask, [ymin, xmin],
                                       rectangle_shape = base.shape,
                                       tile_size = annot_mask.shape)
                base[slice(np.max([ymin, 0]), ymax),
                     slice(np.max([0, xmin]), xmax)] = sliced_mask
                
        #binarizes the output
        all_masks[rect_idx] = (base > 0) * 1.0 
        
    return all_masks

def stitch_roi(roiAnnot):
    """
    Takes in a roiAnnot object and stitches masks of annotations into respective regions.
    Returns a single roi ground truth segmentation mask.
    """
    #create an empty mask
    base = np.zeros(roiAnnot.roi[:, :, 0].shape)
    for annot_name, annot in roiAnnot.bbox_dims.items():
        xmin, ymin = annot['xmin'], annot['ymin']
        xmax, ymax = annot['xmax'], annot['ymax']
        
        #assign relevant region of empty mask to annot mask
        annot_mask = roiAnnot.masks[annot_name][:,:,-1]
        #applies the union of base and annot_mask rather than overwriting
        base[slice(ymin, ymax), slice(xmin, xmax)] += annot_mask
        #binarizes the output
        base = (base > 0) * 1.0 
        
    return base


def produce_masks(wsiAnnot_path, save_path, scale = 1.0, overwrite = True):
    #define save params
    COMPRESSOR = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
    tile_size = int(1024 * scale)
    
    #load in annot and stitch mask together
    name = wsiAnnot_path.stem
    wsiAnnot = load_annot(name, scale = scale, load_path=wsiAnnot_path.parent.parent.as_posix() + '/')
    try:
        stitched_masks = stitch_wsi(wsiAnnot)
    except:
        stitched_masks = {'rectangle_0': stitch_roi(wsiAnnot)}
    
    #save mask to disk via zarr
    for rect_idx, mask in tqdm(stitched_masks.items()):
        if len(stitched_masks.items()) == 1:
            wsi_path = Path(save_path, name)
        else:
            wsi_path = Path(save_path, name, rect_idx)
        
        #create zarr file with read/write access, output_shape, etc.
        synchronizer = zarr.ProcessSynchronizer(wsi_path.with_suffix('.sync'))
        store = zarr.DirectoryStore(wsi_path)
        #delete existing array if it exists
        if overwrite:
            try:
                shutil.rmtree(wsi_path)
            except FileNotFoundError:
                #cannot delete something that doesn't exist.
                pass
        try:
            z = zarr.array(mask, 
                            store=store,
                            chunks = (tile_size, tile_size),
                            write_empty_chunks=False, 
                            compressor = COMPRESSOR,
                            synchronizer = synchronizer)
        except zarr.errors.ContainsArrayError:
            print(f'Skipping {wsi_path} since it already exists.')
            continue
    del wsiAnnot
    return None

        
def agg_masks_dask(mask_function, wsiAnnot_path, save_path, scale = 1.0, overwrite = True):
    annot_list = os.listdir(wsiAnnot_path)
    lazy_results = []
    
    for annot in annot_list:
        # if 'Temporal' in annot or 'expert' in annot: #CHANGE CONDITION!
            # print(annot)
        lazy_results.append(dask.delayed(mask_function)(
                                Path(wsiAnnot_path, annot), save_path, 
                                scale = scale, overwrite = overwrite))
    return lazy_results

def agg_masks(mask_function, wsiAnnot_path, save_path, scale = 1.0, overwrite = True):
    annot_list = os.listdir(wsiAnnot_path)
    for annot in annot_list:
        mask_function(Path(wsiAnnot_path, annot), save_path, scale = scale, overwrite = overwrite)
    return True


def main(args):
    #general args input
    SCALE = args.scale
    DATA_DIR = Path(args.data_dir)

    #create read directories
    input_type_map = {"wsis": 'wsiAnnots', 'rois': 'roiAnnots'}
    assert args.input_type in input_type_map.keys() #make sure --input_type=wsis or rois
    
    INPUT_TYPE = input_type_map[args.input_type]
    WSI_ANNOT_PATH = Path(DATA_DIR, INPUT_TYPE, f'scale_{SCALE}')
    
    #create a directory if needed
    SAVE_PATH = Path(DATA_DIR, 'masks')
    os.makedirs(SAVE_PATH, exist_ok=True)

    if args.num_workers <= 1:
        agg_masks(produce_masks, WSI_ANNOT_PATH, SAVE_PATH, overwrite = True)
    else:
        with Client(threads_per_worker=args.threads_per_worker, n_workers=args.num_workers,
                    silence_logs=50, memory_target_fraction = 0.8) as client:
            lazy_masks = agg_masks_dask(produce_masks, WSI_ANNOT_PATH, SAVE_PATH, overwrite = True)
            dask.compute(lazy_masks, scheduler='distributed')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--input_type", type=str, default='wsis',
                        help = "Define whether to load in wsiAnnots or roiAnnots.")
    parser.add_argument("--scale", type=float, default = 1.0)
    parser.add_argument("--num_workers", type=int, default = 4)
    parser.add_argument("--threads_per_worker", type=int, default = 8)

    #add arg for annot_list or similar?
    args = parser.parse_args()
    main(args)
    print("Running on the folllowing WSIs/ROIs...") #need a checker to make sure files are outputted
    print("################################")
    print("Completed Stitching all WSIs/ROIs!!!")
    print("################################")
    """
    ##-----EXAMPLE USAGE-----## NOTE: MUST RUN ALL SCRIPTS FROM NFT DIRECTORY (even annot_conversion scripts)
    
    python annot_conversion/mask_stitching.py --data_dir=/scratch/sghandian/nft/model_input_v2/ \
        --input_type='wsis' --num_workers=8
    
    python annot_conversion/mask_stitching.py --data_dir=/scratch/sghandian/nft/gutman_input/ \
        --input_type='rois' --num_workers=32          
    
    """