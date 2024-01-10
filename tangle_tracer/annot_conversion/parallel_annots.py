import os
from pathlib import Path
import pandas as pd

from numcodecs import Blosc
#set this for multiple processing
Blosc.use_threads = False

#distributed processing of masks
import dask
from dask.distributed import Client

from WSIAnnotation import WSIAnnotation
from ROIAnnotation import ROIAnnotation


def dask_agg_annots(zarr_path, cz_path, scale = 1.0, save = True):
    lazy_results = []
    filenames = os.listdir(zarr_path)

    for file in filenames:
        # if '02' == file.split('-')[0]: #skip validation set that has no .cz files
        #     continue
        imgPath = Path(zarr_path, file)
        annotPath = Path(cz_path, Path(file).with_suffix('.cz'))
        if not annotPath.is_file():#skip batch2 which has no .cz files
            continue
        wsiAnnot = dask.delayed(WSIAnnotation)(imgPath, annotPath, scale, save = save, 
                                               save_path = args.save_dir)
        lazy_results.append(wsiAnnot)
    return lazy_results

def dask_agg_annots_rois(metadata, save = True):
    lazy_results = []
    #files are prematched, only runs on images that are labeled as a result
    for _, row in metadata.iterrows():
        roi = list(row[['img_fp', 'label_fp', 'roi_corners']])
        roiAnnot = dask.delayed(ROIAnnotation)(*roi, save = save, save_path = args.save_dir)
        lazy_results.append(roiAnnot)
    return lazy_results

def main(args):
    assert args.input_type in ['wsis', 'rois']
    SCALE = args.scale #scale of images to work with

    #DASK cluster initialization
    with Client(threads_per_worker=args.threads_per_worker, n_workers=args.num_workers,
                silence_logs=50, memory_target_fraction = 0.8) as client:
        if args.input_type == 'wsis':
            zarrPath = args.zarr_path
            czPath = args.cz_path
            lazy_results = dask_agg_annots(zarrPath, czPath, scale = SCALE, save = True)
        elif args.input_type == 'rois':
            try:
                metadata = pd.read_csv(Path(args.metadata_path))
            except TypeError:
                print("If running on rois, must pass in a valid --metadata_path pointing to .csv file with columns: ['img_fp', 'label_fp', 'roi_corners']")
            lazy_results = dask_agg_annots_rois(metadata)
        dask.compute(lazy_results, scheduler = 'distributed')

    return None

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    #One of either 'wsis' or 'rois' for WSIAnnotation or ROIAnnotation objs
    parser.add_argument("--input_type", type=str, required=True)
    #Required parent save directory, must have permissions. Creates nested dirs as needed.
    parser.add_argument("--save_dir", type=str, required=True)

    #Use for study dataset (i.e. WSIs + CZ annots)
    parser.add_argument("--zarr_path", type=str)
    parser.add_argument("--cz_path", type=str)

    #Use for gutman dataset (i.e. roi input and metadata csv with 3 cols)
    parser.add_argument("--metadata_path", type=str)
    
    #Completely optional
    parser.add_argument("--scale", type=float, default = 1.0)
    parser.add_argument("--num_workers", type=int, default = 4)
    parser.add_argument("--threads_per_worker", type=int, default = 4)

    args = parser.parse_args()

    #run on all images
    main(args)

    print("################################")
    print("Completed Processing all WSIs!!!")
    print("################################")

    """
    ##-----EXAMPLE USAGE-----## NOTE: MUST RUN ALL SCRIPTS FROM NFT DIRECTORY (even annot_conversion scripts)
    
    python annot_conversion/parallel_annots.py --input_type='wsis' --save_dir=/scratch/sghandian/nft/model_input_v2/ \
      --zarr_path=/scratch/sghandian/nft/raw_data/wsis/zarr/scale_1.0/ --cz_path=/scratch/sghandian/nft/raw_data/annotations/updated_annots/ \
    
    python annot_conversion/parallel_annots.py --input_type='rois' --save_dir=/scratch/sghandian/nft/gutman_input/ \
      --metadata_path=/scratch/sghandian/nft/gutman_input/tables/metadata.csv --num_workers=16
    
    """
    