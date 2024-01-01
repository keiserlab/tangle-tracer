import os, gc
from pathlib import Path
import pandas as pd

import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from modules import NFTDetector
from nft_datasets import TileDataset

def main(args):
    pl.seed_everything(42)

    #set the stride and model names, used to create save_path
    STRIDE = int(args.stride) #param
    MODEL_PATH = args.model_path
    # MODEL_NAME = f"{MODEL_PATH.split('/')[-3]}_{MODEL_PATH.split('/')[-1]}"
    MODEL_NAME = args.model_name
    SAVE_PATH = Path(args.save_path, MODEL_NAME, f"stride_{STRIDE}")

    os.makedirs(SAVE_PATH, exist_ok = True)
    
    # DEFINE number of classes we're predicting
    NUM_CLASSES = 1  
    
    parent_dir = (Path(__file__).resolve().parent.parent)
    #set default table path for this study
    if args.table_path:
        table_path = args.table_path
    else:
        table_path = Path(parent_dir, 'data/tables/study_split.pickle')        

    #load in included files (study_split.pickle has train/test splits)
    batch1_df = pd.read_pickle(table_path)
    batch2_df_path = Path(parent_dir, 'data/tables/NFT_Case_List_Batch2.csv')
    batch2_df = pd.read_csv(batch2_df_path)
    
    paths = {
        'csv_path': table_path,
        'read_path': args.read_path,
        'save_path': SAVE_PATH,
    }

    if args.rois == True:
        print(args.rois, "Running on ROIS ONLY!!")
        #run on ROIs by including ROI_PATH in paths dict!
        paths['save_path'] = args.save_path
        paths['roi_path'] = args.read_path #TileDataset uses roi_path instead of read_path if it exists

    
    if args.wsis == ['batch1']:
        print("Running on batch 1 wsis")
        master_wsis = list(batch1_df['CASE_ID'].unique())
    
    elif args.wsis == ['test']:
        print("Running on test wsis")
        master_wsis = list(batch1_df[batch1_df['train/test'] == 'test']['Random ID'].unique())
    
    elif args.wsis == ['val']:
        print("Running on val wsis")
        master_wsis = ['1-209-Temporal_AT8', '1-466-Temporal_AT8', '1-695-Temporal_AT8',
                       '1-755-Temporal_AT8', '1-838-Temporal_AT8']
    
    elif args.wsis == ['batch2']:
        print("Running on batch 2 wsis")
        master_wsis = list(batch2_df['CASE_ID'].unique())
    
    elif args.table_path and args.wsis == ['all']:
    #get list of WSIs with columns "CASE_ID" and "train/test"; USE FOR CUSTOM DATASET
        try:
            wsi_df = pd.read_csv(args.table_path)
        except:
            wsi_df = pd.read_pickle(args.table_path)
        master_wsis = list(wsi_df['CASE_ID'].unique())

    elif args.wsis == ['all']:
        batch1_wsis = list(batch1_df['CASE_ID'].unique())
        batch2_wsis = list(batch2_df['CASE_ID'].unique())
        master_wsis = (batch1_wsis + batch2_wsis)
        print("Doing batch1 and batch2!!")
        print(master_wsis)

    else:
        #this can be used to filter a novel set of wsis
        master_wsis = args.wsis

    #load model with checkpoint
    model = NFTDetector().load_from_checkpoint(MODEL_PATH)
    #have to override the save_path that was in there (None), for inference only
    model.save_dir = paths['save_path']

    #define dataloader params
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers 
    IMAGENET_NORM = args.imagenet

    for wsi in master_wsis:

        gc.collect()

        trainer = Trainer.from_argparse_args(args)
        print(f"Creating dataset for {wsi}!")
        try:
            dataset = TileDataset(paths, #pass in paths dict
                              stride = STRIDE,
                              num_classes = NUM_CLASSES,
                              save = True,
                              #this is a list input, remove to run on entire wsi list or pass as an arg --wsi_filter=False 
                              wsi_filter = [wsi],
                              imagenet_norm = IMAGENET_NORM)
        except ValueError:
            print("Starting on batch 2..., changing WSI CSV read path")
            paths['csv_path'] = batch2_df_path
            dataset = TileDataset(paths, #pass in paths dict
                              stride = STRIDE,
                              num_classes = NUM_CLASSES,
                              save = True,
                              #this is a list input, remove to run on entire wsi list or pass as an arg --wsi_filter=False 
                              wsi_filter = [wsi],
                              imagenet_norm = IMAGENET_NORM)
            
        dataloader = data.DataLoader(dataset,
                                     shuffle = False,
                                     batch_size = BATCH_SIZE,
                                     num_workers = NUM_WORKERS,
                                     persistent_workers=False,
                                     )

        print(f'Starting inference on {wsi}')
        trainer.predict(model, dataloader) #currently rounding in predict_step

        #release memory?
        del dataset
        del dataloader
        del trainer
        gc.collect()



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # add program level args
    parser.add_argument("--read_path", type=str, required=True)
    parent_dir = (Path(__file__).resolve().parent) #set default table_path
    parser.add_argument("--table_path", type=str, #might need to be refactored?
                        help="""Path to a csv/pickle that contains a CASE_ID column outlining 
                                which wsis to run on. If running on study WSIs, don't need to pass this in.
                                Just pass in correct arg to --wsis.""")
    parser.add_argument("--model_path", type=str, required=True) #need to pass this arg in!
    parser.add_argument("--save_path", type=str, required=True)

    parser.add_argument("--model_name", type=str, default='NFTDetector')
    parser.add_argument("--wsis", nargs='+', default = ['test']) #by default, run on subset of wsis
    parser.add_argument("--rois", default = False, action="store_true", 
                        help = "Flag to run on ROIs or WSI itself. Must be run on batch 1.") #dont run on ROIs by default, run on entire WSI
    
    parser.add_argument("--stride", type=int, default = 1024) #stride needs to be 1024 (tile size) at the moment
    parser.add_argument("--imagenet", default = True, action="store_true", 
                        help = "Flag to use imagenet normalization, check if model was imagenet normalized.") #use a config to feed this into train.py and inference.py and ensure the same param

    # add all trainer options
    # parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--batch_size", type=int, default = 12)
    parser.add_argument("--num_workers", type=int, default = 12)
    parser.add_argument("--accelerator", default = 'gpu')
    parser.add_argument("--devices", default = [1])

    
    #begin program
    args = parser.parse_args()
    main(args)

    """
    # To run on all WSIs with best model, run the following. This takes ~30 minutes/WSI/GPU. This does NOT run on ROIs only!
    python tangle_tracer/inference.py \
        --read_path=/scratch/sghandian/nft/raw_data/wsis/zarr/scale_1.0/ \
        --model_path=/scratch/sghandian/nft/output/logs/NFTDetector/0fnaalid/checkpoints/NFT_trial-epoch=83-valid_loss=0.12.ckpt \
        --save_path=/scratch/sghandian/nft/wsi_heatmaps/ \
        --devices=1 --wsis=all 

    #NOTE: This requires a "recent" run of parallel_annots.py such that pickled WSIAnnots do not have different import strxr than current codebase.
    # To run on all ROIs with best model, run the following. This takes ~1 minute/ROI/GPU. 
    # This does not create WSI-level segmentation, but is well suited for constructing agreement maps.
    python tangle_tracer/inference.py \
        --read_path=/scratch/sghandian/nft/model_input_v2/images/ \
        --table_path=/scratch/sghandian/nft/model_input/tables/NFT_Case_List_Batch1.csv \
        --model_path=/scratch/sghandian/nft/output/logs/NFTDetector/0fnaalid/checkpoints/NFT_trial-epoch=83-valid_loss=0.12.ckpt \
        --save_path=/scratch/sghandian/nft/output/roi_heatmaps/ \
        --wsis=all --devices=1 --rois

    """

