import os, shutil
import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union


import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit

import kornia as K
from kornia import image_to_tensor

from torchvision import transforms
import pytorch_lightning as pl

import zarr
from numcodecs import Blosc
#set this for multiple processing
Blosc.use_threads = False

from annot_conversion.WSIAnnotation import load_annot


"""
Datamodule that defines overall structure of dataset used for training.
"""

class ROIDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            csv_file: str = 'NFT_Case_List_Batch1.csv', #name of csv file containing WSI names in the form: Path(data_dir, tables, csv_file)
            batch_size: int = 32,
            num_workers: int = 4,
            imagenet_norm: bool = False, #whether to normalize the outputs by the imagenet normalization values (i.e. loading a model pretrained on imagenet)
            border_padding: int = 1536, #used in ROIDataset and NFT calculation
            scale: float = 1.0, #scale at which to load in pre-generated wsiAnnot
            train_filter_proportion: float = 1.0, #how much of the cleaned training set WSIs to use
            train_tile_filter_proportion: float = 1.0, #proportion of the positive tiles (contains atleast some of an NFT) in the training set that should be kept
            train_tile_ablate_proportion: float = 0, #how many of the positive tiles (contains atleast some of an NFT) in the training set should be ablated (ground truth mask set to 0)
            k: int = 0, #determines fold, effectively unused in this implementation as we did not end up doing cross-validation
            num_folds: int = 10, #how many folds to use for cross-val; not used in this implementation besides generating valid split of WSIs for train/val/test
        ):

        super().__init__()
        self.data_dir = data_dir 
        self.csv_dir = Path(self.data_dir, 'tables', csv_file)

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)


        self.train_transform  = K.augmentation.AugmentationSequential(
                                    K.augmentation.RandomHorizontalFlip(p=0.8),
                                    K.augmentation.RandomVerticalFlip(p=0.8),
                                    K.augmentation.RandomAffine(p=0.5, 
                                                                degrees=(-45., 45.), 
                                                                shear=20,
                                                                scale=(0.7, 1.3)
                                                                ),                                                     
                                    K.augmentation.ColorJiggle(p=0.6, 
                                                            saturation=(0.75, 1.25),
                                                            hue=(-0.15, 0.15), # use Stain-Augmentation instead
                                                            brightness=(0.7, 1.3),
                                                            contrast=(0.7, 1.3) 
                                                            ),                                
                                    K.augmentation.RandomGaussianBlur(p=0.25, 
                                                                    kernel_size=(3,3),
                                                                    sigma=(0.2, 2.0)
                                                                    ),
                                    # internal filter that determines which transform to apply to each key
                                    data_keys=['input', 'mask']
                                    )

        self.test_transform = K.augmentation.AugmentationSequential(
                                    K.augmentation.RandomHorizontalFlip(p=0.000),
                                data_keys=['input', 'mask']
                                )  

                                                  

    def prepare_data(self):
        """
        Load in csv with CASE_ID + UNIQUE_ID (ROI), and assign to train or test. Save as: '{DATA_DIR}/tables/split.csv'
        """
        # construct csvs to assign train/test to rectangles
        df = pd.read_csv(Path(self.csv_dir))
        #fix the case ID, HOTFIX!, add `1-`, go BACK to csv and fix this
        # if df.loc[0, 'CASE_ID'][:2] != '1-':
            # df['CASE_ID'] = df['CASE_ID'].apply(lambda wsi_name: '1-' + wsi_name)

        #filter problematic WSIs out
        df = df[~df['SCENE'].isin(['0', '1'])] 
        df = df[df['CASE_ID'] != '1-473-Temporal_AT8']#.copy() #need to add this into dataset later
        #get rid of rectangle_2 for WSI 1-343!
        df = df[~((df['CASE_ID'] == '1-343-Temporal_AT8') & (df['ROI'] == 'rectangle_2'))]
        #filter out the _s1 or _s0 WSI which has mostly black space ? check out ROIs for those.
        df = df[(df['CASE_ID'] != '1-960-Temporal_AT8')]
        #reset df indices
        df = df.reset_index(drop = True)

        #add column with positive NFT locations for each ROI
        def attach_nfts(df, scale=1.0):
            all_nfts = []
            for _, row in df.iterrows():
                # try:
                wsiAnnot = load_annot(row['CASE_ID'], load_path = Path(self.data_dir, 'wsiAnnots'), scale=scale)
                # except zarr.errors.ArrayNotFoundError:
                #     print("ERROR!", row['CASE_ID'], row['ROI'])
                nfts = wsiAnnot.img_regions[row['ROI']].get('bboxes')
                coords = list(nfts.values())
                all_nfts.append(coords)

            df['NFTs'] = all_nfts
            return df

        df = attach_nfts(df, self.hparams.scale) #load in annots and add NFTs column
        
        #create empty col
        df['train/test'] = None

        #split by CASE_ID to get train and test
        gs = GroupShuffleSplit(n_splits=self.hparams.num_folds, train_size=.7, random_state=42) #42 is default!
        train_idx, test_idx = [i for i in gs.split(df, y = None, groups = df[['CASE_ID']])][0] 

        #assign train and test to each row
        df.loc[train_idx, 'train/test'] = 'train' #KEY ERROR from train_idx
        df.loc[test_idx, 'train/test'] = 'test'

        df['NFT_centers'] = df['NFTs'].apply(lambda ls: [
                                (nft.get('xmin') + (nft.get('width') // 2), #doesn't need border padding
                                 nft.get('ymin') + (nft.get('height') // 2)) 
                                 for nft in ls])

        missing_prop = (1 - self.hparams.train_filter_proportion)
        print(f"NOTE: FILTERING OUT {missing_prop * 100}% WSIs for EXPERIMENTATION. Visit nft_datasets.py to revert.")
        #filter out some training WSIs
        num_to_disclude = int(df['CASE_ID'].nunique() * missing_prop)
        discluded_wsis = df[df['train/test'] == 'train']['CASE_ID'].unique()[:num_to_disclude]
        print("DISCLUDING THESE WSIs:", discluded_wsis)
        df = df[~df['CASE_ID'].isin(discluded_wsis)]
        print("Training set size:", df['CASE_ID'].nunique())

        if self.hparams.train_filter_proportion == 1.0:
            df.to_pickle(Path(self.data_dir, 'tables', f'split.pickle'))
        else:
            df.to_pickle(Path(self.data_dir, 'tables', f'split_{self.hparams.train_filter_proportion}.pickle'))


    def setup(self, stage: Optional[str] = None):
        #Maybe we shouldn't do this split in setup?

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.ROI_train = ROIDataset(self.data_dir, train=True, 
                                        transform=self.train_transform,
                                        border_padding=self.hparams.border_padding, 
                                        random_sampling = True,
                                        split_proportion=self.hparams.train_filter_proportion)
            train_df = self.ROI_train.data
            
            gs = GroupShuffleSplit(n_splits=self.hparams.num_folds, train_size=.7, random_state=42) #changed state!
            train_idx, val_idx = [i for i in gs.split(train_df, y = None, groups = train_df[['CASE_ID']])][self.hparams.k]   

            self.ROI_val = ROIDataset(self.data_dir, train=True, transform=self.test_transform, 
                                      border_padding=self.hparams.border_padding,
                                      random_sampling = False, 
                                      split_proportion=self.hparams.train_filter_proportion)

            #filter both datasets with train and val indices respectively
            self.ROI_train.data = self.ROI_train.data.loc[train_idx].reset_index(drop = True) #KEY ERROR
            self.ROI_train.tiles = self.ROI_train.tiles[self.ROI_train.tiles['CASE_ID'].isin(self.ROI_train.data['CASE_ID'])].reset_index(drop = True)

            #randomly remove some positive annotations for study purposes
            def remove_pos(df: pd.DataFrame, remove_proportion: float):
                """
                Removes a random subset of rows from the input DataFrame where "pos" equals 1, based on the specified proportion.
                
                Args:
                    df (pd.DataFrame): The input DataFrame containing the "pos" column.
                    remove_proportion (float): The proportion of rows to remove where "pos" equals 1, ranging from 0 to 1.
                
                Returns:
                    pd.DataFrame: The modified DataFrame with random rows removed and index reset.
                """
                # Set a random seed for reproducibility
                np.random.seed(42)
                
                # Filter the DataFrame to get only the rows where "pos" equals 1
                pos_rows = df[df['pos'] == 1]
                
                # Calculate the number of rows to remove based on the remove_proportion
                num_rows_to_remove = int(len(pos_rows) * remove_proportion)
                
                # If num_rows_to_remove is greater than 0, randomly select and remove the rows using NumPy's random.choice
                if num_rows_to_remove > 0:
                    rows_to_remove = np.random.choice(pos_rows.index, num_rows_to_remove, replace=False)
                    df = df.drop(rows_to_remove)
                
                # Reset the index to ensure no skipped values
                df = df.reset_index(drop=True)
                
                return df

            def set_label_to_zero(df: pd.DataFrame, set_proportion: float):
                """
                Set "pos" column value to 0 for a random subset of rows with "pos" equal to 1.

                Args:
                    df (pd.DataFrame): Input DataFrame with "pos" column.
                    set_proportion (float): Proportion of "pos" = 1 rows to set to 0. Value should be in [0, 1].

                Returns:
                    pd.DataFrame: DataFrame with "pos" column updated.
                """
                # Copy the input DataFrame to avoid modifying the original
                df = df.copy()

                # Get the rows where "pos" = 1
                pos_1_rows = df[df['pos'] == 1]

                # Calculate the number of rows to set to 0
                num_rows_to_set = int(len(pos_1_rows) * set_proportion)

                # Randomly shuffle the index of pos_1_rows
                shuffled_index = np.random.permutation(pos_1_rows.index)

                # Set the "pos" column value to 0 for the randomly selected rows
                df.loc[shuffled_index[:num_rows_to_set], 'pos'] = 0

                return df
                
            if self.hparams.train_tile_filter_proportion != 1.0:
                print(f"Using {(self.hparams.train_tile_filter_proportion) * 100}% of positive training tiles!")
                #first approach removes those training examples from the datset entirely (img and mask)
                self.ROI_train.tiles = remove_pos(self.ROI_train.tiles, remove_proportion= 1 - self.hparams.train_tile_filter_proportion)
            if self.hparams.train_tile_ablate_proportion != 0:
                print(f"Ablating {(self.hparams.train_tile_ablate_proportion) * 100}% of positive training tiles!")
                #second approach removes the positive label from this tile, mask signal removed in __getitem__
                self.ROI_train.tiles = set_label_to_zero(self.ROI_train.tiles, set_proportion= self.hparams.train_tile_ablate_proportion)

            self.ROI_val.data = self.ROI_val.data.loc[val_idx].reset_index(drop = True)
            self.ROI_val.tiles = self.ROI_val.tiles[self.ROI_val.tiles['CASE_ID'].isin(self.ROI_val.data['CASE_ID'])].reset_index(drop = True)
            


        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.ROI_test = ROIDataset(self.data_dir, train=False, transform=self.test_transform,
                                       border_padding=self.hparams.border_padding,
                                       random_sampling = False, 
                                       split_proportion=self.hparams.train_filter_proportion)

        # if stage == "predict" or stage is None:
        #     self.ROI_predict = ROIDataset(self.data_dir, train=False, transform=self.test_transform, 
        #                                   random_sampling = False)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        y = y.unsqueeze(1) #make it look like a 3 channel image
        if self.trainer.training:
            x,y  = self.train_transform(x, y)  # => we perform GPU/Batched data augmentation
        else:
            x,y = self.test_transform(x, y)
            # y = y.permute(1,0,2,3) #could change if i add different validation augs

        if self.hparams.imagenet_norm:
            x = K.enhance.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)

        # x, y = x.float().squeeze(), y.float()#.permute(1,0,2,3) #squeeze is default, try without squeeze for resnet50
        x, y = x.float(), y.float()#.permute(1,0,2,3)

        return x, y

    def get_WRS(self, y_train, neg_weight:float = 0.5, pos_weight:float = 0.5):
        """
        params:
            y_train - Vector representing labels of 0s and 1s for training set.
            neg_weight - Float weight representing likelihood that a given sample will be negative.
            pos_weight - Float weight representing likelihood that a given sample will be positive.
        """
        try:
            y_train = y_train.copy()
        except:
            y_train = y_train.clone()
            
        assert pos_weight + neg_weight == 1
        class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
        
        y_train[np.where(y_train == 0)[0]] = neg_weight * class_sample_count[1]
        y_train[np.where(y_train == 1)[0]] = pos_weight * class_sample_count[0]
        
        return data.WeightedRandomSampler(torch.tensor(y_train).type('torch.DoubleTensor'), len(y_train), replacement = True)

    def train_dataloader(self):
        WRS =  self.get_WRS(self.ROI_train.tiles['pos'], neg_weight=0.5, pos_weight = 0.5)
        return DataLoader(self.ROI_train, batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers,# shuffle = True, mutually exclusive with sampler
                          pin_memory=True, sampler = WRS)

    def val_dataloader(self):
        return DataLoader(self.ROI_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle = False, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.ROI_test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True)

    # def predict_dataloader(self):
    #     return DataLoader(self.ROI_predict, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

class GeneralROIDataModule(ROIDataModule):
    def __init__(self, 
                data_dir: str, 
                csv_file: str = 'metadata.csv',
                batch_size: int = 32,
                k: int = 0, 
                num_folds: int = 10, 
                num_workers: int = 4, 
                imagenet_norm: bool = False, 
                train_filter_proportion: float = 1,
                train_tile_filter_proportion: float = 1,
                train_tile_ablate_proportion: float = 0,
                border_padding: int = 1536):
        super().__init__(data_dir, csv_file, batch_size, k,
                        num_folds, num_workers, imagenet_norm,
                        train_filter_proportion, 
                        train_tile_filter_proportion, 
                        train_tile_ablate_proportion,
                        border_padding)
    
    def prepare_data(self):
        """
        Load in csv with CASE_ID + UNIQUE_ID (ROI), and assign to train or test. Save as: '{DATA_DIR}/tables/split.csv'
        """
        # construct csvs to assign train/test to rectangles
        df = pd.read_csv(Path(self.csv_dir))
        #add empty reference for each ROI
        df['ROI'] = [''] * df.shape[0]
        #fix the case ID, HOTFIX!, add `1-`, go BACK to csv and fix this
        df['CASE_ID'] = df['img_fp'].apply(lambda wsi_name: Path(wsi_name).stem)
        #reset df indices
        df = df.reset_index(drop = True)

        #add column with positive NFT locations for each ROI
        def attach_nfts(df):
            all_nfts = []
            for _, row in df.iterrows():
                wsiAnnot = load_annot(row['CASE_ID'], load_path=Path(self.data_dir, 'roiAnnots').as_posix() + '/')
                nfts = wsiAnnot.bbox_dims
                coords = list(nfts.values())
                all_nfts.append(coords)

            df['NFTs'] = all_nfts
            return df

        df = attach_nfts(df) #load in annots and add NFTs column
        
        #create empty col
        df['train/test'] = None

        #split by CASE_ID to get train and test
        gs = GroupShuffleSplit(n_splits=self.hparams.num_folds, train_size=.7, random_state=42) #42 is default!
        train_idx, test_idx = [i for i in gs.split(df, y = None, groups = df[['case']])][0] 

        #assign train and test to each row
        df.loc[train_idx, 'train/test'] = 'train' #KEY ERROR from train_idx
        df.loc[test_idx, 'train/test'] = 'test'

        df['NFT_centers'] = df['NFTs'].apply(lambda ls: [
                                (nft.get('xmin') + (nft.get('width') // 2), #doesn't need border padding
                                 nft.get('ymin') + (nft.get('height') // 2)) 
                                 for nft in ls])

        missing_prop = (1 - self.hparams.train_filter_proportion)
        print(f"NOTE: FILTERING OUT {missing_prop * 100}% WSIs for EXPERIMENTATION. Visit nft_datasets.py to revert.")
        #filter out some training WSIs
        num_to_disclude = int(df['case'].nunique() * missing_prop)
        discluded_wsis = df[df['train/test'] == 'train']['case'].unique()[:num_to_disclude]
        print("DISCLUDING THESE WSIs:", discluded_wsis)
        df = df[~df['case'].isin(discluded_wsis)]
        print("Training set num ROIs and cases:", df['CASE_ID'].nunique(), df['case'].nunique())

        if self.hparams.train_filter_proportion == 1.0:
            df.to_pickle(Path(self.data_dir, 'tables', f'split.pickle'))
        else:
            df.to_pickle(Path(self.data_dir, 'tables', f'split_{self.hparams.train_filter_proportion}.pickle'))


class ROIDataset(data.Dataset):
    def __init__(
             self, 
             data_dir : Union[Path, str],
             transform,
             train : bool = True, 
             samples_per_ROI : int = 150, #should be changed to atleast 210, that gives complete coverage
             tile_size : int = 1024, #must be divisible by 32
             random_sampling: bool = True, #random sample for training, test/val should be False but can customize
             border_padding: int = 1536, #number of pixels that each ROI is padded by on each side
             split_proportion:float = 1.0 #how much of the data to use for training, must be preprocessed
             ):
        #inherit data.Dataset attrs
        super().__init__()

        #load in pickle/csv table with train/test split at ROI level
        self.data_dir = data_dir
        if split_proportion == 1.0:
            self.data = pd.read_pickle(Path(data_dir, 'tables', 'split.pickle')) #otherwise AttributeError when loading in
        else:
            self.data = pd.read_pickle(Path(data_dir, 'tables', f'split_{split_proportion}.pickle'))

        #assign transform to class variable, must handle img and mask together
        self.transform = transform
        #assign the rest of the params as class variables
        self.train = train
        self.samples_per_ROI = samples_per_ROI
        self.tile_size = tile_size
        self.random_sampling = random_sampling
        self.border_padding = border_padding

        #filter for the correct WSIs according to split.csv
        if self.train:
            print("Loading train/val set...")
            self.data = self.data[self.data['train/test'] == 'train'].reset_index(drop = True)

        else:
            print("Loading test set...")
            self.data = self.data[self.data['train/test'] == 'test'].reset_index(drop = True)

        #display names of WSIs
        print(list(self.data['CASE_ID'].unique()))

        #create a sampler for each ROI
        self._create_samplers()
        #choose stride (e.g. overlapping inputs or not?) could be tile_size//2 but would have to evaluate assumptions...
        self.tiles = self._vectorized_tile_df(stride = self.tile_size)
        self._attach_tile_labels()

    def _attach_tile_labels(self):
        #attach label of positive (1) or negative (0) if tile contains an NFT point annotation
        def attach_label(row, tile_size = self.tile_size):
            min_tile_coord = row.read_y, row.read_x
            max_tile_coord = row.read_y + tile_size, row.read_x + tile_size
            for nft in row.NFT_centers:
                if (nft[0] < max_tile_coord[0] and nft[1] < max_tile_coord[1] 
                    and nft[0] > min_tile_coord[0] and nft[1] > min_tile_coord[1]):
                    return True
            return False   

        self.tiles = self.tiles.merge(self.data[['CASE_ID', 'ROI', 'NFT_centers']], on = ['CASE_ID', 'ROI'])
        bools = self.tiles.apply(attach_label, axis = 1)
        self.tiles.loc[bools, 'pos'] = 1
        self.tiles.loc[~bools, 'pos'] = 0
    
    def __len__(self):
        return self.tiles.shape[0]

    def _vectorized_tile_df(self, stride = None):
        """
        Creates a single dataframe with each row representing a valid tile within a WSI.
        Encapsulates all tiles in the dataset, across an arbitrary number of WSIs.
        """
        dfs = []
        if not stride:
            stride = self.tile_size
        for _, roi in self.data.iterrows():
            #generate a meshgrid with shape of input
            x = np.arange(0 + self.border_padding, #start
                          roi['img_shape'][0] - self.border_padding, #end
                          stride) #stride
            y = np.arange(0 + self.border_padding, #start
                          roi['img_shape'][1] - self.border_padding, #end
                          stride) #stride                                                    
            i_coords, j_coords = np.meshgrid(x, y, indexing = 'ij')

            #construct dictionary with which to instantiate df
            out_dict = {
                        'read_x': i_coords.flatten(),
                        'read_y': j_coords.flatten(),
                        }
            df = pd.DataFrame(out_dict, dtype = np.int32)

            df['CASE_ID'] = [roi['CASE_ID']] * df.shape[0] 
            df['ROI'] = [roi['ROI']] * df.shape[0] 

            dfs.append(df)
        return pd.concat(dfs, ignore_index = True)

    def __getitem__(self, idx):
        #random sampling handled by WRS, just get tile from self.tiles using idx
        tile = self.tiles.loc[idx]
        coords = (tile['read_x'], tile['read_y']) #changed order...
        tile_ROI = self.data[ (self.data['CASE_ID'] == tile['CASE_ID'])
                            & (self.data['ROI'] == tile['ROI'])].iloc[0]
        img, mask = self._load_sample(tile_ROI, coords)

        #TEMPORARY: If the label is 0, set the mask = 0 with correct shape
        #This is for the ablation experiment
        if tile['pos'] != 1:
            mask = torch.zeros(mask.shape)

        return img, mask


    def _create_samplers(self):
        #need to init sampler for each row
        self.data['coordGenerator'] = None
        #create img and mask path variables
        self.data['img_path'] = None
        self.data['mask_path'] = None

        #suboptimal speed, but simple (using multiple cols makes vectorization annoying)
        for i, row in self.data.iterrows():
            #query wsi_name and ROI_name
            wsi_name = row['CASE_ID'] 
            roi_name = row['ROI']
            #define paths to open zarr arrays easier
            self.data.loc[i, 'img_path'] = Path(self.data_dir, 'images',
                             wsi_name, roi_name)
            self.data.loc[i, 'mask_path'] = Path(self.data_dir, 'annotations',
                             wsi_name, roi_name)
        #create references to img and mask zarr arrays in table
        self.data['img_zarr'] = self.data['img_path'].apply(lambda img_path: zarr.open(img_path, mode='r'))
        self.data['mask_zarr'] = self.data['mask_path'].apply(lambda mask_path: zarr.open(mask_path, mode='r'))
        #save the shape of this ROI
        self.data['img_shape'] = self.data['img_zarr'].apply(lambda z : z.shape)
        self.data['mask_shape'] = self.data['mask_zarr'].apply(lambda z: z.shape) #use this to assert two cols are the same
        #filter for erroneous rows
        mismatch_df = self.data[self.data['img_shape'].apply(lambda dims: (dims[0], dims[1])) != self.data['mask_shape']]

        assert len(mismatch_df) == 0 #run mask_stitching.py!

        #check if random_sampling, if not then do strxr coords
        if self.random_sampling:
            self.data['coordGenerator'] = self.data.index.to_series().apply(lambda i :
                                                                np.random.default_rng(seed = i))
        else:
            #create a structured coord generator for all rows, delegate choice of using it to __getitem__
            self.data['coordGenerator'] = self.data['img_shape'].apply(lambda dims: 
                                                                product(range(0 + self.border_padding, #start
                                                                             dims[0] - self.border_padding, #end
                                                                             int(self.tile_size)), #stride
                                                                        range(0 + self.border_padding,
                                                                             dims[1] - self.border_padding, 
                                                                             int(self.tile_size))
                                                                        )
                                                                    )

    def _load_sample(self, ROI, coords):
        y, x = coords
        y, x = int(y), int(x)

        #get rid of - indexing
        in_y, in_x = np.max([0, y]), np.max([0, x])

        img = ROI['img_zarr'].oindex[slice(in_y, in_y + self.tile_size), slice(in_x, in_x + self.tile_size)]
        mask = ROI['mask_zarr'].oindex[slice(in_y, in_y + self.tile_size), slice(in_x, in_x + self.tile_size)]
        
        # if img.shape != (1024,1024,3):
        #     print(img.shape, in_y, in_x)
        # # assert img.shape == (1024, 1024, 3)

        img, mask = image_to_tensor(img, keepdim = True).float(), torch.from_numpy(mask).float()
        return img.squeeze().float(), mask.float()
            

"""
Define the dataset classes which will be imported into inference.py.
"""
class TileDataset(data.Dataset):
    def __init__(self, paths:dict, stride:int = 256, num_classes = 1, save = False, wsi_filter:list[str] = None, imagenet_norm: bool = False):
        """
        PyTorch dataset where a single entity (row) represents a single tile in a WSI.

        Inputs:
            paths (dict) - Dictionary of paths to wsi_csv, and zarr files for read and write.
                csv_path (str) - Path to csv with wsi file names and CERAD labels. (consider preprocessing separately i.e. load preprocessed wsi.csv)
                read_path (str) - Path to zarr wsi used for zarr reading in _load_zarr.
                save_path (str) - Path in which wsi heatmaps are initialized via _create_save_destination.
            stride (int) - Stride of tile size movement (used for coordinate generation for indexing)
            num_classes (int) - Number of output channels created in wsi output (i.e. prediction heatmap for how many classes?)
            save (bool) - Boolean representing whether to create an empty zarr file for the output of inference on a WSI/ROI.
            wsi_filter (list[str]) - List of wsis with which to make the dataset. Default none for all wsis.
            imagenet_norm (bool) - Boolean representing whether to normalize image to imagenet mean/std deviation.

        """
        super().__init__()
        #define a static tile size (based on what model has been trained on)
        self.tile_size = 1024
        #create class variables with init parameters
        self.stride = stride
        self.read_path = paths['read_path']
        self.save_path = paths['save_path']
        self.roi_path = paths.get('roi_path', None) #not necessary

        #compressor for writing to zarr files
        # self.compressor = Blosc(cname='lz4', clevel=1, shuffle=1)
        self.compressor = Blosc(cname='zstd',clevel=5, shuffle=Blosc.BITSHUFFLE)

        #define transforms for dev set
        self.imagenet_norm = imagenet_norm
        self.transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.ConvertImageDtype(torch.float32),
                                    ])

        #modify wsi_csv
        def _create_wsi_df(wsi_csv_path):
            try:
                wsi_df = pd.read_csv(wsi_csv_path)
            except:
                wsi_df = pd.read_pickle(wsi_csv_path)
                
            #create img path column
            wsi_df['path'] = None

            #suboptimal speed, but simple (using multiple cols makes vectorization annoying)
            for i, row in wsi_df.iterrows():
                #query wsi_name and ROI_name
                wsi_name = row['CASE_ID'] 
                #define paths to open zarr arrays easier
                wsi_df.loc[i, 'path'] = Path(self.read_path, wsi_name + '.zarr')
                
                #if there's an ROI path, default to making ROI maps (not heatmaps)
                if self.roi_path:
                    roi_name = row['ROI']
                    wsi_df.loc[i, 'path'] = Path(self.roi_path, wsi_name, roi_name)
                            
            #create references to img and mask zarr arrays in table
            wsi_df['zarrReader'] = wsi_df['path'].apply(lambda img_path: zarr.open(img_path, mode='r'))
            wsi_df['shape'] = wsi_df['zarrReader'].apply(lambda z: z.shape) #consider when to insert padding
            wsi_df['Random ID'] = wsi_df['CASE_ID']

            print('Created zarr reader for each WSI')
            return wsi_df
        
        #load in wsi csv with path to WSIs and CERAD labels (and other info if necessary)
        self.wsi_csv = _create_wsi_df(paths['csv_path'])

        #define the number of classes within the dataset, used for creating output shape dimension
        self.num_classes = num_classes

        #allow us to filter for specific wsis
        if wsi_filter:
            self.wsi_csv = self.wsi_csv[self.wsi_csv['Random ID'].isin(wsi_filter)]

        #store number of tiles for easy access
        self.num_wsi = self.wsi_csv.shape[0]

        #create save destination for each WSI depending on WSI size (could naiively make it same size as WSI, but empty)
        if save:
            self._create_save_destination()

        self.tiles = self._vectorized_tile_df()

    
    def __getitem__(self, idx):
        row = self.tiles.loc[idx]
        #load in tile with wsi_name reference and coords of top left point
        if self.roi_path:
            tile = self._load_zarr(row['Random ID'], (row['read_x'], row['read_y']), row['ROI'])
            data = row[['Random ID', 'save_x', 'save_y', 'ROI']].to_dict() #might need to add label for downstream analysis?
        else:
            tile = self._load_zarr(row['Random ID'], (row['read_x'], row['read_y']))
            data = row[['Random ID', 'save_x', 'save_y']].to_dict() #might need to add label for downstream analysis?

        if self.transform:
            tile = self.transform(tile)

        if self.imagenet_norm:
            #expand dimensions so that normalization works on a single image (expects batch)
            tile = K.enhance.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(torch.unsqueeze(tile, 0))
            tile = torch.squeeze(tile)

        return tile, data


    def __len__(self):
        #returning total number of tiles, if need to know num_wsi, check self.num_wsi
        return self.tiles.shape[0]

    
    def _vectorized_tile_df(self):
        """
        Creates a single dataframe with each row representing a valid tile within a WSI.
        Encapsulates all tiles in the dataset, across an arbitrary number of WSIs.
        """
        dfs = []
        for _, wsi in self.wsi_csv.iterrows():
            #generate a meshgrid with shape of input
            x = np.arange(0, wsi['shape'][0] - self.tile_size, self.stride)
            y = np.arange(0, wsi['shape'][1] - self.tile_size, self.stride)
            i_coords, j_coords = np.meshgrid(x, y, indexing = 'ij')

            #construct dictionary with which to instantiate df
            out_dict = {
                        'read_x': i_coords.flatten(),
                        'read_y': j_coords.flatten(),
                        }
            df = pd.DataFrame(out_dict, dtype = np.int32)

            # Use same output coords as input coords for segmentation
            df['save_x'] = (df['read_x']).astype(np.int32)
            df['save_y'] = (df['read_y']).astype(np.int32)

            print(f"WSI RANDOM ID: {wsi['Random ID']}")
            df['Random ID'] = [wsi['Random ID']] * df.shape[0] #???
            if self.roi_path:
                print(f"WSI ROI: {wsi['ROI']}")
                df['ROI'] = [wsi['ROI']] * df.shape[0] #???

            dfs.append(df)
        return pd.concat(dfs, ignore_index = True)
                    
                    
    def _load_zarr(self, randID, coords, ROI = None):
        """
        Use coordinate indexing to access a chunk of the zarr file. 
        The output will be fed into __getitem__ method, which will be how a dataloader accesses the data.

        We are given a set of coordinates from class variable self.tiles that possesses
        rows with a single (x, y) coordinate representing the top left corner of a fixed box.
        This box will navigate around our image via these coordinates.

        Inputs:
            randID (str) - 'Random ID' of the WSI 
            coords (tuple) - Coordinates of top left corner of chunk that we are loading in.
            ROI (str) - Optional, ROI that we would like to load from. Ex: 'rectangle_0'.

        Outputs:
            img_chunk (zarr) - An image chunk of the WSI, represented in zarr array format for parallelized processing.
        """
        #select the correct Zarr reader from wsi_csv (to avoid repetitive zarr.open call)
        mask = self.wsi_csv['Random ID'] == randID
        if ROI:
            mask = (self.wsi_csv['Random ID'] == randID) & (self.wsi_csv['ROI'] == ROI)

        z = self.wsi_csv.loc[mask, 'zarrReader'].values[0]
        #do orthogonal indexing via slices
        x, y = coords
        img_chunk = z.oindex[slice(x, x + self.tile_size), slice(y, y + self.tile_size)]

        return img_chunk

    def _create_save_destination(self):
        """
            Creates an empty zarr file for each WSI to store the output heatmap. 
            Size will be the same as original WSI, and we will leave the rest of the zarr tensor unwritten.
        """
        #create a parent directory for storing all WSI heatmaps of this stride
        print(f'Creating {self.num_wsi} empty zarr files to save heatmaps')
        for _, wsi in self.wsi_csv.iterrows():

            #calculate heatmap output size, assuming wsi_shape includes padding
            wsi_shape = wsi['shape']
            output_shape = ((wsi_shape[0]), (wsi_shape[1]), self.num_classes)

            print(f'WSI Shape:{wsi_shape}, Output shape: {output_shape}')

            #create save path for output wsi_heatmap
            out_path = Path(self.save_path, wsi['Random ID'])
            #if we're saving ROIs, save in this structure
            if self.roi_path:
                out_path = Path(self.save_path, wsi['Random ID'], wsi['ROI'])

            #create .sync file for multiprocessing
            synchronizer = zarr.ProcessSynchronizer(out_path.with_suffix('.sync'))
            #delete file if it already exists
            if os.path.isdir(out_path):
                shutil.rmtree(out_path)

            #create zarr file with read/write access, output_shape, etc.
            out = zarr.open(out_path, mode = 'a', shape = output_shape, #could be the issue
                            chunks = (self.tile_size*2, self.tile_size*2),
                            write_empty_chunks=False,
                            dtype = np.float16, #should be sufficient for output 
                            compressor = self.compressor,
                            synchronizer = synchronizer) #use this to enable file locking



def main(args):
    from PIL import Image
    import torchvision
    pl.seed_everything(42)

    if 'gutman' in args.data_dir:
        print("Creating general dataset!")
        dm = GeneralROIDataModule(data_dir = args.data_dir,
                        batch_size = 32,
                        num_workers = 2) #multiple worker issues ? need to def worker_init_fn
    else:
        dm = ROIDataModule(data_dir = args.data_dir,
                            batch_size = 32,
                            num_workers = 2) #multiple worker issues ? need to def worker_init_fn
    dm.prepare_data()
    dm.setup(stage = 'fit')

    #replace the train transform with test to see raw imgs
    dm.train_transform = dm.test_transform
    dataloader = iter(dm.train_dataloader())

    if args.inference:
        print("Performing inference!!")
        #define paths
        READ_PATH = '/scratch/sghandian/nft/raw_data/wsis/zarr/scale_1.0/' #use the padded wsis
        # WSI_SAVE_PATH = f"/scratch/sghandian/nft/wsi_heatmaps/{MODEL_NAME}/stride_{STRIDE}/"
        # ROI_SAVE_PATH = f"/scratch/sghandian/nft/output/roi_heatmaps/{MODEL_NAME}/stride_{STRIDE}/"
        WSI_CSV_PATH = '/scratch/sghandian/nft/model_input_v2/tables/split.pickle'

        paths = {
            'csv_path': WSI_CSV_PATH,
            'read_path': READ_PATH,
            'save_path': "None_save_path",
                }
        wsi = "1-102-Temporal_AT8"
        dataset = TileDataset(paths, #pass in paths dict
                            stride = 1024,
                            num_classes = 1,
                            save = False,
                            #this is a list input, remove to run on entire wsi list or pass as an arg --wsi_filter=False 
                            wsi_filter = [wsi])
        dataloader = iter(data.DataLoader(dataset,
                                     shuffle = False,
                                     batch_size = 32,
                                     num_workers = 16
                                    ))
                        
    #load a batch                        
    batch_img, batch_mask = next(dataloader)
    batch_img, batch_mask = next(dataloader)

    #create grid from batches, scale to uint8 range and put into BS x C x H x W format
    img_grid = (torchvision.utils.make_grid(batch_img.squeeze() * 255)).permute(1, 2, 0)
    #convert to numpy and uint8 datatype
    np_img = img_grid.numpy().astype(np.uint8)
    #convert to PIL Image for export
    img = Image.fromarray(np_img)
    #create save directory and save img
    os.makedirs(args.save_dir, exist_ok=True)
    img.save(Path(args.save_dir, "train_img.png"))

    if not args.inference: #inference setting has no mask, returns a dict instead of a mask
        mask_grid = (torchvision.utils.make_grid(batch_mask.unsqueeze(axis = 1) * 255)).permute(1, 2, 0)
        #use the below indexing to squeeze extra dimension while preserving color channel --> BS x 1 x H x W format
        # mask_grid = (torchvision.utils.make_grid(batch_mask[:, 0, :, :, :] * 255)).permute(1, 2, 0)

        mask_img = mask_grid.numpy().astype(np.uint8)
        mask = Image.fromarray(mask_img)
        mask.save(Path(args.save_dir, "train_mask.png"))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='/srv/home/sghandian/data/nft/')
    parser.add_argument("--inference", type = bool, default = False)

    args = parser.parse_args()
    main(args)

    """
    ##-----EXAMPLE USAGE-----## NOTE: MUST RUN ALL SCRIPTS FROM NFT DIRECTORY (even annot_conversion scripts)
    
    python nft_datasets.py --data_dir=/scratch/sghandian/nft/model_input_v2/ --save_dir=/srv/home/sghandian/data/nft/
    
    python nft_datasets.py --data_dir=/scratch/sghandian/nft/gutman_input/ --save_dir=/srv/home/sghandian/data/nft/output/
    """
