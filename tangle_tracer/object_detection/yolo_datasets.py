import pandas as pd
import numpy as np
from pathlib import Path
import os
import zarr
from tqdm import tqdm

from tangle_tracer.annot_conversion.WSIAnnotation import load_annot
from nft_datasets import ROIDataModule
import torchvision.ops.boxes as bops
import torch
from histomicstk.annotations_and_masks.masks_to_annotations_handler import get_contours_from_mask

class YOLODataModule(ROIDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup('fit')
        self.setup('test')
        self.datasets = {'train': self.ROI_train,
                          'val': self.ROI_val,
                          'test': self.ROI_test  
                        }
        self.create_bbox_labels()

    def create_bbox_labels(self):     
        #load dataset w/ relevant params
        for _, ds in self.datasets.items():

            df = self.attach_bbox_label(ds.tiles)
            df['yolo_label'] = df.apply(self.convert_to_yolo, axis = 1)
            #assign changes to dataset's tiles df
            ds.tiles = df
            #if we want to save the table, do so here
            # df.to_csv(Path(save_path, 'tables', ds_name).with_suffix('.csv'))

        assert self.ROI_train.tiles.loc[0, 'yolo_label'] is not None
        assert self.ROI_val.tiles.loc[0, 'yolo_label'] is not None
        assert self.ROI_test.tiles.loc[0, 'yolo_label'] is not None
    
    @staticmethod #doesn't need self, and doesn't need other functions
    def get_bounding_coords(mask: np.array):
        mask = mask.copy()[:, :, 0]
        assert len(mask.shape) == 2
        min_coords = np.min(np.nonzero(mask), axis=1)
        max_coords = np.max(np.nonzero(mask), axis=1)
        #need center coords (y, x), width, and height
        center_coords = np.mean([min_coords, max_coords], axis = 0)
        width_height = max_coords - min_coords
        return np.concatenate([center_coords, width_height])
    
    @classmethod #doesn't need to access instance information in self, just static method tied to the class
    def get_yolo_roi_coords(cls, wsiAnnot, norm = False):
        """
        Takes in a WSIAnnot and produces a dictionary containing bounding box representation of all masks, grouped by ROI.
        Bounding boxes are represented in the following way:
            [[center_y, center_x], [width, height]]
        """

        all_bounds = {}
        for rect, vals in wsiAnnot.img_regions.items():
            all_bounds[rect] = {}
            roi_bounds = np.array(list(vals['img'].shape[:2]))
            for annot, bboxes in vals['bboxes'].items():
                #calculate bounds from mask
                try:
                    bbox_bounds = cls.get_bounding_coords(wsiAnnot.masks[annot]).reshape(2,2)
                except ValueError:
                    print(f'ValueError: Empty mask, no NFT found for {annot}!')
                
                #get offset from attached metadata
                bbox_offset = np.array([bboxes['ymin'], bboxes['xmin']])
        
                #add bbox_offset in correct order to shift objects
                bbox_bounds[0] += bbox_offset
                
                #normalized coordinates for ROI, NOT helpful since YOLO cannot intake entire ROIs
                if norm:
                    bbox_bounds = bbox_bounds / roi_bounds
                else:
                    bbox_bounds = bbox_bounds.astype(int)
                    
                #aggregate bbox info for rectangle    
                all_bounds[rect][annot] = bbox_bounds
        return all_bounds

    @staticmethod
    def clip_bounding_box(bbox, image_size=(1024, 1024)):
        # Calculate left, top, right, and bottom coordinates
        bbox = bbox.flatten()
        y_center, x_center, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
        left = x_center - width / 2
        top = y_center - height / 2
        right = x_center + width / 2
        bottom = y_center + height / 2

        # Clip coordinates to fit within the image_size
        left = np.clip(left, 0, image_size[0])
        top = np.clip(top, 0, image_size[1])
        right = np.clip(right, 0, image_size[0])
        bottom = np.clip(bottom, 0, image_size[1])

        # Recalculate width and height after clipping
        width = right - left
        height = bottom - top

        left, top = left + (width/2), top + height/2

        # Create a new clipped bounding box array in YOLO format
        clipped_bbox = np.array([left, top, width, height]).astype(int).T

        return clipped_bbox

    @classmethod
    def bbox_labeler(cls, row, yolo_coords, tile_size):
        #checks to see if a tile contains one of the nft_centers
        bbox_labels = []
        if row['pos'] == 0:
            return []
        
        min_tile_coord = row.read_y, row.read_x
        max_tile_coord = row.read_y + tile_size, row.read_x + tile_size
        for _, nft in yolo_coords.items():
            x_center, y_center = nft[0][0], nft[0][1]
            if (x_center <= max_tile_coord[1] and y_center <= max_tile_coord[0]
                and x_center >= min_tile_coord[1] and y_center >= min_tile_coord[0]): 
                #clip bounds to be within tile
                bbox_label = nft.astype(np.float32)
                bbox_label[0] -= np.flip(np.array(min_tile_coord))
                bbox_label = cls.clip_bounding_box(bbox_label)

                #add label in yolo format to tile
                bbox_labels.append(bbox_label)

        return bbox_labels

    @classmethod 
    def attach_bbox_label(cls, df, load_path = '/scratch/sghandian/nft/model_input_v2/wsiAnnots/', tile_size = 1024):
        #goal: create a column indicating the normalized bounding box label for a specific tile 
        df = df.copy()
        df['bbox_label'] = None
        for case_id in df['CASE_ID'].unique():
            if not load_path:
                wsiAnnot = load_annot(case_id)
            else:
                wsiAnnot = load_annot(case_id, load_path)
            wsi_yolo_coords = cls.get_yolo_roi_coords(wsiAnnot, norm = False)
            #attach info to relevant portion of df
            for rect, annots in wsi_yolo_coords.items():
                #apply transform to correct portion of df
                sub_df = df.loc[(df['CASE_ID'] == case_id) & (df['ROI'] == rect)]
                sub_df['bbox_label'] = sub_df.apply(cls.bbox_labeler, args = (annots, tile_size), axis = 1)
                df.loc[(df['CASE_ID'] == case_id) & (df['ROI'] == rect), 'bbox_label'] = sub_df['bbox_label']
        
        return df

    @staticmethod #doesn't have any references to self or cls necessary
    def convert_to_yolo(row, tile_size = 1024):
        bbox_label = np.array(row.bbox_label) / tile_size
        yolo_label = [[0,*bbox] for bbox in bbox_label]
        yolo_label = yolo_label
        return yolo_label
    
    # <--------- USED FOR GROUND TRUTH and PRED BBOX OVERLAP -------->
    
    @staticmethod
    def yolo_coord_to_box(coord):
        x1, y1 = coord[0] - (coord[1]/2)
        x2, y2 = coord[0] + (coord[1]/2)
        return np.array([x1, y1, x2, y2]).astype(float)

    @classmethod
    def transform_coords_to_corners(cls, coord_dict):
        corner_dict = {}
        for rect, annot_dict in coord_dict.items():
            corner_dict[rect] = {}
            for name, coord in annot_dict.items():
                corner_coords = cls.yolo_coord_to_box(coord) #transform coords here
                corner_dict[rect][name] = corner_coords
            gt_roi_df = pd.DataFrame(corner_dict[rect]).T.reset_index().rename(
                        columns = {'index':'annotation', 0:'ymin', 1:'xmin', 2:'ymax', 3:'xmax'})
            corner_dict[rect] = gt_roi_df
        return corner_dict
    
    @staticmethod
    def calc_iou(box1, box2):
        iou = bops.box_iou(torch.tensor([box1]), torch.tensor([box2]))
        return iou
    
    @staticmethod
    def calc_sim(box1, box2):
        ymin, xmin, ymax, xmax = box1
        box2_ymin, box2_xmin, box2_ymax, box2_xmax = box2
        box1_center = np.array([ymin + (ymax - ymin)//2, xmin + (xmax-xmin)//2])
        box2_center = np.array([box2_ymin + (box2_ymax - box2_ymin)//2, box2_xmin + (box2_xmax-box2_xmin)//2])

        dist = np.linalg.norm(box2_center - box1_center)
        return dist
    
    @staticmethod
    def merge_boxes(box1, box2):
        return [min(box1[0], box2[0]),
                min(box1[1], box2[1]),
                max(box1[2], box2[2]),
                max(box1[3], box2[3])]
    
    @classmethod
    def merge_preds(cls, bboxes, min_distance):
        need_to_merge = True
        # bboxes = [box for box in bboxes if pd.notna(box)]
        bboxes = list(bboxes)
        while need_to_merge:
            need_to_merge = False

            for i, (box_1) in enumerate(bboxes):
                for j, (box_2) in enumerate(bboxes):
                    if j <= i:
                        continue

                    if cls.calc_sim(box_1, box_2) < min_distance:
                        new_box = cls.merge_boxes(box_1, box_2)

                        # Remove the smaller index first to avoid index errors
                        if i < j:
                            del bboxes[j]
                            del bboxes[i]
                        else:
                            del bboxes[i]
                            del bboxes[j]

                        bboxes.append(new_box)
                        need_to_merge = True
                        break  # break the inner loop, go to the next iteration of the outer loop
        
        return np.array(bboxes)


    @classmethod
    def get_pred_bboxes_for_roi(cls, mask, min_size = 30, min_distance = 100):
        #init df
        data_dict = {'group': ['bkgd', 'nft'], 'GT_code': [0, 1], 'color':['(0,0,0)', '(0,255,0)']}
        GT_df = pd.DataFrame(data_dict)
        #feed into function
        contour_df = get_contours_from_mask(mask.copy(), GT_df, groups_to_get = ['nft'],
                                            MIN_SIZE=min_size, MAX_SIZE = 400*400, get_roi_contour=False)
        preds = cls.merge_preds(contour_df[['xmin', 'ymin', 'xmax', 'ymax']].values, min_distance=min_distance)
        return preds

    @classmethod
    def get_roi_ious(cls, roi_pred_df, roi_truth_df):
        """
        Inputs:
           roi_pred_df (pd.DataFrame): DF representing the coordinates of the predictions.
                                       It is generated from get_contours_for_roi();
           roi_truth_df (pd.DataFrame): DF representing the coordiantes of the ground truth.
                                        It is generated via transform_coords_to_corners(),
                                        which produces a df for each ROI tied to a WSIAnnot.
        Returns:                           
           ious (np.array): 1-dimensional output of iou scores for each prediction. Includes
                            false positives and false negatives, which are scores of
                            0 and NaN respectively.                            
        """
        iou_scores = []
        matches = []
        truth_match = []  
        pred_match = [] 
        roi_truth_df = roi_truth_df.set_index('annotation')
        for pred_coord in roi_pred_df:
            best_iou = torch.tensor(0)
            best_match = [None, None, pred_coord]
            for annot, row in roi_truth_df.iterrows():
                row = row[['xmin', 'ymin', 'xmax', 'ymax']].values
                iou = cls.calc_iou(pred_coord.astype(int), row.astype(int))
                if iou > best_iou:
                    best_iou = max([iou, best_iou])[0]
                    best_match = annot, row, pred_coord
            iou_scores.append(best_iou)
            matches.append(best_match[0])
            truth_match.append(tuple(best_match[1].astype(int)) if best_match[1] is not None else best_match[1])
            pred_match.append(tuple(best_match[2].astype(int)) if best_match[2] is not None else best_match[2])
        df = pd.DataFrame({'annot': matches, 'iou': torch.tensor(iou_scores).numpy(), 
                        'truth_coord': truth_match,
                        'pred_coord': pred_match})
        return df['iou'], df

    @classmethod
    def get_dataset_predictions(cls, dataset_df, min_size = 5, min_distance = 150,
                            roi_mask_dir = '/scratch/sghandian/nft/output/roi_heatmaps/0fnaalid_NFT_trial-epoch=83-valid_loss=0.12.ckpt/stride_1024',
                            wsiAnnot_path = '/scratch/sghandian/nft/model_input_v2/wsiAnnots/'):
        #goal: create a dataframe containing all preds from all rois in this dataset
        all_nft_data = []
        for case_id in tqdm(dataset_df['CASE_ID'].unique(), desc='Iterating through WSIs:'):
            wsiAnnot = load_annot(case_id, Path(wsiAnnot_path))
            yolo_coords = cls.get_yolo_roi_coords(wsiAnnot)
            #ground truth bboxes
            gt_coords = cls.transform_coords_to_corners(yolo_coords)
            for rect in gt_coords:
                roi_gt_df = gt_coords[rect]
                #prediction bboxes
                mask_path = Path(roi_mask_dir, case_id, rect)
                mask = zarr.load(mask_path)[:,:,0].astype('uint8')
                try:
                    #filter preds based on distance (create new bboxes if close)
                    roi_preds = cls.get_pred_bboxes_for_roi(mask, min_size = min_size, min_distance = min_distance)
                    
                    #matched gt and preds
                    _, iou_df = cls.get_roi_ious(roi_preds, roi_gt_df)
                    
                    #add back in unmatched gt (fn)
                    merged_df = roi_gt_df.merge(iou_df, left_on = 'annotation', right_on = 'annot', how = 'outer')
                
                except Exception:
                    print(f"No preds found for {case_id} {rect}; returning only ground truth bboxes.")
                    merged_df = roi_gt_df #no preds found in mask, empty

                merged_df['CASE_ID'] = case_id
                merged_df['ROI'] = rect
                all_nft_data.append(merged_df)
        
        return pd.concat(all_nft_data)
    
    def get_segmentation_objdetect_preds(self, min_size = 5, min_distance = 150):
        #load dataset w/ relevant params
        table_path = Path(self.data_dir, 'tables', f'segobjdetect_results') 
        os.makedirs(table_path, exist_ok=True)
        for ds_name, ds in self.datasets.items():
            dataset_df = self.get_dataset_predictions(ds.tiles, min_size, min_distance)
            save_path = Path(table_path, ds_name).with_suffix('.csv')
            dataset_df.to_csv(save_path)

def main(args):
    nft_path = args.data_dir
    dm = YOLODataModule(data_dir = nft_path,
                        batch_size= 4,
                        num_workers= 20,
                        imagenet_norm = False)
    # dm.prepare_data()
    print("Successfully created a YOLODataModule!")
    dm.get_segmentation_objdetect_preds(min_size = 40, min_distance = 150)

        

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)


    """
    Example Usage:

    python yolo_datasets.py  --datadir='/scratch/sghandian/nft/model_input_v2'
    """