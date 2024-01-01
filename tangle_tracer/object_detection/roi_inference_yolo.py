import os, gc
from pathlib import Path

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import cv2
import torch
import zarr
from ultralytics import YOLO

def create_df_from_paths(img_path, label_path):
    """
    Create a dataframe of paths to tiles sliced from all ROIs in a dataset split.
    """
    img_paths = [Path(img_path, sub_path) for sub_path in os.listdir(img_path)]
    df = pd.DataFrame(img_paths, columns=['img_path'])
    df['img_name'] = df['img_path'].apply(lambda x: x.name)
    df['y_coord'] = pd.to_numeric(df['img_name'].str.split('-').apply(lambda x: x[-2][:-2]), errors = 'coerce')
    df['x_coord'] = pd.to_numeric(df['img_name'].str.split('-').apply(lambda x: x[-1][:-4]), errors = 'coerce')
    df['CASE_ID'] = df['img_name'].str.split('_').apply(lambda x: "_".join(x[:2]))
    df['ROI'] = df['img_name'].str.split('_').apply(lambda x: "_".join(x[2:4]))
    df['txtname'] = df['img_name'].str.replace('.png', '.txt')
    
    #some of these might not exist
    # df['label_path'] = df['path'].apply(lambda x: x.with_suffix('.txt')).replace('images', 'labels')
    label_files = pd.DataFrame(os.listdir(label_path), columns=['label_file'])
    df = df.merge(label_files, left_on = 'txtname', right_on = 'label_file', how = 'left').drop(['txtname'], axis = 1)
    df['label_path'] = df['label_file'].apply(lambda x: Path(label_path, x) if pd.notna(x) else x)

    #apply optional sorting by y_coord to get vertical columns?
    df = df.sort_values(by = ['CASE_ID', 'ROI', 'y_coord'], ascending = True)
    return df

def create_all_dfs_from_paths(data_dir):
    #create a dict containing paths
    df_dict = {}
    for ds_name in ['train', 'val', 'test']:
        img_path = Path(data_dir, ds_name,'images')
        label_path = Path(data_dir, ds_name, 'labels')
        df = create_df_from_paths(img_path, label_path)
        df_dict[ds_name] = df
    return df_dict

def read_text(text_path):
    #read a label file in text format to parse ground truth bboxes 
    if pd.isna(text_path):
        return []
    with open(text_path, 'r') as f:
        boxes = f.readlines()
        out_boxes = []
        for box in boxes:
            box_ls = box.strip().split(' ')
            out_boxes.append(box_ls)
    try:
        out_boxes = np.array(out_boxes).astype(np.float64)
    except ValueError:
        print("Malformed text file! Skipping:", text_path)
        out_boxes = []
    return out_boxes
    
def yolo_coord_to_xxyy(coord, img_size = 1024):
    #Converts yolo coordiantes to corner coordinates
    boxes = []
    coord = coord * img_size
    for box in coord:
        box = box[1:]
        box = box.reshape(2, -1)
        x1, y1 = box[0] - (box[1]/2)
        x2, y2 = box[0] + (box[1]/2)
        box = np.array([x1, y1, x2, y2]).astype(int)
        boxes.append(box)
    return np.array(boxes)


def get_results_and_gt(model, results_df, indices, conf = 0.05, 
                       device = (0,1), batch_size = 64):
    """Generates results and loads ground truth bboxes for all tiles in a dataset"""
    paths = results_df.loc[indices, 'img_path'].to_list()
    path_generator = np.arange(0, len(paths), batch_size)
    outputs = []
    gt_boxes = []
    for batch_start in path_generator:
        batch_end = batch_start + batch_size
        results = model.predict(paths[batch_start:batch_end], device = device, 
                        conf = conf, iou = 0.1, show_conf = False)
        gt = [yolo_coord_to_xxyy(read_text(results_df.loc[i, 'label_path'])) for i in range(batch_start, batch_end) if i < len(paths)]
        outputs.extend(result.cpu() for result in results)
        gt_boxes.extend(gt)
    return outputs, gt_boxes

def draw_pred_and_gt(results, gt_boxes):
    drawn_imgs = []
    # i = 0
    for r, gt in zip(results, gt_boxes):
        r = r.cpu()
        # gt = gt_boxes[i]
        #use ultralytics plotting to plot a red box around each prediction
        img = r.plot(line_width = 7)  # plot a BGR numpy array of predictions
        #replace all red pixels with gold
        img[np.all(img == (56, 56, 255), axis=-1)] = (0, 215, 255)

        #plot ground truth bboxes on top of these to produce agreement ROIs
        if len(gt) != 0:
            for box in gt:
                #draw cyan gt bboxes
                img = cv2.rectangle(img, tuple(box[:2]), tuple(box[2:]),
                                    color = (255, 255, 0), thickness=7)

        img = img[..., ::-1] #turn to RGB
        drawn_imgs.append(img)
        i += 1
        del r
        gc.collect()
        torch.cuda.empty_cache()
    return drawn_imgs

def plot_pred_and_gt(images, cols, figsize = None, display = False, save_path = None):
    """
    Plot a grid of images.

    Parameters:
    - images: List of NumPy arrays representing images.
    - cols: Number of columns in the grid.

    Returns:
    - None
    """
    total_images = len(images)
    rows = (total_images + cols - 1) // cols  # Calculate the number of rows
    if not figsize:
        figsize=(cols*2, rows*2)

    # Create a larger array to store the images
    result_image = np.zeros((rows * images[0].shape[0], cols * images[0].shape[1], 3)).astype(np.uint8)

    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            if index < total_images:
                row_start = i * images[0].shape[0]
                row_end = (i + 1) * images[0].shape[0]
                col_start = j * images[0].shape[1]
                col_end = (j + 1) * images[0].shape[1]
                result_image[row_start:row_end, col_start:col_end] = images[index]
            else:
                break  # Stop if we run out of images

    if display:
        # Display the resulting image
        plt.figure(figsize = figsize)
        plt.imshow(result_image)
        plt.axis('off')
        plt.show()

    if save_path and not display:
        zarr.save(Path(save_path), result_image)
        print("Saved ROI:", save_path.as_posix())

    elif display and save_path:
        os.makedirs(save_path.parent, exist_ok=True)
        plt.savefig(Path(save_path))
        print("Saved ROI:", save_path.as_posix())


    return result_image


def main(args):
    model = YOLO(args.ckpt_path)
    save_path = Path(args.save_path)
    #Create a dict mapping dataset labels to sets of tiles in png/jpeg format
    df_dict = create_all_dfs_from_paths()
    for ds_name, df in df_dict.items():
        unique_rois = df[['CASE_ID', 'ROI']].drop_duplicates()
        for i, pair in unique_rois.iterrows():
            df_subset = df[(df['CASE_ID'] == pair.values[0]) & (df['ROI'] == pair.values[1])].drop_duplicates(subset = ['ROI', 'y_coord', 'x_coord']).reset_index(drop = True)
            df_subset = df_subset.sort_values(by = ['x_coord', 'y_coord'], ascending = True).reset_index(drop = True)
            results, gt_boxes = get_results_and_gt(model, df_subset, range(0, len(df_subset)), conf = 0.03,
                                                   batch_size=args.batch_size, device=args.device)
            drawn_imgs = draw_pred_and_gt(results, gt_boxes)
            stitched_preds = plot_pred_and_gt(drawn_imgs, cols = 21, figsize = (200, 200), 
                                                display = False, save_path = Path(save_path, pair.values[0], pair.values[1]).with_suffix('.zarr'))

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True) #expects train/val/test folders within
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default=(0, 1))
    
    args = parser.parse_args()
    main(args)

    """
    Example Usage:

    python roi_inference_yolo.py --data_dir='/scratch/sghandian/nft/yolo_input/' \
        --ckpt_path='/scratch/sghandian/nft/yolo_input/NFTObjDetect/NFTObjDetectBest5/weights/best.pt' \
        --save_path='/scratch/sghandian/nft/output/roi_objdetects/'
    """
