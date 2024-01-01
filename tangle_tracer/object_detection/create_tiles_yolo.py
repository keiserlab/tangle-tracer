from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
import torchvision.transforms as T

def _save_tile_and_label(tile, label, wsi_name, roi_name, coords, save_path):
    #create a name for the tile based on coordinates
    fname = f"{wsi_name}_{roi_name}_y-{coords[0]}_x-{coords[1]}"
    tilepath = Path(save_path, 'images', fname).with_suffix('.png') #it may be the case that we need a flat strxr
    #convert Tensor tile to PIL.Image
    img = T.ToPILImage()(tile).convert("RGB")
    #save PIL image
    img.save(tilepath)

    if len(label) == 0:
        return None #no need to save a label path for empty images!
    #create path for yolo label 
    labelpath = Path(save_path, 'labels', fname).with_suffix('.txt')
    #write text in label col to .txt file
    _convert_label_to_txt(label, labelpath)

def _convert_label_to_txt(label, outpath):
    #NOTE: make sure label string is correct when outputting to .txt file
    label = np.array(label).astype(str)
    # print("LABEL", label)
    with open(outpath, "w+") as f:
        for row in label:
            row[0] = str(int(float(row[0])))
            text=" ".join(row)
            f.write(text)
            f.write("\n")

def save_yolo_tiles(dataset, save_path):
    df = dataset.tiles
    print('WSIs being processed:', df['CASE_ID'].unique())
    for i, row in tqdm(df.iterrows(), total = df.shape[0], 
                       desc = 'Iterating through tiles across all WSIs'):
        wsi_name, roi_name =  row['CASE_ID'], row['ROI']
        y, x = row['read_y'], row['read_x'] #flip coordinates?
        label = row['yolo_label'] #get yolo_label
        # assert len(label) == 0 or (len(label) > 0 and label[0] == 0) #our dataset has only one class
        #get tile via dataset? don't have to deal with numpy, zarr loading, etc.
        tile, _ = dataset[i]
        _save_tile_and_label(tile, label, wsi_name, roi_name, (y, x), save_path = save_path)

def main(args):
    from yolo_datasets import YOLODataModule
    nft_path = Path(args.data_dir)
    dm = YOLODataModule(data_dir = nft_path,
                        batch_size= args.batch_size,
                        num_workers= args.num_workers,
                        imagenet_norm = False)

    #save the tiles listed in dm
    save_path = Path(args.save_dir)
    for name, ds in dm.datasets.items():
        #save all tiles for this specific dataset
        print(f"Saving {name} tiles and their labels...")
        os.makedirs(Path(save_path, name, 'images'), exist_ok=True)
        os.makedirs(Path(save_path, name, 'labels'), exist_ok=True)
        save_yolo_tiles(ds, Path(save_path, name))
        print(f"Done saving {name} tiles and their labels!")

    print('Done saving all tiles.')

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=16)

    args = parser.parse_args()
    main(args)

    """
    Example Usage:

    python create_tiles_yolo.py --data_dir='/scratch/sghandian/nft/model_input_v2' \
    --save_dir='/scratch/sghandian/nft/yolo_input/'
    
    """
