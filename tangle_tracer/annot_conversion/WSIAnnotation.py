import numpy as np
import zarr
import cv2
from tqdm import tqdm
import os, shutil, sys
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import zarr
from numcodecs import Blosc
import pandas as pd

import skimage
import pickle

from tangle_tracer.annot_conversion.cz_utils import CzParser
from tangle_tracer.annot_conversion.czi_to_zarr import get_section
# from cz_utils import CzParser
# from zarr_utils import get_section #these imports cause pytest to glitch?


#create a WSIAnnotation class
class WSIAnnotation():
    """
    Goal: Return an info_dict that can be used to represent bounding boxes within the image (WSI or rectangles?)

    """
    
    def __init__(self, img_path, annot_path, scale = 1.0, save = True, overwrite = True,
                 save_path = '/scratch/sghandian/projects/nft/model_input2/', 
                 reannotations = '/srv/home/lianeb/reannotations.csv'):
        """
        Initialize WSIAnnotation.
        """
        try:
            self.reannotations = pd.read_csv(reannotations)
        except FileNotFoundError:
            print("Default reannotation file not found. Either pass in None for reannotations parameter or change path.")
            self.reannotations = []

        self.scale = scale
        img_path = Path(img_path)
        
        cz_annot_path = self.__create_annot_path(annot_path)
        self.annot_data = CzParser(cz_annot_path).get_regions()

        self.wsi_name = img_path.as_posix().split('/')[-1].split('.')[0]
        self.wsi_img = zarr.open(img_path, mode = 'r', dtype = 'i4')


        #define bbox size
        self.bbox_size = int(400 * self.scale)
        #define input tile size for model, pad ROI by this much + jitter amount
        self.input_tile_size = int(1024 * self.scale)
        #how much to vary random sampling coords of positive class
        self.jitter_amount = self.input_tile_size // 2
        #calculate total padding size
        self.padding_size = self.input_tile_size + self.jitter_amount

        #need save_path for zarr dump of corrected ROIs
        self.save_path = save_path
        self.chunk_size = 1000 #used for zarr chunks
        self.overwrite = overwrite

        try:
            self.img_regions = self._get_img_regions(self.annot_data['regions'])
            self.bboxes = self._get_img_bboxes(self.img_regions)
            self.masks = self._draw_all_masks(biggest = False)
        except FileNotFoundError:
            print(f'No annotation data found for {self.wsi_name}. \
                    Returning WSIAnnot with reference to zarr image only.')

        if save:
            self.save_annot()

    def __create_annot_path(self, annot_path):
        basename = os.path.basename(annot_path)
        if '_s' in basename:
            name = "_".join(basename.split('_')[:-1])
            annot_path = Path("/".join(annot_path.as_posix().split('/')[:-1]),
                              name).with_suffix('.cz')
        return annot_path
                
    def __get_nft_bboxes(self, rectangle):
        """
        Creates bbox DIMENSIONS around each NFT in a given ROI. Filters out NFTs 
        that are not within this ROI.

        Input:
            rectangle (str) - Index to self.annot_data['regions'] which contains
                              metadata specific to this ROI within the larger WSI.

        Returns:
            bboxes (dict) - Maps an NFT (str) to its bbox dimensions (dict).
        """

        #define nft and rectangle metadata
        rect_annots = self.annot_data['regions'][rectangle]
        offset = (rect_annots['Left'] * self.scale, 
                  rect_annots['Top'] * self.scale)
        angle = -(rect_annots['Rotation']) #use negative version
        rect_width, rect_height = rect_annots['Width'] * self.scale, rect_annots['Height'] * self.scale
        rect_center =  np.int32([rect_width//2, rect_height//2])
        
        #use local coords if provided
        if 'nfts' in rect_annots:
            nft_annots = rect_annots['nfts']
        else:
            nft_annots = self.annot_data['nfts']
        
        #define new point annotations
        num_nfts = len(nft_annots) #number of original nfts
        
        #define rotation matrix M
        M = self.get_rotation_matrix(angle)

        def process_annots_into_bboxes(annot_dict, bbox_size, rect_width, rect_height, scale = 1.0):
            #bbox border in each direction from center
            bboxes = {}
            border = int(bbox_size // 2)
            for nft, coord in annot_dict.items():
                x,y = int(coord[0] * scale), int(coord[1] * scale)
                dims = {
                        'xmin': int(x - border),
                        'ymin': int(y - border),
                        'xmax': int(x + border),
                        'ymax': int(y + border),
                        'width': int(border * 2),
                        'height': int(border * 2), #can likely get rid of width and height
                        }
                #check to make sure the bbox is in this rectangle
                if all(int(x) > - border for x in dims.values()): #filtered out by >0 condition
                    if (dims['xmax'] <= rect_width + border) and (dims['ymax'] <= rect_height + border): #could have problems w/ border and edge NFTs
                        bboxes[nft] = dims 
            return bboxes
            
        def apply_rotation_to_bboxes(bboxes, bbox_size, rect_width, rect_height, offset, scale = 1.0):
        #check each global NFT annot for existence in this ROI (after rotation)
            border = int(bbox_size // 2)
            for nft, coord in bboxes.items():
                x, y = int(coord[0] * self.scale), int(coord[1] * self.scale)
                #convert to local coordinates
                xy = np.int32([x - offset[0], y - offset[1]])
                #shift coords about center of rectangle
                xy -= rect_center
                #use rotation matrix, order matters!
                if 'nfts' in rect_annots:
                    x, y = np.int32(M @ xy) + np.flip(rect_center) 
                    #may need to add a check based on final orientation
                else:
                    #rect_center to correct shift after rotation, not for local
                    #shift coordinates back from origin for global coords!
                    x, y = np.int32(M @ xy) + rect_center 
                dims = {
                        'xmin': int(x - border),
                        'ymin': int(y - border),
                        'xmax': int(x + border),
                        'ymax': int(y + border),
                        'width': int(border * 2),
                        'height': int(border * 2), #can likely get rid of width and height
                        }
                #check to make sure the bbox is in this rectangle
                if all(int(x) > - border for x in dims.values()): #filtered out by >0 condition
                    if (dims['xmax'] <= rect_width + border) and (dims['ymax'] <= rect_height + border): #could have problems w/ border and edge NFTs
                        bboxes[nft] = dims
            return bboxes
        
        #post-process these boxes by adding the pad_amount to all of them!
        def modify_all_dims(bboxes, pad_amount = self.padding_size):
            def modify_dims(bbox_dims, pad_amount = pad_amount):
                bbox_dims = bbox_dims.copy()
                for dim, val in bbox_dims.items():
                    if dim in ['width', 'height']:
                        continue
                    bbox_dims[dim] = val + pad_amount
                return bbox_dims
            #copy bboxes and apply modify_dims to each bbox
            bboxes = bboxes.copy()
            for annot, bbox_dim in bboxes.items():
                bboxes[annot] = modify_dims(bbox_dim)
            return bboxes
        
        #convert raw NFT annotations from WSI-level into bboxes
        bboxes = process_annots_into_bboxes(nft_annots, self.bbox_size, rect_width, rect_height, self.scale)
        bboxes = apply_rotation_to_bboxes(bboxes, self.bbox_size, rect_width, rect_height, offset, self.scale)
        bboxes = modify_all_dims(bboxes, pad_amount = self.padding_size)

        if len(self.reannotations) > 0:
            df = self.reannotations
            reannot_roi = df[(df['ROI'] == int(rectangle.split('_')[-1])) & (df['WSI'] == self.wsi_name)]
            nft_reannots = reannot_roi['ROI_coordinates'].apply(eval).to_dict()
            
            new_keys = ['annotation_' + str(k+num_nfts) for k in nft_reannots.keys()]#nft_reannots.keys()]
            new_vals = [list(v) for v in nft_reannots.values()]
            nft_reannots = dict(zip(new_keys, new_vals))

            #no need to apply rotation to these as they were annotated at the ROI-level (pre-rotated)
            new_bboxes = process_annots_into_bboxes(nft_reannots, self.bbox_size, rect_width, rect_height, self.scale)
            new_bboxes = modify_all_dims(new_bboxes, pad_amount = 0)
            bboxes = {**bboxes, **new_bboxes}

        return bboxes


    def _transform_rect(self, metadata):
        #define and scale dimensions
        left, top = int(metadata['Left'] * self.scale), int(metadata['Top'] * self.scale)
        width, height = int(metadata['Width'] * self.scale), int(metadata['Height'] * self.scale)
        border, angle = 0, -metadata['Rotation'] #must use negative rot
        #get a path to a roi jpeg if it exists
        roi_path = metadata.get('Path', None)

        #calculate rotation M for given angle
        rotation_M = self.get_rotation_matrix(angle)

        #define points w/ respect to origin
        pt_A = [0, 0]
        pt_B = [0 + width, 0]
        pt_C = [0 + width, 0 + height]
        pt_D = [0, 0 + height]
        pts = np.int32([pt_A, pt_B, pt_C, pt_D])

        #shift to center of rectangle
        center = np.int32([[width//2, height//2]] * 4)
        pts = pts - center

        #define shifting params (border + offset dims)
        shift = np.float32([[border + left, border + top]] * 4) + center

        #create rotation matrix and input points
        in_pts = np.float32(pts @ rotation_M) + shift
        #define strict output points (height/width of rectangle)
        out_pts = np.float32([[0,         0],
                              [width - 1, 0],
                              [width - 1, height - 1],
                              [0,         height - 1]])

        #before doing transform, feed in a slice that
        #is cropped around the minimal region
        minCoords = np.flip(np.int32(in_pts.min(axis = 0)))
        maxCoords = np.flip(np.int32(in_pts.max(axis = 0)))
        tileSize = maxCoords - minCoords
        if not roi_path and self.wsi_img:
            img_slice = get_section(self.wsi_img, minCoords,
                                 tile_size = (tileSize))
        else:
            #load directory from jpeg if there's no WSI but there are ROIs
            try:
                img_slice = mpl.image.imread(roi_path)
            except FileNotFoundError:
                print('Path to ROI not found: {roi_path}. Fix path to ROI in metadata if no WSI exists.')


        #convert in_pts to local coordinates of img_slice
        in_pts -= np.flip(minCoords)
        #compute the perspective transform M
        M = cv2.getPerspectiveTransform(np.float32(in_pts),
                                        np.float32(out_pts))
        #warp the slice with the rotation matrix
        out = skimage.transform.warp(img_slice, np.linalg.inv(M),
                                     output_shape=(height, width)) 

        return out
    
    def _get_img_regions(self, rectangles):
        """
        Takes in the ROI annotations and generates a nested dictionary mapping rectangles to the ROI image,
        corrected for rotation, as well as the constitutent bbox dimensions for that specific rectangle.

        Input:
            rectangles (dict): Dictionary mapping rectangle name to global coordinates and rotation.

        Returns:
            img_slices (dict): Dictionary mapping rectangle name to corrected ROI view in np.array format,
                               and coordinates of bboxes.
        """
        img_slices = {}

        #can potentially choose the relevant region first, 
        for rectangle, metadata in tqdm(rectangles.items()):

            #uses self.wsi_img and coords to slice minimal inscribed img,
            #then rotates and returns the correct view of this cropped img
            img_slice = self._transform_rect(metadata)
            
            #pad array by padding amount on each side, 
            #MIGHT WANT THIS TO BE TILE SIZE !!! 1024px
            padded_img = np.pad(img_slice, ((self.padding_size, self.padding_size),
                                           (self.padding_size, self.padding_size), (0, 0)),
                                            mode = 'constant', constant_values = 1)

            #convert img_slice to zarr file on disk?
            os.makedirs(Path(self.save_path, 'images', self.wsi_name), exist_ok=True)
            wsi_path = Path(self.save_path,'images', self.wsi_name, rectangle)
            zarr_arr = self.__dump_to_zarr(padded_img, wsi_path)

            #uses rectangle coords + angle to generate bounding boxes      
            bboxes = self.__get_nft_bboxes(rectangle)

            #save to dictionary
            img_slices[rectangle] = {'img': zarr_arr, 'bboxes': bboxes}

        return img_slices

    def __dump_to_zarr(self, img: np.array, path: Path):
        # compressor = Blosc(cname='lz4', clevel=1, shuffle=1)
        compressor = Blosc(cname='zstd',clevel=5, shuffle=Blosc.BITSHUFFLE)
        synchronizer = zarr.ProcessSynchronizer(path.with_suffix('.sync'))
        store = zarr.DirectoryStore(path)
        if self.overwrite:
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                print('Cannot overwrite ROI that does not exist. Creating new zarr array.')
        try:
            z = zarr.array(img, 
                            store=store,
                            chunks = (self.chunk_size, self.chunk_size),
                            write_empty_chunks=False, 
                            compressor = compressor,
                            synchronizer = synchronizer)
        except zarr.errors.ContainsArrayError:
            print(f'Skipping {path} since it already exists.')
            z = zarr.open(path, mode = 'r')

        return z
    

    def _load_bbox(self, rectangle_img, bbox_object):
        """
        Takes in an annotation for a bounding box and retuns a slice of the rectangle image.

        Input:
            rectangle_img (np.array): Slice of WSI in np format
            bbox_object (dict): Dictionary {annotation_# : xmin, ymin, xmax, ymax, width, height}
            scale (float) : Scale at which the image is produced. Will scale bbox coordinates appropriately.

        Returns:
            box_slice (np.array): Slice of the image contained in the bounding box, padded to 400px if necessary.
        """
        left = int(bbox_object['xmin'])
        top = int(bbox_object['ymin']) 
        width = int(bbox_object['width'])
        height = int(bbox_object['height'])
        #add edge case handling via checking coordinates, altering indexing and padding
        box_slice = rectangle_img[slice(np.max([0, top]), top + height),
                                  slice(np.max([0, left]), left + width), :].copy()

        return box_slice
                             

        
    def _get_img_bboxes(self, img_regions):
        """
        Takes in the dictionary of rectangles mapped to its image, as well as its constituent bboxes dimensions,
        and produces all associated bbox images in a dictionary mapping annotation name to image.

        Input:
            img (zarr.array): WSI in Zarr format
            img_regions (dict): Dictionary for a single WSI mapping each annotated rectangle name to another dictionary,
            containing image region and its associated annotated NFTs in bounding box format.
                Example:
                img_regions['rectangle_0'] = {'img': img_slice, 'bboxes': bboxes}
        Returns:
            bbox_imgs (dict): Dictionary mapping an NFT bounding box annotation to an image slice.
                Example:
                bbox_imgs['annotation_0'] = bbox
        """

        bbox_imgs = {}
        for rectangle in img_regions:
            for bbox, dims in img_regions[rectangle]['bboxes'].items():
                bbox_imgs[bbox] = self._load_bbox(img_regions[rectangle]['img'], dims)
        return bbox_imgs
    

    def _create_blobs(self, mask, threshold=1500, max_distance = 125):
        """
        Takes in a binarized bbox image and generates a mask representing a single NFT within it. 
        """
        labels = skimage.measure.label(mask, connectivity=2, background=0)
        out_mask = np.zeros(mask.shape, dtype='uint8')
        sizes = {}
        
        # loop over the unique components
        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue
            # otherwise, construct the label mask and count the
            # number of pixels 
            labelMask = np.zeros(mask.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)
            
            # if the number of pixels in the component is sufficiently
            # large, then add it to our dictionary mapping labels to sizes
            if numPixels > threshold*(self.scale**2):
                #enforce center bias
                blob_center = np.float32([np.average(indices) for indices in np.where(labelMask == 255)])
                img_center = np.float32([200*self.scale, 200*self.scale])
                if np.linalg.norm(blob_center - img_center) < max_distance*self.scale:
                    sizes[label] = numPixels
        
        #get the largest blob
        try:
            maxLabel = max(sizes, key = sizes.get)
        except ValueError: #nothing found
            #---------------- consider adding the next smallest blob!!! -------------
            return out_mask, out_mask
        #pop the max label
        maxPixel = sizes.pop(maxLabel)
        
        #generate a mask with the largest label
        biggest_mask = np.zeros(mask.shape, dtype="uint8")
        biggest_mask[labels == maxLabel] = 255
        # biggest_mask_size = cv2.countNonZero(biggest_mask)
        
        out_mask = cv2.add(out_mask, biggest_mask)
        #get the biggest blobs within X% of max size
        percentage_threshold = 0.50
        #loop through all sufficiently large blobs and get sizes
        for label, pixelSize in sizes.items():
            #check if they're large enough to add to an output mask
            if (pixelSize / maxPixel) > percentage_threshold:
                #create an empty mask, then assign 1s to labeled region
                labelMask = np.zeros(mask.shape, dtype="uint8")
                labelMask[labels == label] = 255
                #add label mask to output mask
                out_mask = cv2.add(out_mask, labelMask)

        return out_mask, biggest_mask

        
    #try otsu thresholding
    def _draw_mask(self, image, show_img = False, overlay = False):
        """
        Takes in an image and converts it from RGB to HED channels. The image is then normalized and 
        converted to uint8, then the DAB channel is isolated and is binarized and thresholded via Otsu.
        Morphological operations then preprocess the image, then it's passed to _create_blobs() to generate
        the segmented mask for the image. Can use show_img and overlay flags to visualize results.

        Input:
            image (np.array): RGB image representing a zoomed-in NFT image from an ROI.
            show_img (bool): Flag to display image in notebook
            overlay (bool): Flag to overlay the mask onto the image. RETURNS the overlaid mask! 
        
        Retuns:
            mask, biggest_mask (tuple): Mask images that represent the most central mask and the biggest masked
                                        region within the bbox.
        """


        og_image = np.array(skimage.color.rgb2hed(image).copy())
        normed_image = None
        normed_image = cv2.normalize(og_image, normed_image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        #select DAB color channel
        try:
            DAB = normed_image[:, :, 2]
        except TypeError:
            print(f"OG image shape {og_image.shape} and image: {og_image}")
            return (None, None)

        #create binary mask
        thresh = cv2.threshold(DAB, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        #use cross as kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        # Apply morphological closing, then opening operations 
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # for _ in range(10):
            # closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
            # opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel) 


        #create blobs and their masks
        mask, biggest_mask = self._create_blobs(closing, threshold = 1000 * self.scale)
        
        if overlay:
            # Choose colormap
            cmap = plt.get_cmap('cool')
            # Get the colormap colors
            my_cmap = cmap(np.arange(cmap.N))
            # Set alpha to 0 for non values of 1
            my_cmap[:,-1][:255] = 0
            # Create new colormap
            my_cmap = ListedColormap(my_cmap)
            #apply colormap to mask
            mask = my_cmap(mask)
            #add an extra dimension to image
            image = cv2.cvtColor(np.float32(image), cv2.COLOR_RGB2RGBA)

            if show_img:
                # fig = plt.figure()
                plt.imshow(image)
                plt.imshow(mask, cmap = 'binary')    
        
        return mask, biggest_mask

    def _draw_all_masks(self, biggest = False):
        """
        Iterates through self.bboxes to retrieve 400x400px images for segmentation. Generates mask(s)
        based on strategy defined in _draw_mask(). Returns dictionary mapping NFT annotation name to
        a single grayscale image.
        """
        masks = {}
        for (nft, bbox) in self.bboxes.items():
            mask, biggest_mask = self._draw_mask(bbox, overlay = True)
            if biggest:
                masks[nft] = biggest_mask
            else:
                masks[nft] = mask
        return masks 

    #generate a rotation matrix using degrees
    def get_rotation_matrix(self, angle):
        """
        Generate a 2d rotation matrix with an input angle, defined in degrees.
        """
        angle = np.deg2rad(angle)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle),  np.cos(angle)]])
        return R

    def save_annot(self):        
        """
        Dumps the WSIAnnot object at the default save location for the specific scale at which
        the original WSI was loaded in (self.scale). 
        
        Can be loaded again with WSIAnnotation.load_annot(WSI_NAME).
        """
        save_path = Path(self.save_path,'wsiAnnots', f'scale_{self.scale}')
        #create a directory if needed
        os.makedirs(save_path, exist_ok=True)

        pickle_path = Path(save_path, f"{self.wsi_name}.pickle")
        #remove pickle file/directory if it does exist
        if os.path.isfile(pickle_path) or os.path.islink(pickle_path):
            os.remove(pickle_path)
        elif os.path.isdir(pickle_path):
            shutil.rmtree(pickle_path)
        
        #dump pickle file
        with open(Path(save_path, f"{self.wsi_name}.pickle"), "wb") as file_to_store:
            pickle.dump(self, file_to_store)


#----------------------------Utility Functions----------------------------#
def plot_bboxes(wsiAnnot, num_imgs = 12, overlay = True):    
    bbox_imgs = wsiAnnot.bboxes
    masks = wsiAnnot.masks

    num_imgs = min([num_imgs, len(bbox_imgs.keys())])
    fig, ax = plt.subplots(num_imgs // 4, 4, figsize=(16, num_imgs), sharex = True, sharey = True)
    ax = ax.flatten()

    max_imgs = (num_imgs // 4) * 4

    for i, (nft, bbox) in enumerate(bbox_imgs.items()):
        if i > max_imgs - 1 or i >= len(bbox_imgs.keys()) - 1:
            break

        ax[i].imshow(bbox)
        if overlay:
            ax[i].imshow(masks[nft])

        ax[i].set_title(nft)

    fig.suptitle(f'NFT Bounding Boxes - {wsiAnnot.wsi_name}')
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])


def load_annot(wsiAnnot_name, load_path = '/scratch/sghandian/nft/model_input_v2/wsiAnnots/', scale = 1.0):
    out_path = Path(load_path, f'scale_{scale}', f"{wsiAnnot_name}.pickle")
    with open(out_path, "rb") as file_to_read:
        try:
            loaded_object = pickle.load(file_to_read)
        except ModuleNotFoundError: #handle unpickling error (pickled with abs import instead of current relative import)
            file_to_read.seek(0)
            curr_dir = os.path.realpath(os.path.dirname(__file__))
            sys.path.insert(0, curr_dir)
            loaded_object = pickle.load(file_to_read)

    return loaded_object
#----------------------------Utility Functions----------------------------#



def main(args):

    SCALE = args.scale 
    ANNOT_PATH  = Path(args.annot_path)
    ZARR_PATH = Path(args.img_path, f'scale_{SCALE}/')
    SAVE_PATH = Path(args.save_path)

    #choose which wsi to create, can use indices for ease but only need to use one of the following
    WSI_NAME = args.wsi
    idx = args.idx

    filenames = os.listdir(ZARR_PATH)
    
    #if wsi_name passed in or idx given, load that one specifically and save it to disk
    if WSI_NAME or idx is not None:
        if WSI_NAME:
            file = WSI_NAME + '.zarr'
        else:
            file = filenames[idx]
        print(f"WSI name: {file}")

        imgPath = Path(ZARR_PATH, file)
        annotPath = Path(ANNOT_PATH, Path(file).with_suffix('.cz'))
        wsiAnnot_single = WSIAnnotation(imgPath, annotPath, SCALE, save = True,
                                        save_path=SAVE_PATH)
        return wsiAnnot_single

    #create and save each WSIAnnotation sequentially, unlike parallel_annots.py
    for file in filenames:
        imgPath = Path(ZARR_PATH, file)
        annotPath = Path(ANNOT_PATH, Path(file).with_suffix('.cz'))
        if not annotPath.isfile():
            print("""Skipping {file} as {annotPath} annotation does not exist.
                     Check whether annot_path arg is correct if this image does have an annotation.""")
            continue #skip files that do not have .cz associated
        wsiAnnot = WSIAnnotation(imgPath, annotPath, SCALE, save = True,
                                 save_path=SAVE_PATH)


if __name__ == '__main__':
    from argparse import ArgumentParser
    #add an argparser
    parser = ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True) #need to pass one of these args in!
    parser.add_argument("--annot_path", type=str, required=True) #need to pass one of these args in!

    parser.add_argument("--wsi", type=str) #need to pass one of these args in!
    parser.add_argument("--idx", type=int) #need to pass one of these args in!

    parser.add_argument("--scale", type=float, default = 1.0)
    parser.add_argument("--save_path", type=str, default='/scratch/sghandian/projects/nft/')
    args = parser.parse_args()
    #run on all images
    main(args)

"""
Example Usage:

python annot_conversion/point_to_mask.py \
    --img_path='/scratch/sghandian/nft/raw_data/wsis/zarr/' \
    --annot_path='/scratch/sghandian/nft/raw_data/annotations/updated_annots/' \
    --idx=0 --scale=0.1 --save_path='../data/'

"""