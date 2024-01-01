import numpy as np
import zarr
import cv2
import os, shutil
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import zarr
from numcodecs import Blosc
import pandas as pd
from PIL import Image
import skimage
import pickle


#create a WSIAnnotation class
class ROIAnnotation():
    
    def __init__(self, img_path, nft_annot_path, corners, scale = 1.0, save = False, overwrite = True,
                 save_path = f'/scratch/sghandian/nft/gutman_input/'):
        """
        Initialize ROIAnnotation object. Takes in the 2 paths, one to the ROI image and another to the nft annotations.
        Also takes in a string list of corners in arbitrary order. Creates an ROIAnnotation object which generates a
        zarr array output representing the ROI at Path(save_path, 'images', img_path.stem).with_suffix('.zarr').

        img_path (str|Path): Path to ROI. Can be jpeg, jpg, png, or zarr. Assumes that the ROI image is cropped to the 
                            minimal region surrounding the ROI, with corner information indicating the corners lying on
                            the periphery of the input image corresponding to the ROI within the input image.
        nft_annot_path (str|Path): Path to nft point annotations in csv format. Table should have three columns indicating
                                   x and y coordinates, as well as Type of NFT (must be one of iNFT or Pre-NFT).
                                   By default, iNFTs are the only NFT that are segmented for training.
        corners (str): String version of a flattened array representing the coordinates of corners of the ROI.
                       These are used to apply a white boundary on the non-annotated region of the image outside of the ROI.
                       Ex: "x1 y1 x2 y2 x3 y3 x4 y4"
        scale (float): Scale at which input and output should be created if downscaling is desired for experimentation.
                       Not stable, not recommended to use.
        save (bool): Boolean indicating whether to save the ROIAnnotation itself at save_path. 
                     The zarr array is saved at save_path REGARDLESS.
        overwrite (bool): Boolean indicating whether to overwrite a potentially already-created zarr array from prior attempt.
                          Default behavior is to overwrite on each ROIAnnotation creation.
        save_path (str|Path): Base path at which to save the generated zarr array and the ROIAnnotation if save = True.
                              Saves at Path(save_path,'roiAnnots', f'scale_{self.scale}', Path(img_path).stem).with_suffix('.zarr').

        """
        img_path, nft_annot_path = Path(img_path), Path(nft_annot_path)
        self.corners = np.array(corners.split(' '), dtype = np.int32)
        self.annot_data = pd.read_csv(nft_annot_path, sep = ' ', names = ['x', 'y', 'Type'])
        self.roi_name = self.wsi_name = img_path.stem

        if img_path.suffix == '.zarr':
            self.roi = zarr.open(img_path, mode = 'r', dtype = 'i4')
        elif img_path.suffix in ['.jpg', '.png', '.jpeg']:
            self.roi = np.asarray(Image.open(img_path))
        else:
            raise ValueError("ROI image path not one of the following: .jpg, .jpeg, .png, or .zarr")

        self.scale = scale
        #define bbox size
        self.bbox_size = int(400 * self.scale)
        #define input tile size for model, pad ROI by this much + jitter amount
        self.input_tile_size = int(1024 * self.scale)
        #how much to vary random sampling coords of positive class
        # self.jitter_amount = self.input_tile_size // 2
        #calculate total padding size
        self.padding_size = self.input_tile_size // 2 #+ self.jitter_amount

        #need save_path for zarr dump of corrected ROIs
        self.save_path = save_path
        self.chunk_size = 1000 #used for zarr chunks
        self.overwrite = overwrite

        try:
            self.roi = self._get_roi(self.corners)
            self.bbox_dims = self._get_nft_bboxes(self.annot_data)
            self.bboxes = self._get_img_bboxes(self.bbox_dims)
            self.masks = self._draw_all_masks(biggest = False)
        except FileNotFoundError:
            print(f'No annotation data found for {self.roi_name}. \
                    Returning WSIAnnot with reference to zarr image only.')

        if save:
            self.save_annot()
            print("Successfully saved!", self.wsi_name)
                
    def _get_nft_bboxes(self, nft_annots):
        """
        Creates bbox DIMENSIONS around each NFT in a given ROI. Filters out NFTs 
        that are not within this ROI.

        Input:
            rectangle (str) - Index to self.annot_data['regions'] which contains
                              metadata specific to this ROI within the larger WSI.

        Returns:
            bboxes (dict) - Maps an NFT (str) to its bbox dimensions (dict).
        """
        
        bboxes = {}
        #bbox border in each direction from center
        border = int(self.bbox_size // 2)

        #check each global NFT annot for existence in this ROI (after rotation)
        for nft, row in nft_annots.iterrows():

            x, y = int(row['x'] * self.scale), int(row['y'] * self.scale)
            label = row['Type']

            if label != 'iNFT': 
                continue #skip any non iNFTs e.g. (pre-NFT)

            dims = {
                    'xmin': int(x - border),
                    'ymin': int(y - border),
                    'xmax': int(x + border),
                    'ymax': int(y + border),
                    'width': int(border * 2),
                    'height': int(border * 2),
                    }

            bboxes[nft] = dims 

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

        bboxes = modify_all_dims(bboxes, pad_amount = self.padding_size)

        return bboxes

    def _transform_rect(self, original_image, corners):
        #results in white borders
        mask = np.ones_like(original_image, dtype=np.uint8) * 255
        corners_array = np.array(corners).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [corners_array], color=(0, 0, 0))
        result_image = cv2.bitwise_or(mask, original_image)
        return result_image

    def _get_roi(self, corners):
        """
        Takes in the ROI annotations and generates a nested dictionary mapping rectangles to the ROI image,
        corrected for rotation, as well as the constitutent bbox dimensions for that specific rectangle.

        Input:
            corners (dict): Metadata corresponding to corners of ROI in original self.ROI.

        Returns:
            zarr_arr (zarr.array): Zarr array with black borders applied according to corner metadata.
        """

        img_slice = self._transform_rect(self.roi, corners) / 255.0
        
        #pad array by padding amount on each side, 
        padded_img = np.pad(img_slice, ((self.padding_size, self.padding_size),
                                        (self.padding_size, self.padding_size), (0, 0)),
                                        mode = 'constant', constant_values = 1)

        #convert img_slice to zarr file on disk
        self.out_path = Path(self.save_path, 'images', self.roi_name)#.with_suffix('.zarr')
        zarr_arr = self.__dump_to_zarr(padded_img, self.out_path)

        return zarr_arr

    def __dump_to_zarr(self, img: np.array, path: Path):
        compressor = Blosc(cname='lz4', clevel=1, shuffle=1)
        synchronizer = zarr.ProcessSynchronizer(path.with_suffix('.sync'))
        store = zarr.DirectoryStore(path)
        if self.overwrite:
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                pass
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
    

    def _load_bbox(self, bbox_object):
        """
        Takes in an annotation for a bounding box and retuns a slice of the rectangle image.

        Input:
            bbox_object (dict): Dictionary {annotation_# : xmin, ymin, xmax, ymax, width, height}

        Returns:
            box_slice (np.array): Slice of the image contained in the bounding box, padded to 400px if necessary.
        """
        rectangle_img = self.roi
        left = int(bbox_object['xmin'])
        top = int(bbox_object['ymin']) 
        width = int(bbox_object['width'])
        height = int(bbox_object['height'])
        #add edge case handling via checking coordinates, altering indexing and padding
        box_slice = rectangle_img[slice(np.max([0, top]), top + height),
                                  slice(np.max([0, left]), left + width), :].copy()
   
        return box_slice
                             

        
    def _get_img_bboxes(self, bboxes):
        """
        Takes in the dictionary of bbox dimensions,
        and produces all associated bbox images in a dictionary mapping annotation name to image.

        Input:
            bboxes (dict): Dictionary for a single ROI mapping each NFT annotation to bbox dimensions.
                Example:
                bboxes['annotation_0'] = {'xmin': 200, 'xmax': 600, 'ymin': 800, 'ymax': 1200}
        Returns:
            bbox_imgs (dict): Dictionary mapping an NFT bounding box annotation to an image slice.
                Example:
                bbox_imgs['annotation_0'] = bbox_arr
        """

        bbox_imgs = {}
        for bbox, dims in bboxes.items():
            bbox_imgs[bbox] = self._load_bbox(dims)
        return bbox_imgs
    

    def _create_blobs(self, mask, threshold=1000, max_distance = 75):
        """
        Takes in a binarized bbox image and generates a mask representing a single NFT within it. 
        """
        labels = skimage.measure.label(mask, connectivity=2, background=0)
        out_mask = np.zeros(mask.shape, dtype='uint8')
        sizes = {}
        distances = {}
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
            blob_center = np.float32([np.average(indices) for indices in np.where(labelMask == 255)])
            img_center = np.float32([200, 200])
            if numPixels > threshold:
                #enforce center bias
                distances[label] = (np.linalg.norm(blob_center - img_center))
                if np.linalg.norm(blob_center - img_center) < max_distance:
                    sizes[label] = numPixels
            elif np.linalg.norm(blob_center - img_center) < max_distance * 4 and numPixels > threshold * 0.25:
                distances[label] = (np.linalg.norm(blob_center - img_center))
            else:
                distances[label] = (np.linalg.norm(blob_center - img_center))

        
        #get the largest blob
        try:
            maxLabel = max(sizes, key = sizes.get)
            maxPixel = sizes.pop(maxLabel)

        except ValueError: #nothing found that meets ideal criteria
            maxLabel = min(distances, key = distances.get)
            maxPixel = distances.pop(maxLabel)
        
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

        # return allSizes, out_mask, (biggest_mask, biggest_mask_size)
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
                plt.imshow(image)
                plt.imshow(mask, cmap = 'binary')
                plt.show()    
        
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


    def save_annot(self):        
        """
        Dumps the WSIAnnot object at the default save location for the specific scale at which
        the original WSI was loaded in (self.scale). 
        
        Can be loaded again with WSIAnnotation.load_annot(WSI_NAME).
        """
        save_path = Path(self.save_path, 'roiAnnots', f'scale_{self.scale}')
        #create a directory if needed
        # if not os.path.exists(save_path): #might not need this line
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
