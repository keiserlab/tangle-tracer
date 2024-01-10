# Automated detection of neurofibrillary tangles in digitized whole slide images using deep learning

Welcome to the tangle-tracer repository! This repo is associated with the study [Ghandian et al. 2024](place-holder-url). The repo offers a few key utilities:
 - Reproducibility of the above study
 - A pipeline to transform point annotations done on AT8-stained WSIs to segmentation ground truth masks
 - UNet model weights which can be fine-tuned to segment neurofibrillary tangles (NFTs) in novel whole slide pathology images (WSIs)
 - YOLOv8 model weights which can be fine-tuned to detect NFTs in WSIs

Refer to the docstring at the bottom of each script to see an example of how to run it from the CLI.

If you encounter any issues or have any questions, feel free to open a [GitHub issue](https://github.com/keiserlab/tangle-tracer/issues) and we will do our best to address it.

## Installation
1. Run `conda create --name tangle_tracer_env python=3.9.7 pytorch=1.12.0 torchvision=0.13.0 cudatoolkit=11.3` to create the environment with a viable python/pytorch version.
2. Run `pip install -r REQUIREMENTS.txt`
3. Run `python setup.py build | python setup.py install` to allow for the internal import structure to work as intended.

## Preprocessing
1. Download WSIs (.czi) and Annotation Metadata (.cz) to local file system (ideally NVMe drive for fastest read/write) and convert to Zarr arrays at full resolution (0.11 microns/px). See an example of these files in `tangle-tracer/data/wsis/czi/` and `tangle-tracer/data/annotations/czi/`.
    - Convert WSIs to Zarr files with `annot_conversion.czi_to_zarr.py`.
2. (Optional) Create a parser for annotations and metadata. This step requires the most effort on the users' end.
    - Add parser to `WSIAnnotation.py` and **test** `WSIAnnotation` creation.
    - You can display the result of the point-to-mask procedure by calling `WSIAnnotation.plot_bboxes()` on a single `WSIAnnotation()`.
3. Run `parallel_annots.py` to create all `WSIAnnotation` objects using parallelized dask workers. Alternatively, you can use the `--input_type=rois` flag to generate `ROIAnnotation` objects instead. 
    - This may be more useful for applying the model to your own novel dataset. Creating either object also saves each ROI according to the associated WSI metadata in a separate directory of your choosing (i.e. `save_path` arg in both constructors). 
    - Both objects are also saved via pickle for easy access of all relevant information regarding a specific WSI/ROI (e.g. references to ROI zarr images, coordinates and bboxes of annotated NFTs, masks generated from DAB channel of each NFT, etc.) at `tangle-tracer/data/wsiAnnots/` or `tangle-tracer/data/roiAnnots/`.
    - A `WSIAnnotation` or `ROIAnnotation` can be loaded in with `WSIAnnotation.load_annot(wsi_name, load_path='tangle-tracer/data/wsiAnnots/', scale)`.
4. Run `mask_stitching.py` to process all `WSIAnnotation` objects and their respective ROIs. The goal of this script is to stitch together a ground truth ROI mask for each ROI referenced within each `WSIAnnotation`. 
    - The ground truth masks are saved in a separate directory (default = `data_dir/annotations/`)
    - These are randomly sampled from via a weighted random sampler during segmentation model training. Positive and negative labels (indicating presence of an NFT within that tile) are attached to non-overlapping tile coordinates during creation of `ROIDataModule`.

## Segmentation Model (UNet + ResNet50 encoder)
### Training Segmentation Model
1. (Optional) Run `train.py` with desired parameters to train a new model. The `data_dir` is the same `data_dir` defined for `mask_stitching.py`. 
    - Logs are dumped to a locataion of your choosing. Default location is `data_dir.parent/output/logs/`. Keep in mind that this is where checkpoints will be saved, so if this is altered, pass in the correct checkpoint directory to `inference.py`.

### Inference with trained Segmentation Model
1. Run `inference.py` on a novel WSI to generate a WSI-level segmentation output mask. 
    - Running it without defining a table_path will pass in the default path to the `tangle_tracer/data/tables/study_split.pickle`, allowing you to define which subset of the data to run inference on (i.e., train, val, test, batch1, batch2, or all). 
    - You can also pass in a custom `table_path` to run on a completely different set of WSIs, and filter this further with the `--wsis` parameter.

## Object Detection Model (YOLOv8)
### Preprocessing object detection dataset
1. Run `object_detection/create_tiles_yolo.py` with the same `data_dir` as created in the earlier preprocessing pipeline. This will create a set of tiles split by train/val/test in `save_dir`.
2. Set the `path` parameter in `yolo.yaml` to be the `save_dir` defined in the prior step. Other paths should not need to be changed as they are relative to this one.
3. (Optional) Check whether you can successfully create a `YOLOModule` and `YOLODataModule` by running the appropriate tests (currently in `yolo_datasets.py` and `modules_yolo.py`).
4. (Optional) To get object detection metrics for the trained **segmentation** model, you can run `object_detection/yolo_datasets.py`. Results will be saved in csv files within `tangle-tracer/data/tables/segobjdetect_results/`. Might want to add these results to .gitignore if you don't want to check it in to your repo.

### Training YOLOv8 Model
1. (Optional) Run `object_detection/train_yolo.py` to train a YOLOv8 model from scratch. Can also continue training or fine-tune a model by passing in a `ckpt_path` (e.g., `/tangle-tracer/data/weights/yolov8/best.pt`).
    - If the `config_path` is not recognized, define the `config_path` to be the location of the `yolo.yaml` file which had its `path` variable modified.
    - Can pass in a `hyperparams_path` which points to a yaml containing a configuration of YOLOv8 augmentation parameters. We have included a `best_hyperparams.yaml` file with the results of our best hyperparameter tuning run on this dataset at `tangle-tracer/data/tables/best_hyperparameters.yaml`.
    - Can run hyperparameter tuning as well with the `--tune` parameter.

### Inference with trained YOLOv8 Model
1. Run `object_detection/wsi_inference_yolo.py`. It largely has the same structure as `inference.py`, but runs with an ultralytics' trainer, so it does not have the same multiprocessing capabilities that the pytorch-lightning trainer offers with its parallel worker write capabilities. As a result, this is not quite as fast it could be, but still runs ~60% faster than the UNet model at about 20 minutes per WSI per GPU.
2. (Optional) Run `object_detection/roi_inference_yolo.py`.
    - Runs inference on the saved dataset of tiles only (i.e., train, val, test).
    - Outputs a set of directories in `save_path` corresponding to each WSI, with each folder containing zarr arrays for each ROI associated with that WSI. The zarr file is an **object detection agreement map**, with blue bboxes representing ground truth NFTs, and yellow bboxes representing predicted NFTs.

## Scoring
1. Run `scoring/calc_tissue_area.py` to calculate tissue pixel areas using histomicsTK.
2. Run `scoring/count_nfts.py` to count NFTs on a set of WSI-level segmentations. This is essentially a low-resolution blob detection done via cv2.
3. Run `scoring/count_bboxes.py` to parse the output .txt files from a WSI-level object detection inference procedure (i.e., outputs from running `object_detection/wsi_inference_yolo.py`).

## Figures (work-in-progress; unsure how to organize this within the repo)
1. Run `figures/CERAD-like_score_correlation.ipynb` to aggregate the results of all the above files into a single table and generate the figures included in the manuscript.