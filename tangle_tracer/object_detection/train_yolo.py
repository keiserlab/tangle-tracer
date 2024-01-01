from ultralytics import YOLO
from pathlib import Path
import os
import yaml

def main(args):
    config_path = Path(args.config_path)
    log_path = Path(args.log_path)
    hyperparams_path = Path(args.hyperparams_path)
    tune = args.tune

    #init yolo model from weights
    if args.ckpt_path:
        model =  YOLO(args.ckpt_path)
    else:
        os.makedirs(args.log_path, exist_ok=True)
        # model = YOLO(Path(args.log_path, 'yolov8n.pt')) #small YOLOv8
        model = YOLO(Path(args.log_path, 'yolov8m.pt')) #medium YOLOv8

    #can run hyperparam search w/ default ultralytics search
    if tune:
        results = model.tune(data=config_path, epochs=60, imgsz=1024,
                             iterations = 300, 
                             save_dir = log_path.as_posix(), project = 'NFTObjDetect',
                             name = 'NFTObjDetectTune', device=args.devices, batch=-1,
                             workers = args.num_workers)
    
    #just do a single training run with default params
    else:
        #if given params, use those instead
        if hyperparams_path:
            with open(Path(hyperparams_path).as_posix()) as f:
                aug_dict = yaml.load(f, Loader = yaml.FullLoader)
            results = model.train(data=config_path.as_posix(), epochs=args.max_epochs, imgsz=1024,
                            save_dir = log_path.as_posix(), project = 'NFTObjDetect',
                            name = 'NFTObjDetect', device=args.devices, batch=-1,
                            workers = args.num_workers, **aug_dict, optimizer = 'SGD')
        #pure default params + SGD
        else:
            results = model.train(data=config_path.as_posix(), epochs=args.max_epochs, imgsz=1024,
                            save_dir = log_path.as_posix(), project = 'NFTObjDetect',
                            name = 'NFTObjDetect', device=args.devices, batch=-1,
                            workers = args.num_workers, optimizer = 'SGD')    
    
    return results 


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default='yolo.yaml')
    parser.add_argument("--log_path", type=str, required = True)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--devices", nargs='+', type=int, default=[0,1])
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default = 16)
    parser.add_argument("--tune", default=False, action='store_true')


    args = parser.parse_args()
    main(args)

"""
Example Usage:
python object_detection/train_yolo.py --config_path='/srv/home/sghandian/lab-notebook-ghandian/projects/nft/object_detection/yolo.yaml' \
    --log_path='/scratch/sghandian/nft/yolo_input/models/NFTObjDetect'
    --hyperparams_path='/scratch/sghandian/nft/yolo_input/NFTObjDetect/tune4/best_hyperparameters.yaml'
"""