from pathlib import Path
import pytorch_lightning as pl


from tangle_tracer.modules import NFTDetector
from tangle_tracer.nft_datasets import ROIDataModule, GeneralROIDataModule

def main(args):
    #original NFT dataset directory
    # data_dir = Path('/scratch/sghandian/nft/model_input')
    #reannotation dataset
    data_dir = Path(args.data_dir)

    #best checkpt after reannotation and retraining
    # model_dir = '/scratch/sghandian/nft/output/logs/NFTDetector/'
    # model_checkpt = '0fnaalid/checkpoints/NFT_trial-epoch=83-valid_loss=0.12.ckpt' #F1=0.56
    model = NFTDetector().load_from_checkpoint(Path(args.ckpt_path))
    # model.save_dir = Path('/scratch/sghandian/nft/output')

    if args.general_roi_dm:
        print("Creating generalized ROI dataset!")
        dm = GeneralROIDataModule(data_dir = data_dir,
                        csv_file=args.csv_path,
                        scale = args.scale,
                        batch_size = 4,
                        num_workers = 16,
                        imagenet_norm=args.imagenet_norm)
    else:
        dm = ROIDataModule(data_dir = data_dir,
                           csv_file=args.csv_path,
                           scale=args.scale,
                           batch_size= 4,
                           num_workers= 16,
                           imagenet_norm = args.imagenet_norm)
    dm.prepare_data()
    dm.setup(stage = 'fit')
    dm.setup(stage = 'test')

    trainer = pl.Trainer(max_epochs=200,
                    accelerator = 'gpu',
                    devices = args.devices,
                    strategy = 'ddp_find_unused_parameters_false',
                    precision = 16, 
                    )
    trainer.test(model, datamodule = dm)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--csv_path", type=str, default='NFT_Case_List_Batch1.csv')
    # parser.add_argument("--save_dir", type=str)
    parser.add_argument("--imagenet_norm", default = True, action="store_true")
    parser.add_argument("--general_roi_dm", default = False, action="store_true")
    parser.add_argument("--scale", type=float, default=1.0)
    parser = pl.Trainer.add_argparse_args(parser) #allows one to change trainer args
    args = parser.parse_args()
    main(args)
    
    """
    Example Usage:

    python tests/metric_check.py --data_dir='/scratch/sghandian/nft/model_input_v2' \
        --ckpt_path='/scratch/sghandian/nft/output/logs/NFTDetector/0fnaalid/checkpoints/NFT_trial-epoch=83-valid_loss=0.12.ckpt' \
        --devices="0"

    python tests/metric_check.py --data_dir='data/' \
        --ckpt_path='/scratch/sghandian/nft/output/logs/NFTDetector/0fnaalid/checkpoints/NFT_trial-epoch=83-valid_loss=0.12.ckpt' \
        --devices="0"
    """