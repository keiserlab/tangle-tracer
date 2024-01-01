import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
import wandb

from argparse import ArgumentParser
from nft_datasets import ROIDataModule
from modules import NFTDetector
from pathlib import Path


def main(args):
    pl.seed_everything(42)

    #define logging directory
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = Path(Path(args.data_dir).parent, 'output/logs/')
    os.makedirs(log_dir, exist_ok = True)

    job_name = f"Reannot_{args.optim}_LR{args.lr}_WD{args.weight_decay}_MM{args.momentum}_Alpha{args.alpha}"
    wandb_logger = WandbLogger(project='NFTDetector', 
                               job_type='train', 
                               name = job_name,
                            #    group = 'Sweep1', #remove after hyperparam opt
                            #    name = 'Reannot_200epoch_lr1e-5_UNet_resnet50_tversky_imgnorm_static_WRS0.5%',
                               log_model=True, save_dir=args.log_dir)

    checkpoint_callback = ModelCheckpoint(monitor='valid_loss', #can monitor valid_dataset_iou as well
                                          filename='NFT_trial-{epoch:02d}-{valid_loss:.2f}',
                                          save_top_k=2, 
                                          save_last=True,
                                          mode = 'min',
                                         )  

    early_stop_callback = EarlyStopping('valid_loss', patience=30)

    model = NFTDetector(optim = args.optim, 
                        lr = args.lr,
                        weight_decay = args.weight_decay,
                        momentum = args.momentum,
                        )#.load_from_checkpoint(checkpt_path)
    
    dm = ROIDataModule(data_dir = args.data_dir, batch_size= args.batch_size, 
                       num_workers=args.num_workers, imagenet_norm=True,
                       train_filter_proportion=1.0, 
                       train_tile_filter_proportion=1.0,
                       train_tile_ablate_proportion=0.0)

    dm.prepare_data()
    
    trainer = Trainer(max_epochs=args.max_epochs,
                    accelerator = 'gpu',
                    devices = args.devices,
                    strategy = 'ddp_find_unused_parameters_false',
                    logger=wandb_logger,
                    callbacks=[                                
                                checkpoint_callback,
                                early_stop_callback, #can't use this and SWA
                                # StochasticWeightAveraging(swa_lrs=1e-4),
                                TQDMProgressBar(refresh_rate=5),
                                ],
                    enable_checkpointing = True,
                    precision = 16,  #16-bit precision!
                    )
    
    trainer.fit(model, datamodule=dm)
    trainer.validate(datamodule=dm, ckpt_path='best') #this is default behavior, but raises warning if not passed.
    trainer.test(datamodule=dm, ckpt_path='best')

    wandb.finish()



if __name__ == "__main__":
    parser = ArgumentParser()
    # add program level args
    parser.add_argument("--data_dir", type=str, default = '/scratch/sghandian/nft/model_input_v2')
    parser.add_argument("--batch_size", type=int, default = 8)
    parser.add_argument("--num_workers", type=int, default = 20)
    parser.add_argument("--log_dir", type=str, 
                        help= """Defines save location for logs. Default is data_dir.parent/output/logs/.
                                 If you have permission issues with saving files here or concerns about disk space 
                                 in this directory, consider changing it to another logging directory.""")

    # add all trainer options
    # parser = Trainer.add_argparse_args(parser)
    # parser.add_argument("--accelerator", default = 'gpu')
    # parser.add_argument("--devices", default = 1)
    # parser.add_argument("--precision", default = 32)
    # parser.add_argument("--strategy", default = 'ddp')

    #argparsing for wandb sweep;
    # would add model specific args if there were any
    parser = NFTDetector.add_model_specific_args(parser)
    # parser.add_argument("--lr",  type=float, default=1e-6)
    # parser.add_argument("--optim",type=str, default= "adamw")
    # parser.add_argument("--weight_decay", type=float, default=0.01)
    # parser.add_argument("--momentum",type=float, default= 0.5)
    # parser.add_argument("--loss_func", type=str, default=["tversky"])
    # parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--devices", nargs='+', type=int, default=[0,1])

    
    #begin program
    args = parser.parse_args()
    #original dataset
    # DATA_DIR = Path('/scratch/sghandian/nft/model_input')
    #reannotation dataset
    # DATA_DIR = Path('/scratch/sghandian/nft/model_input_v2')
    #gutman dataset
    # DATA_DIR = Path('/scratch/sghandian/nft/gutman_input')

    #TEST SET:
    # ['1-102-Temporal_AT8', '1-154-Temporal_AT8', '1-271-Temporal_AT8', '1-343-Temporal_AT8', '1-693-Temporal_AT8', '1-717-Temporal_AT8', '1-907-Temporal_AT8']

    main(args)



