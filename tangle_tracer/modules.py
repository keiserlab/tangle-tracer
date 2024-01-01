import pytorch_lightning as pl
from typing import Any, Dict, List, Optional, Type
import torch
import segmentation_models_pytorch as smp
import wandb
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
import zarr
from pathlib import Path


class NFTDetector(pl.LightningModule):
    def __init__(self, save_dir = None, optim = 'adamw', lr = 1e-5,
                 weight_decay = 0.01, momentum = 0.1, 
                 loss_fn = 'tversky', alpha = 0.3):
        super().__init__()
        self.save_dir = save_dir
        self.optim = optim
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.loss_fn = loss_fn
        assert self.loss_fn in ['tversky']
        self.alpha = alpha

        encoder_name ='resnet50'
        self.model = smp.Unet(
            encoder_name=encoder_name,              # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",             # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                          # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                              # model output channels (number of classes in your dataset)
            )     
                   
        # for image segmentation dice loss could be the best first choice
        # self.loss_fn = smp.losses.JaccardLoss(smp.losses.BINARY_MODE, from_logits=True)
        if self.loss_fn == 'tversky':
            self.loss_fn = smp.losses.TverskyLoss(smp.losses.BINARY_MODE, from_logits=True,
                                                  alpha=alpha, beta=1-alpha)
        # self.loss_fn = smp.losses.FocalLoss(smp.losses.BINARY_MODE,
                                            #   alpha=0.15, gamma=2.0)                                              


    def forward(self, x):
        return self.model(x)

    def compute_loss(self, y_hat, y):
        return self.loss_fn(y_hat, y)


    def shared_step(self, batch, stage):
        imgs, masks = batch
        logits_masks = self(imgs)
        loss = self.compute_loss(logits_masks, masks)

        prob_masks = logits_masks.sigmoid().long()
        pred_masks = (prob_masks > 0.5)
    
        masks = masks.long()
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_masks, masks, mode="binary")
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        #add epsilon to get rid of nans
        eps = 1e-6
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])# + eps
        fp = torch.cat([x["fp"] for x in outputs])# + eps
        fn = torch.cat([x["fn"] for x in outputs])# + eps
        tn = torch.cat([x["tn"] for x in outputs])# + eps

        print(tp.shape, 'TP shape')

        #metric calculations
        # https://smp.readthedocs.io/en/latest/metrics.html#

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise", class_weights=[0.0, 1.0])
        macro_per_image_IOU = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise", class_weights=[0.0, 1.0])
        macro_dataset_IOU = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro", class_weights=[0.0, 1.0])

        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro", class_weights=[0.0, 1.0])

        #calculate other metrics
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro", class_weights=[0.0, 1.0])
        macro_f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro", class_weights=[0.0, 1.0])
        macro_per_image_f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro-imagewise", class_weights=[0.0, 1.0])


        #minimize false positives, beta < 1; NOT at pixel-level (tile level)
        # https://medium.com/@douglaspsteen/beyond-the-f-1-score-a-look-at-the-f-beta-score-3743ac2ef6e3
        fhalf_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=0.5, reduction="micro")

        #AUROC/AUPRC analysis
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
        sensitivity = smp.metrics.sensitivity(tp, fp, fn, tn, reduction="micro")
        specificity = smp.metrics.specificity(tp, fp, fn, tn, reduction="micro")

        micro_mIOU = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro", class_weights=[0.5, 0.5])
        macro_mIOU = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro", class_weights=[0.5, 0.5])



        metrics = {
            # f"{stage}_epoch_loss": outputs[-1]['loss'], #get the last recorded loss
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_f1_score" : f1_score,
            f"{stage}_fhalf_score" : fhalf_score,
            f"{stage}_recall" : recall,
            f"{stage}_precision": precision,
            f"{stage}_sensitivity" : sensitivity,
            f"{stage}_specificity": specificity,
            f"{stage}_macro_per_image_posIOU": macro_per_image_IOU,
            f"{stage}_macro_dataset_posIOU": macro_dataset_IOU,
            f"{stage}_macro_f1_score" : macro_f1_score,
            f"{stage}_macro_per_image_f1_score" : macro_per_image_f1_score,
            f"{stage}_micro_mIOU": micro_mIOU,
            f"{stage}_macro_mIOU": macro_mIOU,
        }
        
        self.log_dict(metrics, prog_bar=True, sync_dist=True)

        if stage == 'test':
            print("Testing...")
            confusion_matrix = {'tp': tp.sum().float().cpu(), 'fp': fp.sum().float().cpu(), 
                                'fn': fn.sum().float().cpu(), 'tn': tn.sum().float().cpu()}
            self.log_dict(confusion_matrix, sync_dist=True)

    def configure_optimizers(self):
        if self.optim == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), 
                                          lr=self.lr, weight_decay=self.weight_decay) #default is 0.01 or 1e-2
        elif self.optim == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), momentum=self.momentum,
                                          lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise Exception("No valid optimizer option passed in. Must be one of: ['adamw', 'sgd']")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, 0)
        return [optimizer], [scheduler]
    
    # def configure_optimizers(self):
    #     if self.optim == 'adam':
    #         optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    #     else:
    #         optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr) #default WD is 0.01 or 1e-2

    #     # else:
    #         # optimizer = Lars(self.parameters(), lr=self.lr)

    #     def warmup_fn(epoch):
    #         warmup_epochs = 10
    #         if epoch < warmup_epochs:
    #             return (epoch / warmup_epochs)*self.lr
    #         elif epoch == 0:
    #             return 1e-5
    #         else:
    #             return self.lr

    #     # Adding warmup scheduler
    #     warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)

    #     # After warmup, use a scheduler of your choice
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)

    #     return [optimizer], [{"scheduler": warmup_scheduler, "interval": "step"}, {"scheduler": scheduler, "interval": "epoch"}]

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def predict_step(self, batch, batch_idx):
        imgs, _ = batch #critical to unpack this! (won't use label for inference)
        pred = self(imgs) 
        out = pred.sigmoid()
        #added rounding here for preds for stability of plotting
        out = torch.round(out, decimals = 4)
        return out

    def on_predict_batch_end(self, batch_preds:torch.tensor, batch, batch_idx, dataloader_idx):
        """
        Called in the predict loop after each batch.

        In this case, we will be using it to "batch write" tiles. Since we can read them into memory in order,
        we can utilize larger zarr chunks to minimize IO calls and allow for more efficient writing processes.

        Inputs:
            batch_preds (Optional[Any]) The outputs of predict_step_end(test_step(x))=
            batch (Any) The batched data as it is returned by the test DataLoader.
            batch_idx (int) the index of the batch
            dataloader_idx (int) the index of the dataloader

        Outputs:
            None
        """

        _, batch_data = batch #unpack original batch to get metadata

        #move coords to cpu to create a df which we can use to groupby 
        batch_data['save_x'] = batch_data['save_x'].cpu()
        batch_data['save_y'] = batch_data['save_y'].cpu()
        batch_df = pd.DataFrame(batch_data)

        #map of randID (name) to df
        wsi_dfs = {}

        #handle the case that there are multiple wsis in one batch (can only be 2?)
        if batch_df['Random ID'].nunique() > 1:
            for randID, randID_group in batch_df.groupby('Random ID'): #COULD BE THE SAME ANYWAYS
                wsi_dfs[randID] = randID_group
        else:
            #if just one wsi, add it to wsi_dfs so that it's included in for loop below
            wsi_dfs[batch_df['Random ID'].iloc[0]] = batch_df
            
        #go through each split df (should be max of 2)
        for randID, df in wsi_dfs.items():
            #group by column so we can batch write the correct span of each
            for x, col_group in df.groupby('save_x'):
                preds = batch_preds[col_group.index]
                #preds need to be on cpu to be saved to disk, fix x for single column indexing
                try:
                    roi_name = col_group["ROI"].iloc[0]
                except KeyError:
                    roi_name = None
                self._save_to_zarr(preds.cpu(), [x, col_group['save_y'].values], randID, roi_name = roi_name)
        #done
    

    def _save_to_zarr(self, batch_preds:torch.tensor, coords:list[list[int, int]], wsi_name:str, roi_name:str = None):
        #do filtering prior to this to prevent two separate wsis from being saved
        assert self.save_dir is not None
        if roi_name:
            out_path = Path(self.save_dir, wsi_name, roi_name)
        else:
            out_path = Path(self.save_dir, wsi_name)
        synchronizer = zarr.ProcessSynchronizer(out_path.with_suffix('.sync'))
        out = zarr.open(out_path, mode = 'a',
                        synchronizer = synchronizer) #use this to enable file locking
        
        #x should be a fixed number
        x, ys = coords

        #save to single column x, y_coords, and along last dim = (num_class = 1)
        # out.oindex[x, ys, :] = batch_preds

        tile_size = batch_preds.shape[-1]
        if torch.cuda.device_count() > 1:
            #need to save these iteratively unfortunately, since outputs are shuffled/sharded
            for i, pred in enumerate(batch_preds):
                out.oindex[slice(x, x + tile_size), slice(ys[i], ys[i] + tile_size), :] = pred.permute(1, 2, 0)
        else:
            stacked_preds = torch.flatten(batch_preds, end_dim = 1).permute(1,0)
            out.oindex[slice(x, x + tile_size), ys] = stacked_preds
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("NFTDetector")
        parser.add_argument("--lr",  type=float, default=1e-5)
        parser.add_argument("--optim",type=str, default= "adamw")
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--momentum",type=float, default= 0.0)
        parser.add_argument("--loss_fn", type=str, default=["tversky"])
        parser.add_argument("--alpha", type=float, default=0.3)
        return parent_parser


# ------ Define custom callbacks for PTL ------ #

class LogPredictionSamplesCallback(pl.Callback):
    
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
        #log less frequently
        if trainer.current_epoch % 4 != 0:
            return None
        val_imgs, val_masks = batch
        val_imgs = self.val_imgs.to(device=pl_module.device)

        logits = pl_module(val_imgs)
        prob_masks = logits.sigmoid()
        # preds = (prob_masks > 0.5).float()
        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case
        
        # Let's log 20 sample image predictions from the first batch
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}' 
                        for y_i, y_pred in zip(y[:n], outputs[:n])]
            
            
            # Option 1: log images with `WandbLogger.log_image`
            WandbLogger.log_image(
                key='sample_images', 
                images=images, 
                caption=captions)


            # Option 2: log images and predictions as a W&B Table
            columns = ['image', 'ground truth', 'prediction']
            data = [[wandb.Image(x_i), y_i, y_pred] 
                    for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
            WandbLogger.log_table(
                key='sample_table',
                columns=columns,
                data=data)   



class ImagePredictionLogger(pl.Callback):
    def __init__(self, num_samples=8, logging_frequency = 4):
        super().__init__()
        self.num_samples = num_samples
        self.logging_frequency = logging_frequency
    
    def custom_load_batch(self, batch, pl_module):
        self.val_imgs, self.val_masks = batch

        #save imgs and ground truth masks to class variables
        self.val_imgs = self.val_imgs[:self.num_samples]
        self.val_masks = self.val_masks[:self.num_samples]

        #generate preds for batch = num_samples
        val_imgs = self.val_imgs.to(device=pl_module.device)
        logits = pl_module(val_imgs)
        prob_masks = logits.sigmoid()

        #save preds to class variable
        self.preds = (prob_masks > 0.5).float().cpu().squeeze().numpy()

        #move imgs and masks to cpu and permute
        self.val_imgs = self.val_imgs.cpu().permute(0,2,3,1).numpy()
        self.val_masks = self.val_masks.cpu().squeeze().numpy()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        #log less frequently
        if trainer.current_epoch % self.logging_frequency != 0:
            return None #do nothing for non-multiples of freq epochs
        #save imgs,masks,preds to class variables
        if batch_idx in [5, 10, 15, 20]:  
            self.custom_load_batch(batch, pl_module)

            for og_image, pred_mask, ground_truth_mask in zip(self.val_imgs, self.preds, self.val_masks):
                #check if ground_truth_mask has any NFT
                # if ground_truth_mask.sum() < 5:
                #     continue #skip this tile
                trainer.logger.experiment.log({
                    "my_image_key" : wandb.Image(og_image, masks={
                            "predictions" : {"mask_data" : pred_mask},
                            "ground_truth" : {"mask_data" : ground_truth_mask}
                        })
                    })
                    