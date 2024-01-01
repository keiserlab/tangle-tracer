import torch
from tangle_tracer.modules import NFTDetector
from pathlib import Path
from ultralytics import YOLO
import zarr

class YOLOModule(NFTDetector):
    def __init__(self, *args, load_path='yolov8m.pt', devices = (1), **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.devices = self.trainer.devices #not necessary
        except:
            self.devices = devices
        # load_path = kwargs['load_path']
        self.model = YOLO(load_path, task = 'detect')
    
    def forward(self, x, verbose = True):
        results = self.model.predict(source = x, device = self.devices, nms = True,
                            conf = 0.05, iou = 0.2, show_conf = False, stream = True,
                            verbose = verbose)
        return results

    def predict_step(self, batch, batch_idx):
        print("THIS IS A BATCH:", batch)
        imgs, labels = batch
        results = self.forward(imgs)
        return results
        
    def on_predict_batch_end(self, batch_preds: torch.tensor, batch, batch_idx, dataloader_idx, save_img = False):
        imgs = []
        _, batch_data = batch
        randID = list(set(batch_data['Random ID']))[0]
        #move coords to cpu to create a df which we can use to groupby 
        bdx = batch_data['save_x'].cpu()
        bdy = batch_data['save_y'].cpu()
        for result, x, y in zip(batch_preds, bdx, bdy):
            save_path = Path(self.save_dir, 'predictions', randID, f"x-{x}_y-{y}").with_suffix('.txt')
            result.save_txt(save_path, save_conf=True)
            #slow process, need to create a worker dedicated to this
            if save_img:
                img = result.plot(line_width = 7) 
                imgs.append(img)
        if save_img:
            super().on_predict_batch_end(torch.tensor(imgs), batch, batch_idx, dataloader_idx)

    def _save_to_zarr(self, batch_preds:torch.tensor, coords:list[list[int, int]], wsi_name:str, roi_name:str = None):
            #do filtering prior to this to prevent two separate wsis from being saved
            assert self.save_dir is not None
            if roi_name:
                out_path = Path(self.save_dir, wsi_name, roi_name)
            else:
                out_path = Path(self.save_dir, wsi_name)
            synchronizer = zarr.ProcessSynchronizer(out_path.with_suffix('.sync'))
            # shutil.rmtree(out_path)
            out = zarr.open(out_path, mode = 'a', 
                            synchronizer = synchronizer) #use this to enable file locking
            
            #x should be a fixed number
            x, ys = coords

            #save to single column x, y_coords, and along last dim = (num_class = 1)
            # out.oindex[x, ys, :] = batch_preds

            tile_size = batch_preds.shape[1]
            #need to save these iteratively unfortunately, since outputs are shuffled/sharded
            for i, pred in enumerate(batch_preds):
                out.oindex[slice(x, x + tile_size), slice(ys[i], ys[i] + tile_size)] = pred#.permute(1, 2, 0)
                #save txt files too

        
if __name__ == "__main__":
    ym = YOLOModule()
