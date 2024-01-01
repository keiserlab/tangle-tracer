import unittest
from pathlib import Path
import torch
from tangle_tracer.object_detection.modules_yolo import YOLOModule

class YOLOModuleTestCase(unittest.TestCase):
    """YOLOModuleTestCase class."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rand_data = torch.rand(1, 3, 1024, 1024)
        cls.device = "cuda:0"
        
        parentPath = (Path(__file__).resolve().parent.parent) 
        cls.ckpt_path = Path(parentPath, 'data/weights/yolov8/best.pt')

    def test_YOLOModule_cpu(self):
        model = YOLOModule(load_path=self.ckpt_path)
        results = list(model(self.rand_data))
        r_img = results[0].cpu()
        # r_img = r.plot()
        assert (str(r_img.shape), r_img.shape) == (str(r_img.shape), (1, 3, 1024, 1024))
        # assert r_img.shape == (1, 3, 1024, 1024)
    
    def test_YOLOModule_gpu(self):
        model = YOLOModule(load_path=self.ckpt_path, devices=0)
        self.rand_batch = torch.rand(4, 3, 1024, 1024, device=self.device)
        out = torch.tensor(model(self.rand_batch))
        assert out.shape == (4, 1, 1024, 1024)