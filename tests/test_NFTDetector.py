import unittest
from pathlib import Path
import torch
from tangle_tracer.modules import NFTDetector

class NFTDetectorTestCase(unittest.TestCase):
    """NFTDetectorTestCase class."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = NFTDetector()
        cls.rand_data = torch.rand(1, 3, 1024, 1024)
        cls.device = "cuda:0"
        
        parentPath = (Path(__file__).resolve().parent.parent) 
        cls.ckpt_path = Path(parentPath, 'data/weights/UNet/best.ckpt')

    def test_NFTDetector_cpu(self):
        out = self.model(self.rand_data)
        assert out.shape == (1, 1, 1024, 1024)
    
    def test_NFTDetector_gpu(self):
        model = self.model.to(self.device)
        self.rand_batch = torch.rand(4, 3, 1024, 1024, device=self.device)
        out = model(self.rand_batch)
        assert out.shape == (4, 1, 1024, 1024)

    def test_NFTDetector_ckpt_loading(self):
        model = self.model.load_from_checkpoint(self.ckpt_path)
        #test on a tile containing an NFT?