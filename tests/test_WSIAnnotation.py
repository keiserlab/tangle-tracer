import unittest
from pathlib import Path

from tangle_tracer.annot_conversion.WSIAnnotation import WSIAnnotation, load_annot
from tangle_tracer.annot_conversion.ROIAnnotation import ROIAnnotation
# from tangle_tracer.annot_conversion import point_to_mask as p2m
import zarr

class WSIAnnotationTestCase(unittest.TestCase):
    """WSIAnnotationTestCase class."""
    @classmethod
    def setUpClass(self) -> None:
        #need an included zarr image and annotation for this to work... 
        self.wsi_name = '1-102-Temporal_AT8'
        self.scale = 0.1
        #define paths to data
        currPath = (Path(__file__).resolve().parent) 
        self.imgPath     = Path(currPath, '..', 'data', 'wsis', 'zarr', f'scale_{self.scale}', self.wsi_name).with_suffix('.zarr')
        self.savePath    = Path(currPath, '..', 'data')
        self.annotPath   = Path(currPath, '..', 'data', 'annotations', self.wsi_name).with_suffix('.cz')
        self.reannotPath = Path(currPath, '..', 'data', 'annotations', 'reannotations').with_suffix('.csv')
        
        #check that the files needed to init are existing
        assert (str(self.imgPath), self.imgPath.is_dir()) == (str(self.imgPath), True)
        assert (str(self.annotPath), self.annotPath.is_file()) == (str(self.annotPath), True)
        assert (str(self.reannotPath), self.reannotPath.is_file()) == (str(self.reannotPath), True)
    
    def test_wsiannotation_zarr_reading(self):
        z_img = zarr.open(self.imgPath, mode = 'r')
        self.assertIsInstance(z_img, zarr.core.Array)
        self.assertAlmostEqual(z_img.shape, (int(116676*self.scale), int(115406*self.scale), 3))

    def test_wsiannotation_save_and_load(self):
        """
        Test creation, saving, and loading of a WSIAnnotation.
        """
        wsiAnnot = WSIAnnotation(self.imgPath, self.annotPath, scale=self.scale, save = True,
                                 save_path=self.savePath, reannotations=self.reannotPath)
        self.assertIsInstance(wsiAnnot, WSIAnnotation)
        self.outPath = Path(self.savePath, 'wsiAnnots', f'scale_{self.scale}', self.wsi_name).with_suffix('.pickle')
        assert (str(self.outPath), self.outPath.is_file()) ==  (str(self.outPath), True)
        assert len(wsiAnnot.masks) == len(wsiAnnot.bboxes)
        mask_shapes = [mask.shape for mask in list(wsiAnnot.masks.values())]
        assert all([mask_shape == (int(400*self.scale), int(400*self.scale), 4) for mask_shape in mask_shapes])
                                                          
        loaded_wsiAnnot = load_annot(self.wsi_name,
                                         load_path=Path(self.savePath, 'wsiAnnots'),
                                         scale=self.scale)
        
        assert wsiAnnot.wsi_img.shape == loaded_wsiAnnot.wsi_img.shape
        assert wsiAnnot.wsi_name == loaded_wsiAnnot.wsi_name
        assert len(wsiAnnot.bboxes) == len(loaded_wsiAnnot.bboxes)
        assert len(wsiAnnot.masks) == len(loaded_wsiAnnot.masks)


    def tearDown(self):
        """Tear down the tests."""
        #could delete the WSIAnnot that was created.
        outPath = Path(self.savePath, 'wsiAnnots', f'scale_{self.scale}', self.wsi_name).with_suffix('.pickle')
        outPath.unlink(missing_ok = True)
        import shutil
        shutil.rmtree(Path(self.savePath, 'images', self.wsi_name),
                      ignore_errors=True)