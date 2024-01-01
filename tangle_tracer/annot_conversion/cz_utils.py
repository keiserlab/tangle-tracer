import xmltodict
from pathlib import Path

#define class with cz parsing functions
class CzParser():
    """
    Object containing parsed metadata for a CZ WSI. Use the method `get_regions()` to obtain
    a dictionary containing the coordinates, width, and height of each annotated rectangle,
    as well as the point coordinates of each NFT.
    """

    def __init__(self, CZ_PATH):
        self.metadata_name = Path(CZ_PATH)
        self.doc = xmltodict.parse(open(self.metadata_name, 'r', encoding='utf-8').read())
        
        self.rectangles = self._extract_rectangles()
        self.annotations = self._extract_annotations()
        self.scaling_factor = self._extract_scaling_factor()
        
        self.regions = {'regions': self.rectangles, 'nfts': self.annotations}
        
    def get_regions(self):
        return self.regions
        
    def _extract_rectangles(self):
        rectangles = {}
        dic_rectangle = self.doc['GraphicsDocument']['Elements']['Rectangle']

        for rect in range(0, len(dic_rectangle)):
            geometry = dic_rectangle[rect]['Geometry']
            attributes = dic_rectangle[rect]['Attributes']
            # text_elems = dic_rectangle[rect]['TextElements']['TextElement']
            rectangle_features = {'Left': int(float(geometry['Left'])), 
                                  'Top': int(float(geometry['Top'])), 
                                  'Width': int(float(geometry['Width'])),
                                  'Height': int(float(geometry['Height'])), 
                                  'Rotation': int(float(attributes.get('Rotation', 0))), 
                                #   'Position': text_elems.get('Position').split(',')
                                 }
            rectangles['rectangle_{}'.format(rect)] = rectangle_features
        return rectangles
    
    def _extract_annotations(self):
        annotations = {}
        try:
            dic_marker = self.doc['GraphicsDocument']['Elements']['Marker']
        #no annotations for this WSI
        except KeyError:
            return annotations
        
        for coordinate in range(0, len(dic_marker)):
            geometry = dic_marker[coordinate]['Geometry']
            annotations['annotation_{}'.format(coordinate)] = [int(float(geometry['X'])), int(float(geometry['Y']))]
        
        return annotations
            
    def _extract_scaling_factor(self):
        scaling_factor = self.doc['GraphicsDocument']['Scaling']['Items']['Distance'][0]['Value']
        scaling_factor = float(scaling_factor)

        return scaling_factor
