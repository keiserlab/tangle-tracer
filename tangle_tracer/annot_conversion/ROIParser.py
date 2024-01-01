import numpy as np
import pandas as pd
from pathlib import Path

class ROIParser():
    def __init__(self, annot_dir):
        self.regions = self.populate_data_structure(Path(annot_dir))

    def get_regions(self):
        return self.regions
    
    def populate_data_structure(self, ROI_directory):
        """
        Populates the data structure with ROI information from the given ROI directory.

        Args:
            ROI_directory (pathlib.Path): Path to the directory containing the ROI files.

        Returns:
            dict: The populated data structure.
        """
        data_structure = {}
        #init regions portion of ds, no need for nfts as these are local nfts not global nfts !
        data_structure['regions'] = {}
        roi_files = ROI_directory.glob('roi*-corners.txt')
        for i, roi_file in enumerate(roi_files):
            roi_name = roi_file.stem.split('-')[0][3:]

            # Load the ROI array and annotations
            roi_img_path = Path(ROI_directory, f'roi{roi_name}.png') # change to ZARR later?

            annotations_file = Path(ROI_directory, f'roi{roi_name}.txt')
            annotation_dict = self.parse_nft_table(annotations_file)

            #Load corner information
            roi_geometry = self.parse_roi_corners(roi_file)

            # Store the ROI data in the data structure
            data_structure['regions'][f"rectangle_{i}"] = {
                'nfts': annotation_dict,
                'Left': int(float(roi_geometry['Left'])), 
                'Top': int(float(roi_geometry['Top'])), 
                'Width': int(float(roi_geometry['Width'])),
                'Height': int(float(roi_geometry['Height'])),
                'Rotation': float(roi_geometry['Rotation']),
                'Path': roi_img_path,
            }
        
        return data_structure

    def parse_nft_table(self, file_path):
        """
        Parses a table of values from a text file into a pandas DataFrame.

        The table should be formatted with space-separated values in the following order:
        Label y1 x1 y2 x2

        Args:
            file_path (str): Path to the text file containing the table.

        Returns:
            pandas.DataFrame: The parsed table as a pandas DataFrame.
        """
        # Read the table from the text file
        df = pd.read_csv(file_path, delimiter=' ', header=None, names=['Label', 'y1', 'x1', 'y2', 'x2'])

        # Calculate the centroid and add as a new column
        df['center_coords'] = df[['y1', 'x1', 'y2', 'x2']].apply(
            lambda row: (int((row['y1'] + row['y2']) / 2), int((row['x1'] + row['x2']) / 2)), axis=1)
        
        #Convert to annotation dictionary congruent with prior data structure
        annot_dict = {}
        for i, row_data in df.iterrows():
                coords = row_data['center_coords']
                label = row_data['Label']
                annot_dict[f'annotation_{i}'] = [int(coords[0]), int(coords[1]), label]

        return annot_dict

    def parse_roi_corners(self, file_path):
        """
        Parses a table of roi-corners from a text file into a pandas DataFrame.

        The table should be formatted with space-separated values in the following order:
        [x1, y1, x2, y2, x3, y3, x4, y4]

        Args:
            file_path (str): Path to the text file containing the table.

        Returns:
            geometry (dict): The parsed table as a dictionary with relevant geometry for the ROI.
        """
        # Read the table from the text file
        df = pd.read_csv(file_path, delimiter=' ', header=None, names=['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])
        
        #Parse width and height from corner coordinates
        def calculate_width_height(corner_coords):
            """
            Calculates the width and height of an object given the corner coordinates of a rotated rectangle.

            Args:
                corner_coords (numpy.ndarray): Array of corner coordinates in the order [x1, y1, x2, y2, x3, y3, x4, y4].
                Where (x1, y1) is touching the top of the bounding rectangle (i.e. y1 = 0) and each point 
                is generated clock-wise from there.

            Returns:
                tuple: Width and height of the object.
            """
            # Convert the corner coordinates to a numpy array
            corners = np.array(corner_coords)

            # Calculate the width and height
            width = np.linalg.norm(corners[:2] - corners[2:4])
            height = np.linalg.norm(corners[2:4] - corners[4:6])

            return height, width
        
        #Calculate angle from corner information
        # def calc_angle_from_corners(corners):
        #     #get slope from two coordinates 
        #     m = (corners['y3'] - corners['y2']) / (corners['x3'] - corners['x2'])
        #     # m = (corners['y4'] - corners['y3']) / (corners['x4'] - corners['x3'])
        #     #convert slope to angle (rad) then to degrees
        #     angle = ((np.arctan(m) / (2*np.pi)) * 360) #- 90

        #     return angle
            #Calculate angle from corner information
        def calc_angle_from_corners(pairs):
            #get slope from two coordinates
            print(pairs, pairs[-1], pairs[-2])
            x4, y4 = pairs[-1]
            x3, y3 = pairs[-2]
            m = (y4 - y3) / (x4 - x3)
            #convert slope to angle (rad) then to degrees
            angle = ((np.arctan(m) / (2*np.pi)) * 360)
            return angle
        
        # #Convert to ROI dictionary congruent with prior data structure
        # geometry = {}
        # for i, row_data in df.iterrows():
        #     #calculate width and height from corners
        #     width, height = calculate_width_height(row_data.values)
        #     angle = calc_angle_from_corners(row_data)
        #     if row_data['x1'] == 0:
        #         x, y = row_data['x1'], row_data['y1']
        #     elif row_data['y1'] == 0:
        #         x, y = row_data['y1'], row_data['x1']
        #     elif row_data['x2'] == 0:
        #         x, y = row_data['x2'], row_data['y2']
        #     else:
        #         x, y = row_data['x1'], row_data['y1']
        #         assert x == 0

        #     geometry['Left'] = x
        #     geometry['Top'] = y
        #     geometry['Width'] = width
        #     geometry['Height'] = height
        #     geometry['Rotation'] = angle
            
        # return geometry
        #Convert to ROI dictionary congruent with prior data structure
        geometry = {}
        for i, row_data in df.iterrows():
            pairs = np.array(row_data.values).reshape(-1, 2)
            #for the first one, do an array pivot
            def arr_pivot(arr, i):
                new_arr = np.concatenate([arr[i:], arr[:i]])
                return new_arr
            height, width = calculate_width_height(row_data.values)
            if row_data['x1'] != 0:
                print(i)
                pairs = arr_pivot(pairs, 3)
                height, width = width, height
            #calculate width and height from corners
            angle = calc_angle_from_corners(pairs)
            
            #find smallest y in (x, y) pairs
            top_index = np.argmin(pairs[:, 1])
            left_index = np.argmin(pairs[1, :])
            x, y = pairs[top_index]
            assert y == 0
            
            #assign to dict
            geometry['Width'] = width
            geometry['Height'] = height
            geometry['Rotation'] = angle
            geometry['Left'] = x
            geometry['Top'] = y
            
        return geometry    
    
