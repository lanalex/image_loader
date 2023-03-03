#from __future__ import annotations
from shapely.geometry import box as shapely_box
import shapely.geometry
from shapely.geometry import Polygon
from typing import List, Union
import torchvision.transforms.functional as Fv
import pandas as pd
import torch.nn.functional as F
import scipy
import numpy as np
from PIL import Image
import skimage
from skimage.measure._regionprops import RegionProperties
from shapely.geometry import box
import re
import cv2
import torch
from shapely import affinity


def to_polygon(x):
    assert len(x) in [4, 8]
    if len(x) == 4:
        return box(x[0], x[1], x[0] + x[2], x[1] + x[3])
    elif len(x) == 8:
        return Polygon([(x[2 * i], x[2 * i + 1]) for i in range(4)])


def string_convert_to_polygon(points, rotation=None):
    if isinstance(points, str):
        points = points.split(",")
        points = [float(i) for i in points]
        points[0] = int(np.floor(points[0]))
        points[1] = int(np.floor(points[1]))
        points[2:] = [np.rint(i) for i in points[2:]]

    if len(points) == 4:
        result = box(*points)
    else:
        columns = points[::2]
        rows = points[1::2]
        polygon = list(zip(columns, rows))
        result = shapely.geometry.Polygon(polygon)

    if rotation:
        result = affinity.rotate(result, rotation, 'center')

    return result


class Region:
    def __init__(self, row: int = None, column: int = None, width: int = None, height: int = None, new_bbox=None,
                 rotation=None, polygon = None,
                 mask: np.ndarray = None, region_props: RegionProperties = None, metadata={}):

        self._children = []
        self.polygon = None
        if metadata is None:
            metadata = {}

        self.box = None
        if new_bbox is not None:
            new_bbox = string_convert_to_polygon(new_bbox, rotation)
            self.box = new_bbox
            self.polygon = self.box
        elif polygon is not None:
            polygon = string_convert_to_polygon(polygon, rotation)
            self.polygon = polygon
            new_bbox = box(*p.bounds)
            self.box = new_bbox

        if new_bbox:
            if not isinstance(new_bbox, shapely.geometry.Polygon):
                self.box = box(*new_bbox)

            column, row, max_column, max_row = self.box.bounds
            width = max_column - column
            height = max_row - row

        if region_props is not None:
            if self.box is None:
                min_row, min_column, max_row, max_column  = region_props.bbox
                width = max_column - min_column
                height = max_row - min_row
                self.row = min_row
                self.column = min_column
                self.width = width
                self.height = height
                self.box = shapely_box(self.column, self.row, self.column + width, self.row + height)
            mask = region_props.image

        height = int(height)
        width = int(width)

        if row is not None:
            if self.box is None:
                self.box = shapely_box(column, row, column + width, row + height)
            self.row = row
            self.column = column
            self.width = width
            self.height = height
            if self.polygon is None:
                self.polygon = self.box

        if mask is None:
            self.mask = np.zeros(shape=(height, width))
        else:
            self.mask = mask

        self._region_image_properties = region_props

        self.row = int(self.row)
        self.column = int(self.column)
        self.meta_data = metadata

    @staticmethod
    def from_bbox_string_array(string_array):
        """
        Load from bbox string array where the format is:
        [top left column, top left row, bottom right column]
        :param string_array:
        :return:
        """
        return Region(new_bbox=[int(i) for i in re.findall('([0-9]+)', string_array)])

    @property
    def region_props(self) -> RegionProperties:
        """
        returns the skimage regionproperties object that corresponds to this region in the original image.
        """
        return self._region_image_properties

    def slice(self, img, copy: bool = False, epsilon = (0, 0, 0, 0),
              return_pillow=False): # -> Union[np.ndarray, Image, torch.Tensor] Union[np.ndarray, Image, torch.Tensor]:
        """
        Return the slice of the img in that area the return type is the same as the input type
        epsilon: (top pad, left pad, bottom pad, right pad) -
        """
        top_pad, left_pad, bottom_pad, right_pad = epsilon
        if isinstance(img, (torch.Tensor, np.ndarray)):
            if len(img.shape) > 2:
                # Decide if channels last (ie last shape is small or equal to 4) - probably
                if img.shape[-1] <= 4:
                    axis_order = np.arange(0, len(img.shape)).tolist()
                    if isinstance(img, np.ndarray):
                        img = img.transpose([axis_order[-1]] + axis_order[0:-1])
                    elif isinstance(img, torch.Tensor):
                        img = img.permute([axis_order[-1]] + axis_order[0:-1])
                    else:
                        raise ValueError(f'Invalid type passed: {type(img)}, expected one of np.ndarry or torch.Tensor')

                result = img[..., max(self.row - top_pad, 0):max(0, min(self.row + self.height + bottom_pad + top_pad,
                                                                        img.shape[1])),
                         max(self.column - left_pad, 0):max(0, min(self.column + self.width + right_pad + left_pad,
                                                                   img.shape[2]))]
            else:
                result = img[self.row - top_pad:self.row + self.height + bottom_pad,
                         self.column - left_pad:self.column + self.width + right_pad]

            if copy:
                if isinstance(result, np.ndarray):
                    result = np.ascontiguousarray(result.copy())
                else:
                    result = result.clone()

            if return_pillow:
                result = Fv.to_pil_image(torch.tensor(result))

            return result

    @staticmethod
    def from_connected_components(mask: np.ndarray, image: np.ndarray = None): #-> List[Region]:
        labels = cv2.connectedComponentsWithStats(mask, connectivity=8)
        results = []
        for region_props in skimage.measure.regionprops(labels[1], intensity_image=image):
            results.append(Region(region_props=region_props))

        return results

    def add_child(self, child):
        """
        Allow to add a child region for this region. Meaning another region
        that represents the same object (for example using object tracking)
        as this region in the image. For example in object tracking scenario,
        the first anchor object will be this object and subsquenst instances
        of the same object (say car) in the subsequent frames, will be children
        """
        self._children.append(child)

    def __eq__(self, other_region):
        """
        Override eq operator so if two regions have at least 50% overlap, they are considered the same object
        """
        iou = self.box.intersection(other_region.box).area / self.box.union(other_region.box).area
        return iou > 0.5

    @staticmethod
    def group_regions_df(df: pd.DataFrame, coordinate_columns = ['xmin', 'ymin', 'xmax', 'ymax'], cluster_column_name: str= 'cluster_id', \
                         max_distance_in_pixels: float = 10.0) -> pd.DataFrame:
        """
        Group close rectangels into one, assigning them the same cluster_id
        :param df: The dataframe
        :param coordinate_copumns: The columns that repreesnt the coordiantes
        :param max_distance_in_pixels: the maximum distance to consider two rectangle to be the same
        :return: the same dataframe, with a new column
        """

        if len(df) == 0:
            return df

        if isinstance(coordinate_columns, list):
            m = df[coordinate_columns].values
        else:
            """
            In case we have a column, that each row in that column represents a list of coordinates, then 
            the values will return a list of lists. or list of np.array
            """
            m = df[coordinate_columns].values

            """
            if its a list of np.ndarray, we want to create a single np.ndarray so we use hstack (horizontal sstack )
            to stack them one on top of eache other to get one matrix of num_rows = len(df) and num_columns = m[0].shape[0]
            """
            if isinstance(m[0], np.ndarray):
                m = np.hstack(m)
            else:
                """
                if its a list of lists, then we will acheive the same thing using np.array(m). 
                """
                m = np.array(m)

        m = m.astype('float')
        m = scipy.spatial.distance_matrix(m, m)
        m2 = np.zeros_like(m)
        m2[np.where(m < max_distance_in_pixels)] = 1
        np.fill_diagonal(m2, 1)
        components, labels = scipy.sparse.csgraph.connected_components(csgraph=m2, directed=False, return_labels=True, connection='strong')
        df[cluster_column_name] = labels
        return df

    @staticmethod
    def assign_single_object_id(df: pd.DataFrame) -> pd.DataFrame:
        """
        This functions assumes the dataframe has an object_id column and a cluster_id column name
        :param df: The dataframe we want to re-assign object_ids
        :return: The new dataframe with re-assigned ojbect_ids
        """
        if len(df) == 0:
            df['object_id'] = ''
            return df
        df['object_id'] = df.groupby(['cluster_id'])['object_id'].transform(lambda x: x.min())

        return df

    @staticmethod
    def draw(image, regions, labels: List[str] = [],
             colors: List[tuple] = [],
             fonts: List[int] = [cv2.FONT_HERSHEY_SIMPLEX], thickness: int = 1, overlay_transparency = 0.0):

        """

        :param image: The image to draw on
        :param regions: The list of regions to draw
        :param labels: The labels of the regions (needs to be same lenght and regions)
        :param colors: The colors (needs to be same length as labels)
        :param fonts: The font to use (Same length as regions)
        :param thickness:  The thickness of the box
        :param with_overlay: Draw a transparent (default is 0.0 - no fill) fill in the box same color as colors, WARNING: slows down performance.
        :return:
        """
        image = np.array(image).copy()

        if not labels:
            labels = [""] * len(regions)

        if not colors:
            colors = [(30, 255, 30)] * len(regions)

        if len(fonts) == 1:
            fonts = fonts * len(regions)

        offset = 0
        for current_region, label, color, font in zip(regions, labels, colors, fonts):
            if 'label' in current_region.meta_data:
                label = current_region.meta_data['label']
            if 'color' in current_region.meta_data:
                color = current_region.meta_data['color']

            if not isinstance(label, dict):
                label = {'label' : label}

            image = cv2.rectangle(image, (current_region.column, current_region.row, current_region.width, current_region.height), color)
            offset = 0
            for k,v in label.items():
                final_label = f"{k}: {v}"
                label_width, label_height = cv2.getTextSize(final_label, font, 0.8, thickness)[0]
                image = cv2.putText(image, final_label, (current_region.column + current_region.width + 10, current_region.row - 10 + offset + label_height), font, 0.8, color, thickness)
                offset += label_height + 20

            polygon = current_region.polygon

            if overlay_transparency > 0:
                exterior = [np.array(polygon.exterior.coords).round().astype(np.int32)]
                alpha = overlay_transparency
                overlay = image.copy()
                cv2.fillPoly(overlay, exterior, color=color)
                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
                image = cv2.polylines(image, exterior,
                                      True, color, thickness)

        return image


if __name__ == "__main__":
    import os
    points = '69.09780643249724,720.0,71.66209386281844,697.7761732852014,65.42382671480482,682.1805054151664,63.3444043321324,666.5848375451296,76.86064981949676,656.187725631773,93.49602888086883,654.1083032491006,108.05198555957031,645.7906137184145,110.13140794224091,630.1949458483778,118.44909747292695,615.6389891696781,134.04476534296373,612.5198555956704,153.0,610.2000000000007,160.3000000000011,621.2000000000007,153.7992779783417,648.9097472924223,153.7992779783417,665.5451263537943,156.91841155234943,682.1805054151664,165.23610108303546,696.7364620938661,174.5935018050568,710.2527075812304,181.3416273257426,720.0'
    points = points.split(",")
    points = [float(i) for i in points]

    r = Region(new_bbox=points)
    b = r.box.bounds
    im = cv2.imread(os.path.expanduser("~/Downloads/region_test_image.png"))
    im2 = Region.draw(im, [r])
    cv2.imwrite("./output.png", im2)
    print(b)
    import glob
    from dateutil import parser
    import pandas as pd
    import os
    files = os.listdir(os.path.expanduser("~/Downloads"))
    files = [f for f in files if f.endswith('0000.mp4')]
    rows = []
    for f in files:
        elements = f.split("-")
        elements[3] = '20' + elements[3]
        bus_id = "-".join(elements[0:3])
        video_time = "T".join(elements[3:5])
        rows.append({'bus_id' : bus_id, 'video_time' : parser.parse(video_time), 'file' : f})

    df = pd.DataFrame(rows)
    df = df.sort_values(by = ['bus_id', 'video_time'], ascending = [True, True])
    with open(os.path.expanduser("~/Downloads/file_list.txt"), "w") as f:
        for file in df.file.values:
            f.write(f"file '{file}'\n")

    "ffmpeg -f concat -safe 0 -i filelist.txt -c:v  copy london0.mp4"


    r = Region(new_bbox='1115.166015625,95.1103515625,1199.013671875,182.7119140625')
    p = Polygon([(0, 0), (1.0, 1.0), (1.0, 0)])
    print(p.bounds)
    b = box(*p.bounds)
    print(b.intersection(b).area)
