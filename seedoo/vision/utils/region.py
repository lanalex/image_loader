#from __future__ import annotations
from shapely.geometry import box as shapely_box
import shapely.geometry
import colorsys
from shapely.geometry import Polygon
from typing import List, Union
import torchvision.transforms.functional as Fv
import pandas as pd
import numpy as np
import torch.nn.functional as F
import scipy
from shapely import affinity
from scipy.sparse.csgraph import connected_components
import scipy.spatial
import numpy as np


import numpy as np
from PIL import Image
from PIL import Image, ImageDraw
import json

import skimage
from skimage.measure._regionprops import RegionProperties
from shapely.geometry import box
import re
import cv2
import torch
from shapely import affinity
from scipy.spatial import ConvexHull
from itertools import combinations
from shapely.geometry import MultiPoint



def is_collinear(points):
    for combination in combinations(points, 3):
        x1, y1 = combination[0]
        x2, y2 = combination[1]
        x3, y3 = combination[2]

        # Calculate the area (should be zero if the points are collinear)
        area = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

        if area != 0:
            return False

    # If we haven't returned yet, then all areas were zero, so the points are collinear
    return True
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

    if len(points) == 4 and not isinstance(points[0], list):
        result = box(*points)
    else:
        if isinstance(points[0], list):
            result = shapely.geometry.Polygon(points)
        else:
            columns = points[::2]
            rows = points[1::2]
            polygon = list(zip(columns, rows))
            result = shapely.geometry.Polygon(polygon)

    if rotation:
        result = affinity.rotate(result, rotation, 'center')

    return result

class RegionStub():
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._cached = None

        #magic_methods = ['__add__', '__sub__', '__mul__', '__truediv__']  # add more if needed
        #for method in magic_methods:
        #    def wrapper(self, other):
        #        print(f"Calling {method} with {other}")
        #        original_method = getattr(other, method)
        #        return original_method(self, other)

        #    setattr(self, method, wrapper)

    def __getattr__(self, name):
        return getattr(self.instance, name)

    @property
    def instance(self):
        if not self._cached:
            r = Region(*self.args, **self.kwargs)
            self._cached = r

        return self._cached

    def __getstate__(self):
        return {'args' : self.args, 'kwargs' : self.kwargs}

    def __setstate__(self, state):
        self.args = state['args']
        self.kwargs = state['kwargs']

class Region:
    def __init__(self, row: int = None, column: int = None, width: int = None, height: int = None, new_bbox=None,
                 rotation=None, polygon = None,
                 mask: np.ndarray = None, region_props: RegionProperties = None, metadata={}, simplified = True):

        self._inner = None
        self._children = []
        self._polygon = None
        self.simplified = False
        self._box = None

        if metadata is None:
            metadata = {}

        self.box = None
        if new_bbox is not None:
            new_bbox = string_convert_to_polygon(new_bbox, rotation)
            self.box = new_bbox
            self.polygon = new_bbox
        elif polygon is not None:
            if not isinstance(polygon, shapely.geometry.Polygon):
                polygon = string_convert_to_polygon(polygon, rotation)

            self.polygon = polygon
            new_bbox = box(*self.polygon.bounds)
            self.box = new_bbox

        if new_bbox:
            if not isinstance(new_bbox, shapely.geometry.Polygon):
                self.box = box(*new_bbox)

            column, row, max_column, max_row = self.box.bounds
            width = max(max_column - column, 14)
            height = max(max_row - row, 14)

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
                # get the coordinates of the pixels in the region
                coords = region_props.coords

                # Check if the coordinates span more than a point in any direction
                if len(coords) >= 3:
                    self.polygon = MultiPoint(coords).convex_hull
                else:
                    self.polygon = self.box

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

    @property
    def polygon(self):
        polygon = self._polygon
        return polygon

    @property
    def box(self):
        return self._box

    @polygon.setter
    def polygon(self, value):
        self._polygon = value

    @box.setter
    def box(self, value):
        self._box = value

    @property
    def xywh_bbox(self):
        return [self.row, self.column, self.width, self.height]

    def __str__(self):
        return self.__repr__()

    def __sub__(self, other: 'Region') -> float:
        """
        Override the subtraction operator to calculate the Euclidean distance
        between the centers of two Region instances.

        Args:
            other (Region): The other Region instance to calculate the distance from.

        Returns:
            float: The Euclidean distance between the centers of the two Region instances.

        Raises:
            TypeError: If the 'other' parameter is not an instance of the Region class.
        """
        if not isinstance(other, Region):
            raise TypeError("Unsupported operand type(s) for -: 'Region' and '{}'".format(type(other).__name__))

        center_self = self.box.centroid
        center_other = other.box.centroid

        distance = np.sqrt((center_self.x - center_other.x) ** 2 + (center_self.y - center_other.y) ** 2)
        return distance

    def iou(self, other: 'Region') -> float:
        polygon1 = self.polygon
        polygon2 = other.polygon

        if not polygon1.intersects(polygon2):  # if the polygons don't intersect return 0
            return 0

            # Calculate intersection and union areas
        intersection_area = polygon1.intersection(polygon2).area
        union_area = polygon1.area + polygon2.area - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area

        return iou

    def min_intersect(self, other: 'Region') -> float:
        polygon1 = self.polygon
        polygon2 = other.polygon

        if not polygon1.intersects(polygon2):  # if the polygons don't intersect return 0
            return 0

            # Calculate intersection and union areas
        intersection_area = polygon1.intersection(polygon2).area
        union_area = min(polygon1.area,polygon2.area)

        # Calculate IoU
        min_intersect = intersection_area / union_area

        return min_intersect

    def to_dict(self):
        polygon = self.polygon

        box = [int(i) for i in polygon.bounds]
        result = {'row': self.row, 'column': self.column, 'width': self.width, 'height': self.height, 'metadata': self.meta_data,
         'polygon': polygon.exterior.coords._coords.astype('float16').tolist(), 'box' : box}

        del polygon
        del box

        return result

    def __repr__(self):
        return json.dumps(self.to_dict())

    @staticmethod
    def from_bbox_string_array(string_array):
        """
        Load from bbox string array where the format is:
        [top left column, top left row, bottom right column]
        :param string_array:
        :return:
        """
        return RegionStub(new_bbox=[int(i) for i in re.findall('([0-9]+)', string_array)])

    @staticmethod
    def from_xyxy(x0, y0, x1, y1, metadata = {}):
        """
        upper left corner
        upper top corner
        lower right corner
        lower bottom corner 
        """
        return RegionStub(y0, x0, x1 - x0, y1 - y0, metadata = metadata)


    @staticmethod
    def from_xywh(x0, y0, w, h, metadata = {}, offset = None):
        """
        upper left corner
        upper top corner
        lower right corner
        lower bottom corner
        """
        if offset is not None:
            x0 = offset.column + x0
            y0 = offset.row + y0

        return RegionStub(y0, x0, w, h, metadata = metadata)



    @staticmethod
    def from_quadrilateral(quadrilateral, metadata = {}, min_inflate = 0):
        polygon = Polygon(quadrilateral)
        current_length = np.sqrt(polygon.area)
        if min_inflate > 0 and current_length < min_inflate:
            # Calculate the current length of the polygon
            # Calculate the scaling factor to achieve a length of at least 28
            scaling_factor = float(min_inflate) / current_length

            # Calculate the desired buffer width for symmetric inflation
            buffer_width = max((scaling_factor - 1) * current_length / 2, 4)

            # Inflate the polygon symmetrically
            polygon = polygon.buffer(distance=buffer_width)

        return RegionStub(polygon=polygon, metadata = metadata)

    @staticmethod
    def from_polygon(polygon, metadata = {}):
        return RegionStub(polygon=polygon, metadata=metadata)

    @property
    def region_props(self) -> RegionProperties:
        """
        returns the skimage regionproperties object that corresponds to this region in the original image.
        """
        return self._region_image_properties

    def slice(self, img, copy: bool = False, epsilon = (0, 0, 0, 0),
              return_pillow=False, is_masked = False, inflate=0.0, inflate_transparency = 0.5, return_mask = False): # -> Union[np.ndarray, Image, torch.Tensor] Union[np.ndarray, Image, torch.Tensor]:
        """
        Return the slice of the img in that area the return type is the same as the input type
        epsilon: (top pad, left pad, bottom pad, right pad) -
        """
        is_expanded = False
        if isinstance(img, Image.Image):
            img = np.array(img)

        if len(img.shape) == 2:
            is_expanded = True
            img = img[..., None]

        # Create a mask with the same size as the image
        mask = Image.new('F', img.shape[0:2], 0.2)  # Use 'F' mode for a floating-point mask
        draw = ImageDraw.Draw(mask)

        # Define the original polygon
        original_polygon = self.polygon

        # Inflate the polygon if needed
        if inflate != 0:
            area = original_polygon.length
            scaling_factor =   area * (1 + inflate) - area
            inflated_polygon = original_polygon.buffer(distance = scaling_factor)
        else:
            inflated_polygon = original_polygon

        # Draw the inflated polygon onto the mask with intensity 0.5
        draw.polygon(list(inflated_polygon.exterior.coords), fill=inflate_transparency)
        draw.polygon(list(original_polygon.exterior.coords), fill=1.0)

        # Apply the original mask to the inflated polygon mask, making the delta darker
        mask = np.array(mask)

        if is_masked:
            mask = mask.transpose(1,0)
            mask = mask[..., None]
            mask = np.repeat(mask, img.shape[2], axis=-1)
            img = mask * img

        polygon = inflated_polygon

        # Get the bounding rectangle coordinates (minx, miny, maxx, maxy)
        minx, miny, maxx, maxy = polygon.bounds

        # Calculate the row, column width, and height
        row = int(miny)
        column = int(minx)
        width = int(maxx - minx)
        height = int(maxy - miny)

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

                result = img[..., max(row - top_pad, 0):max(0, min(row + height + bottom_pad + top_pad,
                                                                        img.shape[1])),
                         max(column - left_pad, 0):max(0, min(column + width + right_pad + left_pad,
                                                                   img.shape[2]))]


                mask = mask[max(row - top_pad, 0):max(0, min(row + height + bottom_pad + top_pad,
                                                                   img.shape[1])),
                         max(column - left_pad, 0):max(0, min(column + width + right_pad + left_pad,
                                                              img.shape[2]))]

            else:
                result = img[row - top_pad:row + height + bottom_pad,
                         column - left_pad:column + width + right_pad]


            if copy:
                if isinstance(result, np.ndarray):
                    result = np.ascontiguousarray(result.copy())
                else:
                    result = result.clone()

            if return_pillow:
                result = Fv.to_pil_image(torch.tensor(np.ascontiguousarray(result.copy())))
            else:
                result = result.transpose(1, 2, 0)

                if is_expanded:
                    result = result[:, :]

            if return_mask:
                result = (result, mask)
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
    def highest_bbox_matches(df: pd.DataFrame, query_bbox: tuple, bbox_column: str = 'bbox', value_column = None,\
                             bbox_list: List = None, iou_threshold = 0.5) -> pd.DataFrame:
        """
        :param @bbox_column The name of the bbox column where each value format is [left, top, width, height] (can
        be either tuple or list)
        :param bbox_query: The query bbox in the same format as the column (left, top, width, height)
        """
        if df is not None and bbox_list is not None:
            raise ValueError('Use either d or bbox_list not both!')

        if bbox_list is not None:
            df = pd.DataFrame([{'bbox': bbox} for bbox in bbox_list])

        df = df.copy()

        query_region = Region.from_xywh(*query_bbox)
        if not isinstance(df[bbox_column].values[0], (Region, RegionStub,)):
            df['region_column'] = df[bbox_column].apply(lambda x: Region.from_xywh(*x))
        else:
            df['region_column'] = df[bbox_column]

        if value_column is None:
            df['iou'] = df['region_column'].apply(lambda r: r.min_intersect(query_region))
        else:
            df['iou'] = df[value_column]


        iou_non_zero_values = df[df['iou'] > 0].iou.values
        iou_values = df.iou.values
        threshold = iou_threshold
        #iou_values[iou_values < threshold] = -10000
        sorted_indices = np.argsort(iou_values)[::-1]  # Sort in descending order
        sorted_iou_values = iou_values[sorted_indices]

        # Find the elbow point
        gradient = np.gradient(sorted_iou_values)

        if np.max(np.abs(gradient)) <= 0.001:
            elbow_point = len(iou_values)
        else:
            second_derivative = np.gradient(gradient)
            second_derivative = np.diff(np.sign(second_derivative))

            elbow_point = np.where(second_derivative != 0)[-1]
            if len(elbow_point) == 0:
                elbow_point = 1
            else:
                elbow_point = elbow_point[-1] + 1

        elbow_point = max(elbow_point, 1)
        if elbow_point < 5:
            elbow_point = -1

        # Select the rows corresponding to the top bounding boxes up to the elbow point
        selected_rows = df.iloc[sorted_indices[0:-1]]
        selected_rows = selected_rows.query("iou >= @threshold")

        return selected_rows

    def to_xywh(self):
        return (self.column, self.row, self.width, self.height)

    @staticmethod
    def combine_highest_scoring_bboxes(df: pd.DataFrame, bbox_column: str, score_column: str,
                                       max_relative_distance: float = 0.7) -> (tuple, pd.DataFrame):
        # Sort the DataFrame by score in descending order
        sorted_df = df.sort_values(by=score_column, ascending=False)
        scores = sorted_df[score_column].values
        original_columns = df.columns.values.tolist()
        grouped_df = Region.group_regions_df(sorted_df, "bbox", cluster_column_name="region_group",
                                             max_relative_distance=max_relative_distance, score_column_name=score_column)

        original_columns.append("region_group")
        grouped_df['region_group_score_median'] = grouped_df.groupby(['region_group'])[score_column].transform('mean')
        grouped_df['region_column'] = grouped_df['bbox'].apply(lambda x: Region.from_xywh(*x))
        #grouped_df = grouped_df[grouped_df['region_group_score_median'] >= grouped_df[score_column].max() * 0.7]

        filtered_regions = grouped_df.region_column.values.tolist()

        # Combine the selected bounding boxes into a new bounding box
        lefts, tops, widths, heights = zip(*[r.to_xywh() for r in filtered_regions])
        new_left = min(lefts)
        new_top = min(tops)
        new_right = max(left + width for left, width in zip(lefts, widths))
        new_bottom = max(top + height for top, height in zip(tops, heights))
        new_width = new_right - new_left
        new_height = new_bottom - new_top

        combined_bbox = (new_left, new_top, new_width, new_height)

        return combined_bbox, grouped_df[original_columns]

    @staticmethod
    def group_regions_df(df: pd.DataFrame, coordinate_columns=['xmin', 'ymin', 'xmax', 'ymax'],
                         cluster_column_name: str = 'cluster_id',
                         max_relative_distance: float = 0.1, score_column_name = None, score_tolerance = 0.65) -> pd.DataFrame:

        if len(df) == 0:
            return df

        scores = None
        # Extract bounding box coordinates
        if isinstance(coordinate_columns, list):
            m = df[coordinate_columns].values
        elif isinstance(coordinate_columns, str):
            col_values = df[coordinate_columns].values
            if isinstance(col_values[0], (tuple, list)):
                m = np.array([[x[0], x[1], x[0] + x[2], x[1] + x[3]] for x in col_values])
            elif isinstance(col_values[0], Region):
                m = np.array([[x.column, x.row, x.column + x.width, x.row + x.height] for x in col_values])

            else:
                raise ValueError("Unsupported format for coordinate_columns.")
        else:
            raise ValueError("coordinate_columns must be a list of column names or a single column name.")

        if score_column_name is not None:
            scores = df[score_column_name].values

        # Calculate the distance matrix
        m = m.astype('float')
        distance_matrix = scipy.spatial.distance_matrix(m, m)

        # Calculate the diagonal length for each bounding box
        diagonals = np.sqrt((m[:, 2] - m[:, 0]) ** 2 + (m[:, 3] - m[:, 1]) ** 2)
        max_score = None

        if scores is not None:
            max_score = np.max(scores)

        # Calculate the relative distance matrix
        relative_distance_matrix = np.zeros_like(distance_matrix)
        for i in range(len(m)):
            if scores is None or (scores[i] >= max_score * score_tolerance):
                for j in range(len(m)):
                    max_diagonal_length = max(diagonals[i], diagonals[j])
                    relative_distance = distance_matrix[i, j] / (2 * max_diagonal_length)
                    if relative_distance < max_relative_distance:
                        if scores is None or scores[j] > max_score * score_tolerance:
                            relative_distance_matrix[i, j] = 1

        #np.fill_diagonal(relative_distance_matrix, 1)

        # Find connected components
        components, labels = connected_components(csgraph=relative_distance_matrix, directed=False, return_labels=True,
                                                  connection='strong')
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

        if image.dtype != np.uint8:
            image = ((image / image.max()) * 255.0).astype('uint8')

        if not labels:
            labels = [""] * len(regions)

        if not colors:
            colors = [(30, 255, 30)] * len(regions)

        if len(fonts) == 1:
            fonts = fonts * len(regions)

        def assign_colors(elements):
            color_dict = {}
            num_elements = len(elements)

            for i, element in enumerate(elements):
                # Generate a unique and easily distinguishable HSV color for each element
                hue = i / float(num_elements)
                saturation = 1.0
                value = 1.0

                # Convert the HSV color to BGR format
                r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
                color = (int(b * 255), int(g * 255), int(r * 255))
                color_dict[element] = color
            return color_dict

        offset = 0
        unique_labels = set([])
        for current_region, label, color, font in zip(regions, labels, colors, fonts):
            if 'label' in current_region.meta_data:
                label = current_region.meta_data['label']

            if isinstance(label, str):
                unique_labels.add(label)

        if len(unique_labels) > 0:
            label_to_color = assign_colors(list(unique_labels))

        for current_region, label, current_color, font in zip(regions, labels, colors, fonts):
            color = None
            if 'label' in current_region.meta_data:
                label = current_region.meta_data['label']
            if 'color' in current_region.meta_data:
                color = current_region.meta_data['color']
            elif len(colors) == 0:
                color = label_to_color.get(label, colors[0])

            if not isinstance(label, dict):
                label = {'label' : label}

            if not color:
                color = current_color

            #image = cv2.rectangle(image, (current_region.row, current_region.column, current_region.height, current_region.width), color)
            offset = 0

            polygon = current_region.polygon
            exterior = [np.array(polygon.exterior.coords).round().astype(np.int32)]

            for k,v in label.items():
                if v:
                    final_label = f"{v}"
                    label_width, label_height = cv2.getTextSize(final_label, font, 1.0, thickness)[0]
                    image = cv2.putText(image, final_label, (current_region.column + current_region.width // 2 - label_width, current_region.row - 10 + offset + label_height), font, 0.75, color, thickness)
                    offset += label_height + 20


            # Draw the transformed polygon on the image
            image = cv2.polylines(image, exterior, True, color, thickness)

            if 'overlay_transparency' in current_region.meta_data:
                current_overlay_transparency = current_region.meta_data['overlay_transparency']
            else:
                current_overlay_transparency = overlay_transparency

            if current_overlay_transparency > 0:
                alpha = current_overlay_transparency
                overlay = image.copy()
                cv2.fillPoly(overlay, exterior, color=color)
                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        return image



def draw_label_inside_polygon(image, bounding_rect, label_text):
    # Find the bounding rectangle of the polygon
    x, y, w, h = bounding_rect

    # Determine the resolution of the image
    image_height, image_width = image.shape[:2]

    # Calculate a resolution factor to take into account the image resolution
    resolution_factor = min(image_width, image_height) / 1000.0

    # Select font face and thickness
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = max(int(resolution_factor), 1)

    # Determine the initial font scale based on the resolution factor
    font_scale = resolution_factor * 2

    # Iterate to find the best font scale that fits within the bounding rectangle
    while True:
        text_size, _ = cv2.getTextSize(label_text, font_face, font_scale, font_thickness)
        text_width, text_height = text_size
        if text_width < w * 0.8 and text_height < h * 0.8:  # Ensure text occupies less than 80% of the bounding rectangle
            break
        font_scale -= 0.1 * resolution_factor

    # Calculate the center of the bounding rectangle
    center_x = x + w // 2
    center_y = y + h // 2

    # Calculate the position to start the text such that it is centered
    start_x = center_x - text_width // 2
    start_y = center_y + text_height // 2

    # Draw the text on the image
    cv2.putText(image, label_text, (start_x, start_y), font_face, font_scale, (0, 0, 255), font_thickness)

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
