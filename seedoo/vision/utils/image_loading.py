import base64
import numpy as np
from PIL import Image
import io
from typing import List, Union
import pandas as pd
from urllib.parse import urlparse
import tempfile
import boto3
import os
import cv2
import hashlib
from io import BytesIO
from seedoo.vision.utils.region import Region

class PathFrameLoader:
    """
    This holds the video capture objects per video file. We don't want to create
    a new instance per file, so they are held globaly by the type instance.

    """
    INNER_CAP = {}

    def __init__(self, path: str, frame: int):
        """
        :param path: The path to the video file
        :param frame: The frame index of this specific image we want
        """
        self.frame = frame
        self.path = path
        if path not in PathFrameLoader.INNER_CAP:
            PathFrameLoader.INNER_CAP[path] = cv2.VideoCapture(path)

    def read(self, frame=None):
        if frame is None:
            frame = self.frame

        cap = PathFrameLoader.INNER_CAP[self.path]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, img = cap.read()
        img = img[..., ::-1].copy()
        return np.ascontiguousarray(img)

import base64
import boto3
import numpy as np
from PIL import Image
import tempfile


class ImageLoader:
    def __init__(self, path: str = '', image=None, callback=None, always_in_memory: bool = False):
        """
        :param path: The full path to the file
        :param image: An instance of the image
        :param callback: A lambda/function that will return the image (lazy evaluation) upon call.
        :param always_in_memory: Weather to keep the image in memory always, or delete it and reload on access
        """
        self.path = path
        self.callback = callback
        self.always_in_memory = always_in_memory
        self.__image = None

        if not path and image is not None:
            self.__image = image
            self.always_in_memory = True

        elif self.callback is None and image is not None:
            if path is not None:
                self.callback = lambda: cv2.imread(path)[..., ::-1]
            else:
                self.callback = lambda: image

        elif self.callback is None and path.startswith('s3://'):
            # Create a callback to download the image from S3 and return the path of the temp file
            self.callback = lambda: self.download_from_s3(path)

    def download_from_s3(self, s3_path):
        s3 = boto3.client('s3')
        parsed = urlparse(s3_path)
        bucket = parsed.netloc
        key = parsed.path[1:]
        hash_str = hashlib.sha1(s3_path.encode()).hexdigest()
        temp_dir = tempfile.gettempdir()
        name, extension = os.path.splitext(key)
        temp_file_path = os.path.join(temp_dir, f"imgloader_cache_{hash_str}.{extension}")
        if not os.path.isfile(temp_file_path):
            config = boto3.s3.transfer.TransferConfig(
                multipart_threshold=128,
                max_concurrency=10
            )
            s3.download_file(bucket, key, temp_file_path, Config=config)

        self.orig_path = self.path
        self.path = temp_file_path

        return cv2.imread(temp_file_path)[..., ::-1]

    @property
    def image(self):
        if self.__image is not None:
            return self.__image
        else:
            if self.callback is not None:
                image = self.callback()

            elif self.path:
                image = cv2.imread(self.path)[..., ::-1]
            else:
                temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                if isinstance(image, np.ndarray):
                    cv2.imwrite(temp.name, image[..., ::-1])
                else:
                    image.save(temp.name)

                self.path = temp.name

        if self.always_in_memory:
            self.__image = image

        return image

    def __str__(self):
        base64_string = self.to_base64()
        return f"<image src='data:image/png;base64,{base64_string}'>"

    @classmethod
    def from_base64(cls, base64_str):
        im = Image.open(BytesIO(base64.b64decode(base64_str)))
        return ImageLoader(image=np.array(im))

    @staticmethod
    def from_slice(self, region, image):
        if isinstance(image, Image.Image):
            image = np.array(image)

        callback = lambda x: np.array(region.slice(image.copy(), return_pillow=True, epsilon=(10, 10, 10, 10)))
        return ImageLoader(callback=callback)

    def to_base64(self):
        image = self.image
        if image is None:
            return ''

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        if len(image.shape) == 2:
            _, buffer = cv2.imencode(".jpg", image)
        else:
            if image.shape[0] <= 4:
                image = np.transpose(image, [1,2,0])
            _, buffer = cv2.imencode(".jpg", image[..., ::-1])

        buffer_io = io.BytesIO(buffer)
        base64_string = base64.b64encode(buffer_io.getvalue()).decode("utf-8")
        return base64_string

    def __getstate__(self):
        return self.path

    def __setstate__(self, state):
        self.path = state


    def draw_regions(self, regions, *args, **kwargs):
        image = self.image
        return ImageLoader(image = Region.draw(image, regions, *args, **kwargs))

class ListOfImageLoaders:
    def __init__(self, list_of_image_loaders: List[ImageLoader], columns_per_row: int = 3):
        self.list_of_image_loaders = list_of_image_loaders
        self.columns_per_row = columns_per_row

    def __str__(self):
        row = {}
        rows = []
        for i, loader in enumerate(self.list_of_image_loaders):
            row[str(i % self.columns_per_row)] = str(loader)
            if i % self.columns_per_row == 0:
                rows.append(row)
                if i > 0:
                    row = {}

        df = pd.DataFrame(rows)
        return df.to_html(escape=False).replace("\n", "")
