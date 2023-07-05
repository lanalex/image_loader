import base64
import uuid

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

import asyncio
from concurrent.futures import ThreadPoolExecutor

class ImageLoader:
    def __init__(self, path: str = '', image=None, callback=None, always_in_memory: bool = False, asynch_save = False):
        self.path = path
        self.callback = callback
        self.always_in_memory = always_in_memory
        self.__image = None

        if image is not None and isinstance(image, (Image.Image)):
            image = np.array(image)

        # Retrieve the caching directory from the environment variable
        self.caching_dir = os.getenv("IMAGE_CACHE_DIR", os.path.expanduser("~/temp_images_cache"))

        if not path and image is not None:
            if always_in_memory:
                self.__image = image
            else:
                # Save the image to the caching directory asynchronously
                if asynch_save:
                    self.loop = asyncio.get_event_loop()
                    self.loop.run_until_complete(self._save_image_to_cache(image))
                else:
                    self.save_to_temp(image)


        elif self.callback is None and image is not None:
            if path is not None:
                self.callback = lambda: cv2.imread(path, cv2.IMREAD_UNCHANGED)
            else:
                self.callback = lambda: image

        elif self.callback is None and path.startswith('s3://'):
            self.callback = lambda: self.download_from_s3(path)

    def save_to_temp(self, image):
        if image.dtype == np.uint8:
            temp_file_path = os.path.join(self.caching_dir, f"imgloader_cache_{uuid.uuid4().hex}.png")
            cv2.imwrite(temp_file_path, image[..., ::-1])
        else:
            temp_file_path = os.path.join(self.caching_dir, f"imgloader_cache_{uuid.uuid4().hex}.tiff")
            cv2.imwrite(temp_file_path, image, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

        self.path = temp_file_path
        self.__image = None

    async def _save_image_to_cache(self, image):
        temp_file_path = os.path.join(self.caching_dir, f"imgloader_cache_{uuid.uuid4().hex}.png")
        with ThreadPoolExecutor() as executor:
            await self.loop.run_in_executor(executor, self.save_to_temp, image)


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

        return cv2.imread(temp_file_path, cv2.IMREAD_UNCHANGED)[..., ::-1]

    @property
    def image(self):
        if self.__image is not None:
            return self.__image
        else:
            image = None
            if self.callback is not None:
                image = self.callback()

            elif self.path:
                image = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
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
        # Determine the correct progress bar to use
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

    def to_base64(self, down_scale = None):
        image = self.image
        if image is None:
            return ''


        #encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        if image.dtype != np.uint8:
            image = ((image / image.max()) * 255.0).astype('uint8')

        if down_scale:
            if isinstance(down_scale, (int, float)):
                scaling_ratio = down_scale / np.sqrt(image.shape[0] * image.shape[1])
                if scaling_ratio > 1.5:
                    down_scale = int(np.sqrt(image.shape[0] * image.shape[1]) * 1.5)

                ratio = image.shape[0] / image.shape[1]
                down_scale = (down_scale, int(down_scale * ratio))

            image = cv2.resize(image, down_scale)

        if len(image.shape) == 2:
            _, buffer = cv2.imencode(".jpg", image)
        else:
            if image.shape[0] <= 4:
                image = np.transpose(image, [1,2,0])
            image = image[..., ::-1]
            _, buffer = cv2.imencode(".png", image)

        buffer_io = io.BytesIO(buffer)
        base64_string = base64.b64encode(buffer_io.getvalue()).decode("utf-8")
        return base64_string

    def _repr_png_(self):
        # Convert numpy array to PIL Image
        img = Image.fromarray(self.image)

        # Create byte stream and save image as PNG
        b = BytesIO()
        img.save(b, format='png')

        # Return PNG image as bytes
        return b.getvalue()

    def __getstate__(self):
        return self.path

    def __setstate__(self, state):
        if os.environ.get('IMAGE_LOADER_OVERRIDE_ROOT_PATH', ''):
            new_root_path = os.environ['IMAGE_LOADER_OVERRIDE_ROOT_PATH']
            state = os.path.basename(state)
            state = os.path.join(new_root_path, state)

        self.__init__(path = state)

    def draw_regions(self, regions, *args, **kwargs):
        image = self.image
        return ImageLoader(image = Region.draw(image, regions, *args, **kwargs))

class ListOfImageLoaders:
    def __init__(self, list_of_image_loaders: List[ImageLoader], columns_per_row: int = 3):
        self.list_of_image_loaders = list_of_image_loaders
        self.columns_per_row = columns_per_row

    def to_df(self):
        row = {}
        rows = []
        for i, loader in enumerate(self.list_of_image_loaders):
            row[str(i % self.columns_per_row)] = str(loader)
            if i % self.columns_per_row == 0:
                rows.append(row)
                if i > 0:
                    row = {}

        df = pd.DataFrame(rows)
        return df

    def __str__(self):
        df = self.to_df()
        return df.to_html(escape=False).replace("\n", "")
