import cv2
import numpy as np
import asyncio


class AsynchCameraCapture:

    def __init__(self, source):
        if '/dev' in source or "./" in source:
            #base_string = f'v4l2src device=/dev/video0 ! video/x-raw,width=1920,height=1080 ! jpegenc ! appsink'
            base_string = source
        elif 'rtsp://' in source:
            base_string = f"rtspsrc location={source} latency=2000 ! queue ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
        else:
            base_string = f'filesrc location={source} ! qtdemux ! queue ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! queue ! videoconvert ! queue ! video/x-raw, format=BGR ! appsink'

        base_string = source
        #cap = cv2.VideoCapture(base_string, cv2.CAP_GSTREAMER)
        import cv2


        cap = cv2.VideoCapture(base_string)

        max_largest = 640
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


        # Determine the largest dimension
        if original_height > original_width:
            # If height is the larger dimension
            aspect_ratio = (original_width + 1) / (original_height + 1)
            new_height = max_largest
            new_width = int(max_largest * aspect_ratio)
        else:
            # If width is the larger dimension or they are equal
            aspect_ratio = (original_height + 1) / (original_width + 1)
            new_width = max_largest
            new_height = int(max_largest * aspect_ratio)

        # Check if the largest dimension of the image is greater than max_largest
        if original_height < max_largest and original_width < max_largest:
            # Resize the image
            new_height = original_height
            new_width = original_width

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)
        cap.set(cv2.CAP_PROP_FPS, 3)

        self.captures = {'camera' : cap}


    @staticmethod
    async def read_frame(capture):
        # The method for using OpenCV grab() - retrieve()
        # We are not using read() here because, documentation insists that it is slower in multi-thread environment.
        capture.grab()
        ret, frame = capture.retrieve()
        return frame

    @staticmethod
    async def show_frame(window_name: str, frame: np.array):
        # Just making the OpenCV imshow awaitable in order to be able to run through asyncio
        cv2.imshow(window_name, frame)

    async def async_camera_gen(self):
        for camera_name, capture in self.captures.items():
            yield camera_name, capture