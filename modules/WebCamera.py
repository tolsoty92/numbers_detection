import cv2
from threading import Thread
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class VideoStream:
    def __init__(self, width=640, height=480, camera_num=0):
        # Initialize the video camera stream
        self.stream = cv2.VideoCapture(camera_num)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_img(self):
        ret, img = self.stream.read()
        img = cv2.resize(img, (640, 480))
        return ret, img

    def stop(self):
        self.stream.release()

if __name__ == '__main__':
    description = \
    "Module WebCamera read data from webcam."
    print(description)
