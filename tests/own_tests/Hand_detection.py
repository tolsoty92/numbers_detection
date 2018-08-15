# -*- coding:utf8 -*-
import cv2
import numpy as np
import os
from time import time
from modules.Hands import HandDetector
from modules.WebCamera import VideoStream
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Camera init
width, height = 640, 480
stream = VideoStream(width=width, height=height, camera_num=0)
RUN = True
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

detector = HandDetector()

TF = True

while RUN:
    gesture = None
    t = time()
    ret, img = stream.get_img()
    if ret:
        if TF:
            # Looking for hands
            actual_boxes = detector.detect_hands(img, im_widh=width, im_height=height)
            if len(actual_boxes) > 0:
                boxes = actual_boxes
                for box in boxes:
                    op_box = detector.tf_box_to_op_box(box, padding=1)
                    x, y, dx, dy = op_box
                    cv2.rectangle(img, (x, y), (x + dx, y + dy), (0, 0, 255), 5)
                

    # Calculate FPS
    t = time() - t
    fps = 1.0 / t
    # Visualization
    cv2.putText(img, 'FPS = %f' % fps, (20, 20),
                        0, 0.5, (0, 0, 255), thickness=4)

    cv2.imshow('Video', img)
    if cv2.waitKey(10) & 0xFF == 27:
        RUN = not RUN

cv2.destroyAllWindows()
stream.stop()
