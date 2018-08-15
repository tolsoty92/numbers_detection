import os
import cv2
import joblib
import numpy as np
import PyOpenPose as OP
from modules.Pose import Pose
from modules.WebCamera import VideoStream


# Constants.
WIDTH, HEGHT = 640, 360
CAM_NUM = 0
OP_ROOT= "/home/user/openpose"
POSE_KNN_PATH = './classifiers/pose_classifier/knn_pose'
POSE_DICT_PATH = './classifiers/pose_classifier/poses_combinatiob_dict'
WINDOW_NAME = 'Video'
OP_NET_INPUT = (640, 320)
OP_NET_RESOLUTION = (240, 240)


def main():
    # Init classes.
    camera = VideoStream(width=WIDTH, height=HEGHT,
                                           camera_num=CAM_NUM)
    ps = Pose()
    op = OP.OpenPose(
                                    OP_NET_INPUT, OP_NET_RESOLUTION,
                                    (WIDTH, HEGHT),"COCO",
                                    OP_ROOT + os.sep + "models" + os.sep,
                                    0, False,  OP.OpenPose.ScaleMode.ZeroToOne,
                                    False, False
                                    )
    # Load data and init key variables.
    pose_knn = joblib.load(POSE_KNN_PATH)
    pose_dict = joblib.load(POSE_DICT_PATH)
    poses_list = []     # Append by pose  frame by frame
    RUN = True

    while RUN:
        ret, img = camera.get_img()
        if not ret:
            raise Exception("Can't to find camera!")
        else:
            op.detectPose(img)
            key_points = op.getKeypoints(op.KeypointType.POSE)[0]
            kp_features = ps.compute_pose_features(key_points,
                                                                              WIDTH, HEGHT)

            if type(kp_features) != type(None):
                pose_position = pose_knn.predict(kp_features)[0]
            else:
                pose_position = ''

            if len(poses_list) > 30:
                poses_list = poses_list[1:]

            poses_list.append(pose_position)
            poses_list, combination =  ps.check_pose_consistently(poses_list,
                                                                                                       pose_dict)

            img = op.render(img)
            if type(combination) != type(None):
                msg = combination
            else:
                msg = ''
            cv2.putText(img, msg, (20, 80),
                                0, 1, (0, 255, 0), thickness=4)
            cv2.putText(img, pose_position, (20, 40),
                                0, 1, (0, 0, 255), thickness=4)
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, img)
            if cv2.waitKey(10) & 0xFF == 27:
                RUN = not RUN

if __name__ == '__main__':
    main()