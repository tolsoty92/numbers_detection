# -*- coding:utf8 -*-
import PyOpenPose as OP
import joblib
import cv2

import numpy as np
import os

OPENPOSE_ROOT = "/home/user/openpose"

knn = joblib.load('/home/user/PycharmProjects/Hands and pose gestures recognition/classifiers/pose_classifier/knn_pose')

poses_lst = []
poses_dict = {( 'ruki_vniz', 'ladoni_na_urovne_loktey', 'lokti_vniz_ladon na urovne_plechey'): 'Come on!',
                                ('lokti_vniz_ladon na urovne_plechey', 'ladoni_na_urovne_loktey', 'lokti_vniz_ladon na urovne_plechey', 'ladoni_na_urovne_loktey'): 'Go away!'}



def kpt(key_points, img_width, img_height):
    if type(key_points) != type(None):
        key_points = key_points[0]

        diff_X_p2_p3 = key_points[2][0] / img_width - key_points[3][0] / img_width
        diff_Y_p2_p3 = key_points[2][1] / img_height - key_points[3][1] / img_height
        diff_X_p3_p4 = key_points[3][0] / img_width - key_points[4][0] / img_width
        diff_Y_p3_p4 = key_points[3][1] / img_height - key_points[4][1] / img_height
        diff_X_p2_p4 = key_points[2][0] / img_width - key_points[4][0] / img_width
        diff_Y_p2_p4 = key_points[2][1] / img_height - key_points[4][1] / img_height
        diff_X_p8_p4 = key_points[8][0] / img_width - key_points[4][0] / img_width
        diff_Y_p8_p4 = key_points[8][1] / img_height - key_points[4][1] / img_height
        diff_X_p8_p3 = key_points[8][0] / img_width - key_points[3][0] / img_width
        diff_Y_p8_p3 = key_points[8][1] / img_height - key_points[3][1] / img_height

        diff_X_p5_p6 = key_points[5][0] / img_width - key_points[6][0] / img_width
        diff_Y_p5_p6 = key_points[5][1] / img_height - key_points[6][1] / img_height
        diff_X_p6_p7 = key_points[6][0] / img_width - key_points[7][0] / img_width
        diff_Y_p6_p7 = key_points[6][1] / img_height - key_points[7][1] / img_height
        diff_X_p5_p7 = key_points[5][0] / img_width - key_points[7][0] / img_width
        diff_Y_p5_p7 = key_points[5][1] / img_height - key_points[7][1] / img_height
        diff_X_p11_p7 = key_points[11][0] / img_width - key_points[7][0] / img_width
        diff_Y_p11_p7 = key_points[11][1] / img_height - key_points[7][1] / img_height
        diff_X_p11_p6 = key_points[11][0] / img_width - key_points[6][0] / img_width
        diff_Y_p11_p6 = key_points[11][1] / img_height - key_points[6][1] / img_height

        kp = [diff_X_p2_p3, diff_Y_p2_p3,
                    diff_X_p3_p4, diff_Y_p3_p4,
                    diff_X_p2_p4, diff_Y_p2_p4,
                    diff_X_p8_p4, diff_Y_p8_p4,
                    diff_X_p8_p3, diff_Y_p8_p3,
                    diff_X_p5_p6, diff_Y_p5_p6,
                    diff_X_p6_p7, diff_Y_p6_p7,
                    diff_X_p5_p7, diff_Y_p5_p7,
                    diff_X_p11_p7, diff_Y_p11_p7,
                    diff_X_p11_p6, diff_Y_p11_p6]
        #
        kp = np.array(kp).reshape(1, 20)

        return kp
    else:
        return None

def pose_consistently(pose_dict, pose_lst):
    #  Определяем, была ли заранее продемонстрирована
    #  последовательность жестов.
    #  gest_dict -  словарь жестовых последовательностей
    #  gest_list - обновляемый список фиксируемых камерой
    #  зарезервированных жестов.
    pose_combination = None
    for combination in pose_dict:
        if len(combination) <= len(pose_lst):
            found = True
            p = pose_lst[:]
            for comb_num in range(len(combination)):
                if combination[comb_num] in p:
                    p = p[p.index(combination[comb_num]):]
                else:
                    found = False
            if found:
                pose_combination = pose_dict[combination]
                print(pose_combination)
                pose_lst = []
    return  pose_lst, pose_combination


def run(cap):
    op = OP.OpenPose((640, 320), (240, 240), (640, 360), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0, False,  OP.OpenPose.ScaleMode.ZeroToOne, False, False)
    cv2.namedWindow( "OpenPose result", cv2.WINDOW_NORMAL)
    comb = ''
    poses_lst = []
    while True:
        try:
            ret, frame = cap.read()
            rgb = frame

        except Exception as e:
            print("Failed to grab", e)
            break

        op.detectPose(rgb)

        Keypoints = op.getKeypoints(op.KeypointType.POSE)[0]
        res = op.render(rgb)
        kp = kpt(Keypoints, 1280, 720)
        if type(kp) != type(None):
            pose = knn.predict(kp)
            pose = str(pose[0])
        else:
            pose = ''
        if len(poses_lst) > 24:
            poses_lst = poses_lst[1:]
        poses_lst.append(pose)
        poses_lst, combination = pose_consistently(poses_dict, poses_lst)
        if type(combination) != type(None):
            comb = combination
        cv2.putText(res, comb, (20, 80), 0, 1, (0, 255, 0),thickness=4)
        cv2.putText(res, pose, (20, 40), 0, 1, (0, 0, 255), thickness=4)

        cv2.imshow("OpenPose result", res)
        key = cv2.waitKey(1)
        if key & 255 == 32:
            paused = not paused

        if key & 255 == 27:
            break

if __name__ == '__main__':
    # video = cv2.VideoCapture('./video2/' + 'ruki_vstoroni.mp4')
    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    run(video)