import numpy as np


class Pose():
    """ Class'Pose' work with pose's key points
            and finds sequences of gestures."""
    def compute_pose_features(self, key_points, img_width, img_height):
        if type(key_points) != type(None):
            key_points = key_points[0]
            kp = []
            main_points = ((2, 3), (3, 4), (2, 4), (8, 4), (8, 3),
                           (5, 6), (6, 7), (5, 7), (11, 7), (11, 6))

            for point in main_points:
                kp.append(
                    key_points[point[0]][0] / img_width -
                    key_points[point[1]][0] / img_width)
                kp.append(
                    key_points[point[0]][1] / img_height -
                    key_points[point[1]][1] / img_height)

            # Convert options list to numpy array

            kp = np.array(kp).reshape(1, 20)
            return kp
        else:
            return None

    def check_pose_consistently(self, poses_lst, pose_dict):

        pose_combination = None
        for combination in pose_dict:
            if len(combination) <= len(poses_lst):
                found = True
                p = poses_lst[:]
                for comb_num in range(len(combination)):
                    if combination[comb_num] in p:
                        p = p[p.index(combination[comb_num]):]
                    else:
                        found = False
                if found:
                    pose_combination = pose_dict[combination]
                    print(pose_combination)
                    poses_lst = []
        return poses_lst, pose_combination

if __name__ == '__main__':
    description =\
     "Module 'Pose' work with pose's key points and finds sequences of gestures."
    print(description)