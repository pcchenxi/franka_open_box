import math
import numpy as np
from time import sleep
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation
from travelbox import TravleBox
from franka import Franka
from franky import Affine, JointWaypointMotion, JointWaypoint, CartesianMotion, Robot, ReferenceType
import matplotlib.pyplot as plt

def norm_rotation(rot):
    z_rot = -math.pi/4 + rot
    twoPI = math.pi*2
    z_rot = z_rot - twoPI* math.floor((z_rot + math.pi) / twoPI)
    return z_rot

def get_target_idx(start_idx, path_x, path_y, path_z, robot_x, robot_y, robot_z):
    threshold_xy, threshold_pan = 0.01, math.pi/180

    idx = len(path_x) - 1
    for i in range(start_idx, len(path_x)):
        x, y, z = path_x[i], path_y[i], path_z[i]
        dist_xyz = np.linalg.norm([x-robot_x, y-robot_y, z-robot_z])
        # dist_pan = norm_rotation(pan) - robot_pan
        if dist_xyz < threshold_xy: # and dist_pan < threshold_pan:
            idx = i+1
        else:
            idx = i
        break
    return idx

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--host', default='172.16.3.1', help='FCI IP of the robot')
    args = parser.parse_args()

    franka = Franka(args.host)
    franka.set_home_pose()

    travelbox = TravleBox(0.495, 0.3, 0.4, 0.54, 0.4, 0.03)  # 0.5 0.375
    travelbox.generate_path(direction='counterclockwise') #counterclockwise  clockwise

    ee_trans, ee_quat, ee_rpy = franka.get_ee_pose()
    path_x, path_y, path_z, path_roll, path_pitch, path_yaw = travelbox.get_path(ee_trans[0], ee_trans[1], ee_trans[2])
    franka.set_ee_pose_plane(path_x[0]-0.05, path_y[0], path_z[0], path_roll[0], path_pitch[0], path_yaw[0])
    franka.open_gripper()
    input('press enter to move')
    sleep(3)
    franka.set_ee_pose_plane(path_x[0], path_y[0], path_z[0], path_roll[0], path_pitch[0], path_yaw[0], asynchronous=False)
    franka.close_gripper()

    # plt.scatter(travelbox.x_sample[74], travelbox.y_sample[74], color='g', s=50)
    # plt.quiver(travelbox.x_sample, travelbox.y_sample, travelbox.tangents_x, travelbox.tangents_y, scale=20, color='blue', label="切线方向")  # 绘制切线方向
    # plt.scatter(path_x, path_y, color='g', s=5)
    # plt.show()

    idx = 0
    while idx != len(path_x) -1:
        ee_trans, ee_quat, ee_rpy = franka.get_ee_pose()
        idx = get_target_idx(idx, path_x, path_y, path_z, ee_trans[0], ee_trans[1], ee_trans[2])

        print(idx, path_x[idx], path_y[idx], path_z[idx])
        print(idx, np.degrees(path_roll[idx]), np.degrees(path_pitch[idx]), np.degrees(path_yaw[idx]))
        # input()


        franka.set_ee_pose_plane(path_x[idx], path_y[idx], path_z[idx], path_roll[idx], path_pitch[idx], path_yaw[idx])
        # franka.close_gripper()

        # plt.scatter(ee_trans[0], ee_trans[1], color='r', s=5)
        # plt.scatter(path_x[idx], path_y[idx], color='b', s=5)
        sleep(0.015)
    # sleep(0.5)
    franka.robot.join_motion()
    franka.open_gripper()
    print('done!')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)    
    plt.show()