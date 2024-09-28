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

def get_target_idx(start_idx, path_x, path_y, path_pan, robot_x, robot_y, robot_pan):
    threshold_xy, threshold_pan = 0.005, math.pi/180

    idx = len(path_x) - 1
    for i in range(start_idx, len(path_x)):
        x, y, pan = path_x[i], path_y[i], path_pan[i]
        dist_xy = np.linalg.norm([x-robot_x, y-robot_y])
        dist_pan = norm_rotation(pan) - robot_pan
        if dist_xy < threshold_xy and dist_pan < threshold_pan:
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
    input('press enter to move')

    travelbox = TravleBox(0.5, 0.0, 0.4, 0.2, 0.05)
    travelbox.generate_path(direction='clockwise') #counterclockwise  clockwise

    ee_trans, ee_quat, ee_rpy = franka.get_ee_pose()
    path_x, path_y, path_pan = travelbox.get_path(ee_trans[0], ee_trans[1])

    # plt.scatter(travelbox.x_sample[74], travelbox.y_sample[74], color='g', s=50)
    # plt.quiver(travelbox.x_sample, travelbox.y_sample, travelbox.tangents_x, travelbox.tangents_y, scale=20, color='blue', label="切线方向")  # 绘制切线方向
    plt.scatter(path_x, path_y, color='g', s=5)
    # plt.show()

    idx = 0
    while idx != len(path_x) -1:
        ee_trans, ee_quat, ee_rpy = franka.get_ee_pose()
        idx = get_target_idx(idx, path_x, path_y, path_pan, ee_trans[0], ee_trans[1], ee_rpy[2])
        franka.set_ee_pose_plane(path_x[idx], path_y[idx], path_pan[idx])
        print(idx, path_x[idx], path_y[idx], path_pan[idx])

        plt.scatter(ee_trans[0], ee_trans[1], color='r', s=5)
        plt.scatter(path_x[idx], path_y[idx], color='b', s=5)

        # print()
        sleep(0.1)
    print('done!')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)    
    plt.show()

    # for x,y,pan in zip(path_x, path_y, path_pan):
    #     print(x, y, pan)
    #     franka.set_ee_pose_plane(x,y,pan)
    #     ee_trans, ee_quat, ee_rpy = franka.get_ee_pose()
    #     print(ee_trans[0]-x, ee_trans[1]-y)

    # while True:
    #     state = robot.state
    #     robot_pose = robot.current_pose
    #     print(get_eepose())
    #     # print('\nPose: ', robot_pose.end_effector_pose)
    #     # print('O_TT_E: ', state.O_T_EE)
    #     # print('Joints: ', state.q)
    #     # print('Elbow: ', state.elbow)
    #     sleep(0.05)