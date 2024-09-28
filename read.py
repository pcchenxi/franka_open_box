from argparse import ArgumentParser
from time import sleep

from franky import Affine, Robot


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--host', default='172.16.3.1', help='FCI IP of the robot')
    args = parser.parse_args()

    robot = Robot(args.host)

    while True:
        state = robot.state
        robot_pose = robot.current_pose
        print('\nPose: ', robot_pose.end_effector_pose)
        print('O_TT_E: ', state.O_T_EE)
        print('Joints: ', state.q)
        print('Elbow: ', state.elbow)
        sleep(0.05)