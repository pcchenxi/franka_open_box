import math
from time import sleep
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation
from franky import Robot, Gripper
from franky import Affine, JointWaypointMotion, JointWaypoint, CartesianMotion, ReferenceType, CartesianWaypointMotion, CartesianWaypoint

class Franka:
    def __init__(self, robot_ip, relative_dynamics_factor=0.08) -> None:
        self.robot = Robot(robot_ip)
        self.gripper = Gripper(robot_ip)
        self.robot.relative_dynamics_factor = relative_dynamics_factor
        self.relative_dynamics_factor = relative_dynamics_factor

        imp_value = 800
        self.robot.set_joint_impedance([imp_value, imp_value, imp_value, imp_value, imp_value, imp_value, imp_value])

        reset_joint = [0.07237725877102819, -0.8143505166613988, -0.15113935475521537, -2.5729412096546853, -0.05028053215321201, 1.8019384916307615, 0.8581810196212317]

        m1 = JointWaypointMotion([JointWaypoint(reset_joint)])
        self.robot.move(m1)

        # quat = Rotation.from_euler('xyz', [math.pi, 0, -math.pi/4]).as_quat()

        # m9 = CartesianWaypointMotion([
        #     CartesianWaypoint(Affine([0.4, 0.0, 0.4], quat)),
        #     CartesianWaypoint(Affine([0.4, 0.1, 0.4], quat)),
        #     CartesianWaypoint(Affine([0.4, 0.15, 0.4], quat)),
        #     CartesianWaypoint(Affine([0.4, 0.2, 0.4], quat))
        # ])
        # self.robot.move(m9, asynchronous=False)

    def norm_pan(self, pan):
        # # z_rot = -math.pi/4 + pan
        # twoPI = math.pi*2
        # z_rot = pan - twoPI* math.floor((pan + math.pi) / twoPI)
        return pan        

    def open_gripper(self):
        self.gripper.open(0.02)
        # success = self.gripper.move(0.05, 0.02)

    def close_gripper(self):
        success = self.gripper.move(0.0, 0.02)

    def set_home_pose(self):
        # self.robot.relative_dynamics_factor = 0.1
        ee_trans, ee_quat, ee_rpy = self.get_ee_pose()
        home_quat = Rotation.from_euler('xyz', [0, math.pi/2, 0]).as_quat()
        # home_quat = Rotation.from_euler('xyz', [math.pi, -math.pi/4, 0]).as_quat()
        home_xyz = [0.25, 0.3, 0.5]

        # if ee_trans[0] > 0.45:
        #     motion = CartesianMotion(Affine([ee_trans[0], -0.2, 0.31], ee_quat))
        #     self.robot.move(motion)
        #     motion = CartesianMotion(Affine([home_xyz[0], -0.2, 0.31], home_quat))
        #     self.robot.move(motion)

        motion = CartesianMotion(Affine(home_xyz, home_quat))
        self.robot.move(motion)

        self.robot.relative_dynamics_factor = self.relative_dynamics_factor

    def set_ready_pose(self):
        quat = Rotation.from_euler('xyz', [math.pi, 0, -math.pi/4]).as_quat()
        motion = CartesianMotion(Affine([0.4, 0.0, 0.3], quat))
        self.robot.move(motion)

    def set_ee_pose_plane(self, x, y, z, roll, pitch, yaw, asynchronous=True):
        z_rot = self.norm_pan(yaw)
        y_rot = self.norm_pan(pitch)
        x_rot = self.norm_pan(roll)
        quat = Rotation.from_euler('xyz', [x_rot, y_rot, z_rot]).as_quat()
        motion = CartesianMotion(Affine([x, y, z], quat))
        # print(x, y, pan)
        self.robot.move(motion, asynchronous=asynchronous)

    def get_ee_pose(self):
        robot_pose = self.robot.current_pose
        ee_trans = robot_pose.end_effector_pose.translation
        ee_quat = robot_pose.end_effector_pose.quaternion
        ee_rpy = Rotation.from_quat(ee_quat).as_euler('xyz')
        # quat = Rotation.from_euler('xyz', [math.pi, 0, 0]).as_quat()
        # rpy = Rotation.from_quat().as_euler('xyz')
        return ee_trans, ee_quat, ee_rpy