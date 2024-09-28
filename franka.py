import math
from time import sleep
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation
from franky import Affine, JointWaypointMotion, JointWaypoint, CartesianMotion, Robot, ReferenceType, CartesianWaypointMotion, CartesianWaypoint, CartesianState

class Franka:
    def __init__(self, robot_ip, relative_dynamics_factor=0.1) -> None:
        self.robot = Robot(robot_ip)
        self.robot.relative_dynamics_factor = relative_dynamics_factor

        quat = Rotation.from_euler('xyz', [math.pi, 0, -math.pi/4]).as_quat()

        # m9 = CartesianWaypointMotion([
        #     CartesianWaypoint(Affine([0.4, 0.0, 0.4], quat)),
        #     CartesianWaypoint(Affine([0.4, 0.1, 0.4], quat)),
        #     CartesianWaypoint(Affine([0.4, 0.15, 0.4], quat)),
        #     CartesianWaypoint(Affine([0.4, 0.2, 0.4], quat))
        # ])
        # self.robot.move(m9, asynchronous=False)

    def norm_pan(self, pan):
        z_rot = -math.pi/4 + pan
        twoPI = math.pi*2
        z_rot = z_rot - twoPI* math.floor((z_rot + math.pi) / twoPI)        
        # z_rot = (-math.pi/4 + pan-math.pi/2)%(math.pi*2)
        # if z_rot > math.pi:
        #     z_rot -= math.pi*2
        # if z_rot < -math.pi:
        #     z_rot += math.pi*2
        return z_rot        

    def set_home_pose(self):
        quat = Rotation.from_euler('xyz', [math.pi, 0, -math.pi/4]).as_quat()
        motion = CartesianMotion(Affine([0.4, 0.0, 0.4], quat))
        self.robot.move(motion)

    def set_ready_pose(self):
        quat = Rotation.from_euler('xyz', [math.pi, 0, -math.pi/4]).as_quat()
        motion = CartesianMotion(Affine([0.4, 0.0, 0.3], quat))
        self.robot.move(motion)

    def set_ee_pose_plane(self, x, y, pan):
        z_rot = self.norm_pan(pan)
        quat = Rotation.from_euler('xyz', [math.pi, 0, z_rot]).as_quat()
        motion = CartesianMotion(Affine([x, y, 0.4], quat))
        # print(x, y, pan)
        self.robot.move(motion, asynchronous=True)

    def get_ee_pose(self):
        robot_pose = self.robot.current_pose
        ee_trans = robot_pose.end_effector_pose.translation
        ee_quat = robot_pose.end_effector_pose.quaternion
        ee_rpy = Rotation.from_quat(ee_quat).as_euler('xyz')
        # quat = Rotation.from_euler('xyz', [math.pi, 0, 0]).as_quat()
        # rpy = Rotation.from_quat().as_euler('xyz')
        return ee_trans, ee_quat, ee_rpy