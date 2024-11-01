import numpy as np
import pickle, time
from argparse import ArgumentParser
from franka import Franka
from scipy.spatial.transform import Rotation as R

def invert_pose(translation, quaternion):
    # Invert the quaternion (rotation)
    rotation = R.from_quat(quaternion)
    inv_rotation = rotation.inv()
    inv_quaternion = inv_rotation.as_quat()
    
    # Invert the translation
    inv_translation = -inv_rotation.apply(translation)
    
    return inv_translation, inv_quaternion

def multiply_poses(translation1, quaternion1, translation2, quaternion2):
    # Convert quaternions to rotations
    rotation1 = R.from_quat(quaternion1)
    rotation2 = R.from_quat(quaternion2)
    
    # Compute the combined rotation
    combined_rotation = rotation1 * rotation2
    result_quaternion = combined_rotation.as_quat()
    
    # Compute the combined translation
    result_translation = rotation1.apply(translation2) + translation1
    
    return result_translation, result_quaternion

def compute_relative_motion(poses):
    relative_poses = []
    
    for i in range(len(poses) - 1):
        # Extract current and next pose
        translation1, quaternion1 = poses[i]
        translation2, quaternion2 = poses[i + 1]
        
        # Invert the current pose
        inv_translation, inv_quaternion = invert_pose(translation1, quaternion1)
        
        # Compute the relative pose from the current to the next pose
        rel_translation, rel_quaternion = multiply_poses(inv_translation, inv_quaternion, translation2, quaternion2)
        
        # Store the relative pose
        relative_poses.append((rel_translation, rel_quaternion))
    
    return relative_poses

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--host', default='172.16.3.1', help='FCI IP of the robot')
    args = parser.parse_args()

    franka = Franka(args.host)
    # franka.set_default_pose()
    robot_trans, robot_quat, robot_rpy = franka.get_ee_pose()
    franka.move_ee([0.4, 0.0, 0.4], robot_quat)

    file_path = './pose_only_ccw/0.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    robot_trans, robot_quat, robot_rpy = franka.get_ee_pose()
    print('current robot translation', robot_trans)
    trans_list, quat_list = data['translation'], data['rotation']

    first_translation = trans_list[0]
    gripper_quat = R.from_quat(quat_list[0]).as_euler('xyz')
    first_quaternion = R.from_euler('xyz', [0, 0, gripper_quat[2]]).as_quat()
    # first_quaternion = R.from_euler('xyz', [0, 0, np.pi/2]).as_quat()

    inv_first_translation, inv_first_quaternion = invert_pose(first_translation, first_quaternion)
    
    ext_trans_list, ext_rot_list = [], []
    for i in range(len(trans_list)-10):
        tra_trans = trans_list[i]
        tra_quat = quat_list[i]

        relt_trans, relt_quat = multiply_poses(
            inv_first_translation, inv_first_quaternion, tra_trans, tra_quat
        )

        tra_rpy = R.from_quat(tra_quat).as_euler('xyz')
        relt_rpy = R.from_quat(relt_quat).as_euler('xyz')

        ee_trans = robot_trans + relt_trans
        ext_trans_list.append(ee_trans)
        ext_rot_list.append(relt_quat)

        tra_trans_next = trans_list[i+10]
        z_diff = tra_trans_next[2]-tra_trans[2]   
        if z_diff > 0.1:
            break

    franka.move_ee(ext_trans_list[0], ext_rot_list[0], asynchronous=False)
    input()
    for trans, rot in zip(ext_trans_list, ext_rot_list):
        franka.move_ee(trans, rot, asynchronous=True)
        time.sleep(0.1)

    franka.robot.join_motion()

    # for key, value in data.items():
    #     print(key)
    #     print(value)
