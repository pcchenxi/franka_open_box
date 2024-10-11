import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def rotation_matrix_from_rpy(roll, pitch, yaw):
    """
    Compute the rotation matrix from roll, pitch, and yaw angles.
    
    Parameters:
    roll, pitch, yaw: Roll, pitch, and yaw angles in radians
    
    Returns:
    3x3 rotation matrix
    """
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    # Total rotation matrix R = R_z * R_y * R_x
    R_total = np.dot(R_z, np.dot(R_y, R_x))
    
    return R_total

def extract_rpy_from_rotation_matrix(R):
    """
    Extract roll, pitch, and yaw from the rotation matrix.
    
    Parameters:
    R: 3x3 rotation matrix
    
    Returns:
    roll, pitch, yaw: Roll, pitch, and yaw angles in radians
    """
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    
    return roll, pitch, yaw

def rotate_box_y_axis(roll, pitch, yaw, rotation_angle_deg):
    """
    Rotate the box around the y-axis by a specified angle and compute the new roll, pitch, and yaw.
    
    Parameters:
    roll, pitch, yaw: Initial roll, pitch, and yaw angles in radians
    rotation_angle_deg: Rotation angle around the y-axis in degrees
    
    Returns:
    new_roll, new_pitch, new_yaw: New roll, pitch, and yaw angles in radians after the rotation
    """
    rotation_angle_rad = np.radians(rotation_angle_deg)
    R_y = np.array([[np.cos(rotation_angle_rad), 0, np.sin(rotation_angle_rad)],
                    [0, 1, 0],
                    [-np.sin(rotation_angle_rad), 0, np.cos(rotation_angle_rad)]])
    
    # Arrays to store the new roll, pitch, and yaw values
    new_roll_array = np.zeros_like(roll)
    new_pitch_array = np.zeros_like(pitch)
    new_yaw_array = np.zeros_like(yaw)
    
    # Iterate through each set of roll, pitch, and yaw angles
    for i in range(len(roll)):
        # Convert the roll, pitch, and yaw for the current element to a rotation matrix
        R_initial = rotation_matrix_from_rpy(roll[i], pitch[i], yaw[i])
        
        # Apply the y-axis rotation to the initial rotation matrix
        R_final = np.dot(R_y, R_initial)
        
        # Extract the new roll, pitch, and yaw from the final rotation matrix
        new_roll, new_pitch, new_yaw = extract_rpy_from_rotation_matrix(R_final)
        
        # Store the new values in the arrays
        new_roll_array[i] = new_roll
        new_pitch_array[i] = new_pitch
        new_yaw_array[i] = new_yaw
    
    return new_roll_array, new_pitch_array, new_yaw_array

def rotation_matrix(roll, pitch, yaw):
    """
    计算总的旋转矩阵 (roll, pitch, yaw).
    
    参数:
    roll, pitch, yaw: 滚转, 俯仰, 偏航 角度（弧度）
    
    返回:
    3x3 旋转矩阵
    """
    # Rotation matrices around x (roll), y (pitch), and z (yaw)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    # Total rotation matrix R = R_z * R_y * R_x
    R_total = np.dot(R_z, np.dot(R_y, R_x))
    
    return R_total


def plot_gripper_frame(ax, origin, R, scale=1.0):
    """
    绘制夹爪的局部坐标系 (x, y, z 轴) 在给定点的位置.
    
    参数:
    ax: Matplotlib 3D 轴
    origin: 夹爪的起始位置 [x, y, z]
    R: 旋转矩阵
    scale: 轴的缩放因子
    """
    # Local frame axes (unit vectors)
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])
    
    # Rotate the local axes using the rotation matrix
    x_rot = np.dot(R, x_axis) * scale
    y_rot = np.dot(R, y_axis) * scale
    z_rot = np.dot(R, z_axis) * scale
    
    # Plot the gripper frame's axes
    ax.quiver(*origin, *x_rot, color='r', label='X-axis (gripper)', length=scale)
    ax.quiver(*origin, *y_rot, color='g', label='Y-axis (gripper)', length=scale)
    ax.quiver(*origin, *z_rot, color='b', label='Z-axis (gripper)', length=scale)




def rotate_y_axis(x_sample, y_sample, z_sample, angle, units="DEGREES"):
    """
    旋转给定的3D坐标 (x_sample, y_sample, z_sample) 沿y轴旋转w度，返回新的坐标.
    
    参数:
    x_sample, y_sample, z_sample: 初始的3D点坐标
    angle: 绕y轴旋转的角度（度数）
    
    返回:
    x_rot, y_rot, z_rot: 旋转后的3D坐标
    """
    # 将角度从度转换为弧度
    if units == "DEGREES":
        w_radians = np.radians(angle)
    
    # 计算旋转矩阵的 cos 和 sin
    cos_w = np.cos(w_radians)
    sin_w = np.sin(w_radians)
    
    # 初始化旋转后的坐标列表
    x_rot = np.zeros_like(x_sample)
    y_rot = np.zeros_like(y_sample)
    z_rot = np.zeros_like(z_sample)
    
    # 对每个点应用旋转矩阵
    for i in range(len(x_sample)):
        x = x_sample[i]
        z = z_sample[i]
        
        # 旋转后的x' 和 z'
        x_rot[i] = cos_w * x + sin_w * z
        y_rot[i] = y_sample[i]  # y 坐标保持不变
        z_rot[i] = -sin_w * x + cos_w * z
    
    return x_rot, y_rot, z_rot

def rotate_counterclockwise(x, y, angle, x_shift=0, y_shift=0, units="DEGREES"):
    """
    Rotates a point in the xy-plane counterclockwise through an angle about the origin
    https://en.wikipedia.org/wiki/Rotation_matrix
    :param x: x coordinate
    :param y: y coordinate
    :param x_shift: x-axis shift from origin (0, 0)
    :param y_shift: y-axis shift from origin (0, 0)
    :param angle: The rotation angle in degrees
    :param units: DEGREES (default) or RADIANS
    :return: Tuple of rotated x and y
    """
    # Shift to origin (0,0)
    x = x - x_shift
    y = y - y_shift

    # Convert degrees to radians
    if units == "DEGREES":
        angle = math.radians(angle)

    # Rotation matrix multiplication to get rotated x & y
    xr = (x * math.cos(angle)) - (y * math.sin(angle)) + x_shift
    yr = (x * math.sin(angle)) + (y * math.cos(angle)) + y_shift

    return xr, yr

def create_rounded_rectangle_path(w_box, h_box, r_box, num_points_per_arc=20):
    """
    使用 matplotlib 生成圆角矩形的轨迹，并返回该轨迹的所有点。
    
    参数:
    x_box, y_box: 矩形的中心坐标
    w_box, h_box: 矩形的宽度和高度
    r_box: 圆角的半径
    num_points_per_arc: 每个圆角弧的插值点数量
    
    返回:
    x_coords, y_coords: 生成的轨迹上的所有点
    """
    # 四个角的圆弧生成
    # 1. 从右上角圆弧开始 (0到90度)
    angles = np.linspace(0, np.pi / 2, num_points_per_arc)
    x_right_top_arc = (w_box / 2 - r_box) + r_box * np.cos(angles)
    y_right_top_arc = (h_box / 2 - r_box) + r_box * np.sin(angles)
    
    # 2. 上边直线
    x_top_line = np.linspace(w_box / 2 - r_box, -w_box / 2 + r_box, num_points_per_arc)
    y_top_line = np.full_like(x_top_line, h_box / 2)
    
    # 3. 左上角圆弧 (90到180度)
    angles = np.linspace(np.pi / 2, np.pi, num_points_per_arc)
    x_left_top_arc = (-w_box / 2 + r_box) + r_box * np.cos(angles)
    y_left_top_arc = (h_box / 2 - r_box) + r_box * np.sin(angles)
    
    # 4. 左边直线
    y_left_line = np.linspace(h_box / 2 - r_box, -h_box / 2 + r_box, num_points_per_arc)
    x_left_line = np.full_like(y_left_line, - w_box / 2)
    
    # 5. 左下角圆弧 (180到270度)
    angles = np.linspace(np.pi, 3 * np.pi / 2, num_points_per_arc)
    x_left_bottom_arc = (-w_box / 2 + r_box) + r_box * np.cos(angles)
    y_left_bottom_arc = (-h_box / 2 + r_box) + r_box * np.sin(angles)
    
    # 6. 下边直线
    x_bottom_line = np.linspace(-w_box / 2 + r_box, w_box / 2 - r_box, num_points_per_arc)
    y_bottom_line = np.full_like(x_bottom_line, -h_box / 2)
    
    # 7. 右下角圆弧 (270到360度)
    angles = np.linspace(3 * np.pi / 2, 2 * np.pi, num_points_per_arc)
    x_right_bottom_arc = (w_box / 2 - r_box) + r_box * np.cos(angles)
    y_right_bottom_arc = (-h_box / 2 + r_box) + r_box * np.sin(angles)
    
    # 8. 右边直线
    y_right_line = np.linspace(-h_box / 2 + r_box, h_box / 2 - r_box, num_points_per_arc)
    x_right_line = np.full_like(y_right_line, w_box / 2)
    
    # 合并所有点
    x_coords = np.concatenate([
        x_right_top_arc, x_top_line, x_left_top_arc, x_left_line,
        x_left_bottom_arc, x_bottom_line, x_right_bottom_arc, x_right_line
    ])
    y_coords = np.concatenate([
        y_right_top_arc, y_top_line, y_left_top_arc, y_left_line,
        y_left_bottom_arc, y_bottom_line, y_right_bottom_arc, y_right_line
    ])
    
    return x_coords, y_coords

def sample_points_from_path(x_coords, y_coords, num_sample_points):
    """
    从给定的轨迹点中等距离采样指定数量的点，逆时针顺序。
    
    参数:
    x_coords, y_coords: 轨迹的 x 和 y 坐标点
    num_sample_points: 要采样的点的数量
    
    返回:
    x_sample, y_sample: 采样后的点
    """
    # 计算每个点之间的距离，累加得到总距离
    distances = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
    cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
    
    # 根据总距离等间距采样
    total_distance = cumulative_distances[-1]
    sample_distances = np.linspace(0, total_distance, num_sample_points)
    
    # 通过插值获取等距点
    x_sample = np.interp(sample_distances, cumulative_distances, x_coords)
    y_sample = np.interp(sample_distances, cumulative_distances, y_coords)
    
    return x_sample, y_sample

def calculate_tangent_vectors(x_sample, y_sample):
    """
    计算每个采样点的切线方向。
    
    参数:
    x_sample, y_sample: 采样点的 x 和 y 坐标
    
    返回:
    tangents_x, tangents_y: 每个采样点的切线方向向量
    """
    num_points = len(x_sample)
    
    # 初始化切线向量
    tangents_x = np.zeros(num_points)
    tangents_y = np.zeros(num_points)
    
    # 对中间点，使用相邻点的差值
    num = len(x_sample)
    for i in range(0, num_points):
        s_i = (i+1)%num
        e_i = i-1
        dx = x_sample[s_i] - x_sample[e_i]
        dy = y_sample[s_i] - y_sample[e_i]
        norm = np.sqrt(dx**2 + dy**2)
        tangents_x[i] = dx / norm
        tangents_y[i] = dy / norm
    
    return tangents_x, tangents_y


def calculate_3d_tangent_vectors(x_sample, y_sample, z_sample):
    """
    计算每个采样点的3D切线向量 (dx, dy, dz).
    
    参数:
    x_sample, y_sample, z_sample: 3D点的 x, y, z 坐标
    
    返回:
    tangents_x, tangents_y, tangents_z: 3D切线向量的各个分量
    """
    normals_x = np.zeros_like(x_sample)
    normals_y = np.zeros_like(y_sample)
    normals_z = np.zeros_like(z_sample)
    
    tangents_x = np.zeros_like(x_sample)
    tangents_y = np.zeros_like(y_sample)
    tangents_z = np.zeros_like(z_sample)
    
    num = len(x_sample)
    for i in range(0, num):
        # 法线：相邻点的差值
        s_i = (i+1)%num
        dx = x_sample[s_i] - x_sample[i - 1]
        dy = y_sample[s_i] - y_sample[i - 1]
        dz = z_sample[s_i] - z_sample[i - 1]
        norm = np.sqrt(dx**2 + dy**2 + dz**2)
        normals_x[i] = dx / norm
        normals_y[i] = dy / norm
        normals_z[i] = dz / norm
        
        tangents_x[i] = -dy
        tangents_y[i] = dx
        tangents_z[i] = 0
    
    return normals_x, normals_y, normals_z, tangents_x, tangents_y, tangents_z


def compute_rpy_with_roll(normals_x, normals_y, normals_z, tangents_x, tangents_y, tangents_z):
    """
    根据法线和切线计算每个点的全局滚转 (roll), 俯仰 (pitch), 偏航 (yaw) 角度.
    
    参数:
    normals_x, normals_y, normals_z: 法线分量
    tangents_x, tangents_y, tangents_z: 切线分量
    
    返回:
    roll_angles, pitch_angles, yaw_angles: 每个点的roll, pitch, yaw 角度
    """
    roll_angles = np.zeros_like(normals_x)
    pitch_angles = np.zeros_like(normals_x)
    yaw_angles = np.zeros_like(normals_x)
    
    for i in range(len(normals_x)):
        # Normal vector at each point
        nx = normals_x[i]
        ny = normals_y[i]
        nz = normals_z[i]
        
        # 1. Compute pitch (rotation about global y-axis)
        pitch = np.arctan2(-nz, np.sqrt(nx**2 + ny**2))
        
        # 2. Compute yaw (rotation about global z-axis)
        yaw = np.arctan2(ny, nx)
        
        # 3. Compute roll: align the gripper's fingers with the box surface using the tangents
        tx = tangents_x[i]
        ty = tangents_y[i]
        roll = np.arctan2(ty, tx)
        
        roll_angles[i] = roll
        pitch_angles[i] = pitch
        yaw_angles[i] = yaw
    
    return roll_angles, pitch_angles, yaw_angles

def get_trajectory():
    # # 设置矩形和圆角参数
    x_box, y_box = 0.4, 0  # 矩形的中心
    w_box, h_box = 40, 20  # 矩形的宽度和高度
    height = 20
    r_box = 2  # 圆角半径
    direction = 'clockwise'
    num_sample_points = 100
    tangent_angles = 0

    # 生成圆角矩形的所有点
    x_coords, y_coords = create_rounded_rectangle_path(w_box, h_box, r_box, num_points_per_arc=50)

    # 从轨迹中采样点
    if direction == 'clockwise':
        x_sample, y_sample = sample_points_from_path(x_coords, y_coords, num_sample_points)  # clockwise
    else:
        x_sample, y_sample = sample_points_from_path(np.flip(x_coords), np.flip(y_coords), num_sample_points) # counterclockwise

    x_sample, y_sample = rotate_counterclockwise(x_sample, y_sample, 90)
    z_sample = np.zeros(len(x_sample))*height
    tangents_x, tangents_y = calculate_tangent_vectors(x_sample, y_sample)
    initial_yaw = np.arctan2(tangents_y, tangents_x)
    initial_roll = np.zeros(len(initial_yaw))
    initial_pitch = np.zeros(len(initial_yaw))

    angle = 45
    roll_angles, pitch_angles, yaw_angles = rotate_box_y_axis(initial_roll, initial_pitch, initial_yaw, angle)
    x_sample, y_sample, z_sample = rotate_y_axis(x_sample, y_sample, z_sample, angle)


    # tangents_x, tangents_y, tangents_z = calculate_3d_tangent_vectors(x_rot, y_rot, z_rot)
    # normals_x, normals_y, normals_z, tangents_x, tangents_y, tangents_z = calculate_3d_tangent_vectors(x_sample, y_sample, z_sample)
    # roll_angles, pitch_angles, yaw_angles = compute_rpy_with_roll(normals_x, normals_y, normals_z, tangents_x, tangents_y, tangents_z)

    # 设置 3D 图
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the original surface points
    ax.scatter(x_sample, y_sample, z_sample, color='black', label='Sample Points')

    # 在每个样本点绘制夹爪的坐标系
    for i in range(len(x_sample)):
        origin = np.array([x_sample[i], y_sample[i], z_sample[i]])
        
        # 计算旋转矩阵
        R = rotation_matrix(roll_angles[i], pitch_angles[i], yaw_angles[i])
        
        # 绘制每个点的夹爪坐标系
        plot_gripper_frame(ax, origin, R, scale=1)

    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Gripper Frames with Roll, Pitch, Yaw')
    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()



    # for y,p,r in zip(roll, pitch, yaw):
    #     print(y,p,r)

    # orientations = compute_gripper_orientation(x_rot, y_rot, z_rot)
    # draw_gripper_orientation(x_rot, y_rot, z_rot, orientations)
    # print(orientations)

    # plot_3d(x_sample, y_sample, z_sample, x_rot, y_rot, z_rot)
    # tangents_x, tangents_y = calculate_tangent_vectors(x_sample, y_sample)
    # tangent_angles = np.arctan2(tangents_y, tangents_x)
    # tangent_angles_degrees = np.degrees(tangent_angles)

    # # 绘制采样后的点
    # plt.figure(figsize=(10, 10))
    # # plt.plot(x_coords, y_coords, label="原始轨迹")
    # # # plt.scatter(x_sample[70:], y_sample[70:], color='red', s=10, label="采样点")
    # # # plt.scatter(x_sample_cc[70:], y_sample_cc[70:], color='b', s=10, label="采样点")
    # # plt.scatter(x_sample[0], y_sample[0], color='r', s=50)
    # # plt.scatter(x_sample[79], y_sample[79], color='g', s=50)

    # # plt.quiver(x_sample, y_sample, tangents_x, tangents_y, scale=20, color='blue', label="切线方向")  # 绘制切线方向
    # # # for i in range(0, len(x_sample), 5):  # 每隔 10 个点显示角度
    # #     # plt.text(x_sample[i], y_sample[i], f'{tangent_angles_degrees[i]:.1f}°', fontsize=8, color='green')

    # plt.scatter(x_sample, y_sample, z_sample, color='blue', label='Original Points')
    # plt.plot(x_sample, y_sample, z_sample, color='blue', linestyle='--')

    # # 绘制旋转后的点
    # plt.scatter(x_rot, y_rot, z_rot, color='red', label='Rotated Points')
    # plt.plot(x_rot, y_rot, z_rot, color='red', linestyle='--')

    # plt.title("从圆角矩形中均匀采样点")
    # plt.xlabel("X坐标")
    # plt.ylabel("Y坐标")
    # plt.legend()
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.grid(True)
    # plt.show()

    return x_sample, y_sample, tangent_angles

def plot(x_sample, y_sample, tangents_x, tangents_y):
    plt.figure(figsize=(10, 10))
    # plt.plot(x_coords, y_coords, label="原始轨迹")
    # plt.scatter(x_sample[70:], y_sample[70:], color='red', s=10, label="采样点")
    # plt.scatter(x_sample_cc[70:], y_sample_cc[70:], color='b', s=10, label="采样点")
    # plt.scatter(x_sample[0], y_sample[0], color='r', s=50)
    plt.scatter(x_sample[74], y_sample[74], color='g', s=50)

    plt.quiver(x_sample, y_sample, tangents_x, tangents_y, scale=20, color='blue', label="切线方向")  # 绘制切线方向
    # for i in range(0, len(x_sample), 5):  # 每隔 10 个点显示角度
        # plt.text(x_sample[i], y_sample[i], f'{tangent_angles_degrees[i]:.1f}°', fontsize=8, color='green')

    plt.title("从圆角矩形中均匀采样点")
    plt.xlabel("X坐标")
    plt.ylabel("Y坐标")
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()

def plot_3d(x_sample, y_sample, z_sample, x_rot, y_rot, z_rot):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制原始点
    ax.scatter(x_sample, y_sample, z_sample, color='blue', label='Original Points')
    ax.plot(x_sample, y_sample, z_sample, color='blue', linestyle='--')

    # 绘制旋转后的点
    ax.scatter(x_rot, y_rot, z_rot, color='red', label='Rotated Points')
    ax.plot(x_rot, y_rot, z_rot, color='red', linestyle='--')

    # 添加图例和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Rotation Around Y-axis by {45} Degrees')
    ax.legend()

    plt.show()    


def draw_gripper_orientation(x_rot, y_rot, z_rot, orientations):
    """
    绘制机械手末端执行器的姿态（用欧拉角表示的方向）在每个采样点上的向量。
    
    参数:
    x_rot, y_rot, z_rot: 旋转后的箱子轮廓的3D坐标
    orientations: 每个点的欧拉角姿态
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制原始点
    ax.scatter(x_rot, y_rot, z_rot, color='blue', label='Box Contour')
    ax.plot(x_rot, y_rot, z_rot, color='blue', linestyle='--')
    
    # 绘制每个点的姿态方向
    for i in range(1, len(orientations)):
        yaw, pitch, roll = orientations[i - 1]
        
        # 将欧拉角转换为向量 (简单起见，用单位向量表示方向)
        dx = np.cos(yaw) * np.cos(pitch)
        dy = np.sin(pitch)
        dz = np.sin(yaw) * np.cos(pitch)
        
        # 绘制姿态向量
        ax.quiver(x_rot[i], y_rot[i], z_rot[i], dx, dy, dz, length=0.5, color='red')
    
    # 设置标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Gripper Orientation Vectors on Rotated Box Contour')
    ax.legend()
    
    plt.show()


class TravleBox:
    def __init__(self, x_box, y_box, z_box, w_box, h_box, r_box=0.1) -> None:
        self.x_box = x_box
        self.y_box = y_box
        self.z_box = z_box
        self.w_box = w_box
        self.h_box = h_box
        self.r_box = r_box
        self.y_rotation = 45

        x_coords, y_coords = create_rounded_rectangle_path(self.w_box, self.h_box, self.r_box, num_points_per_arc=50)
        self.x_coords, self.y_coords = rotate_counterclockwise(x_coords, y_coords, 90)

        # self.x_coords += self.x_box
        # self.y_coords += self.y_box

    def get_path(self, x_robot=None, y_robot=None, z_robot=None):
        if x_robot is None or y_robot is None:
            return self.x_sample, self.y_sample, self.z_sample, self.roll_sample, self.pitch_sample, self.yaw_sample

        start_idx = 0
        min_dist = 99999
        for i in range(self.path_steps):
            x = self.x_sample[i]
            y = self.y_sample[i]
            z = self.z_sample[i]

            diff_x = x_robot - x
            diff_y = y_robot - y
            diff_z = z_robot - z
            dist = np.linalg.norm([diff_x, diff_y, diff_z])
            if dist < min_dist:
                start_idx = i
                min_dist = dist
        print('start idx', start_idx, min_dist)
        print(x_robot, y_robot, self.x_sample[start_idx], self.y_sample[start_idx])

        x_path, y_path, z_path = [], [], []
        roll_path, pitch_path, yaw_path = [], [], []
        max_x = np.max(self.x_sample)

        for i in range(start_idx, start_idx+self.path_steps):
            idx = i%self.path_steps
            x_path.append(self.x_sample[idx])
            y_path.append(self.y_sample[idx])
            z_path.append(self.z_sample[idx])

            roll_path.append(self.roll_sample[idx])
            pitch_path.append(self.pitch_sample[idx])
            yaw_path.append(self.yaw_sample[idx])

            # print(self.x_box, self.y_box, self.w_box, self.h_box)
            if self.x_sample[idx] > max_x-0.1:
                print('stop', len(x_path))
                break
        return np.array(x_path), np.array(y_path), np.array(z_path), np.array(roll_path), np.array(pitch_path), np.array(yaw_path)

    def generate_path(self, path_steps=50, direction='clockwise'):
        self.path_steps = path_steps

        if direction == 'counterclockwise':
            x_sample, y_sample = sample_points_from_path(self.x_coords, self.y_coords, path_steps)  # counterclockwise
            yaw_shift = math.pi/2
        else:
            x_sample, y_sample = sample_points_from_path(np.flip(self.x_coords), np.flip(self.y_coords), path_steps) # clockwise
            yaw_shift = -math.pi/2

        tangents_x, tangents_y = calculate_tangent_vectors(x_sample, y_sample)
        initial_yaw = np.arctan2(tangents_y, tangents_x)
        initial_roll = np.zeros(len(initial_yaw))
        initial_pitch = np.zeros(len(initial_yaw))

        angle = 45
        roll, pitch, yaw = rotate_box_y_axis(initial_roll, initial_pitch, initial_yaw, angle)
        x_sample, y_sample, z_sample = rotate_y_axis(x_sample, y_sample, np.zeros(len(x_sample)), angle)

        # tangents_x, tangents_y = calculate_tangent_vectors(x_sample, y_sample)
        # pan_sample = np.arctan2(tangents_y, tangents_x)

        self.x_sample = x_sample + self.x_box 
        self.y_sample = y_sample + self.y_box 
        self.z_sample = z_sample + self.z_box
        self.roll_sample = roll
        self.pitch_sample = pitch
        self.yaw_sample = yaw + yaw_shift

        # self.pan_sample = pan_sample + pan_shift
        # self.tangents_x = tangents_x
        # self.tangents_y = tangents_y

        # plot(x_sample, y_sample, tangents_x, tangents_y)

get_trajectory()
input()