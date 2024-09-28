import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

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
    for i in range(1, num_points - 1):
        dx = x_sample[i + 1] - x_sample[i - 1]
        dy = y_sample[i + 1] - y_sample[i - 1]
        norm = np.sqrt(dx**2 + dy**2)
        tangents_x[i] = dx / norm
        tangents_y[i] = dy / norm
    
    # 对第一个点和最后一个点，使用前后的差值
    tangents_x[0] = x_sample[1] - x_sample[0]
    tangents_y[0] = y_sample[1] - y_sample[0]
    norm = np.sqrt(tangents_x[0]**2 + tangents_y[0]**2)
    tangents_x[0] /= norm
    tangents_y[0] /= norm
    
    tangents_x[-1] = x_sample[-1] - x_sample[-2]
    tangents_y[-1] = y_sample[-1] - y_sample[-2]
    norm = np.sqrt(tangents_x[-1]**2 + tangents_y[-1]**2)
    tangents_x[-1] /= norm
    tangents_y[-1] /= norm
    
    return tangents_x, tangents_y

def get_trajectory():
    # # 设置矩形和圆角参数
    x_box, y_box = 0.4, 0  # 矩形的中心
    w_box, h_box = 40, 20  # 矩形的宽度和高度
    r_box = 2  # 圆角半径
    direction = 'clockwise'
    num_sample_points = 100

    # 生成圆角矩形的所有点
    x_coords, y_coords = create_rounded_rectangle_path(w_box, h_box, r_box, num_points_per_arc=50)
    x_coords, y_coords = rotate_counterclockwise(x_coords, y_coords, 90)

    # 从轨迹中采样点
    if direction == 'clockwise':
        x_sample, y_sample = sample_points_from_path(x_coords, y_coords, num_sample_points)  # clockwise
    else:
        x_sample, y_sample = sample_points_from_path(np.flip(x_coords), np.flip(y_coords), num_sample_points) # counterclockwise

    tangents_x, tangents_y = calculate_tangent_vectors(x_sample, y_sample)
    tangent_angles = np.arctan2(tangents_y, tangents_x)
    # tangent_angles_degrees = np.degrees(tangent_angles)

    # 绘制采样后的点
    plt.figure(figsize=(10, 10))
    plt.plot(x_coords, y_coords, label="原始轨迹")
    # plt.scatter(x_sample[70:], y_sample[70:], color='red', s=10, label="采样点")
    # plt.scatter(x_sample_cc[70:], y_sample_cc[70:], color='b', s=10, label="采样点")
    plt.scatter(x_sample[0], y_sample[0], color='r', s=50)
    plt.scatter(x_sample[79], y_sample[79], color='g', s=50)

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


class TravleBox:
    def __init__(self, x_box, y_box, w_box, h_box, r_box=0.1) -> None:
        self.x_box = x_box
        self.y_box = y_box
        self.w_box = w_box
        self.h_box = h_box
        self.r_box = r_box

        x_coords, y_coords = create_rounded_rectangle_path(self.w_box, self.h_box, self.r_box, num_points_per_arc=50)
        self.x_coords, self.y_coords = rotate_counterclockwise(x_coords, y_coords, 90)

        self.x_coords += self.x_box
        self.y_coords += self.y_box

    def get_path(self, x_robot=None, y_robot=None):
        if x_robot is None or y_robot is None:
            return self.x_sample, self.y_sample, self.pan_sample

        start_idx = 0
        min_dist = 99999
        for i in range(self.path_steps):
            x = self.x_sample[i]
            y = self.y_sample[i]

            diff_x = x_robot - x
            diff_y = y_robot - y
            dist = np.linalg.norm([diff_x, diff_y])
            if dist < min_dist:
                start_idx = i
                min_dist = dist
        print('start idx', start_idx, min_dist)
        print(x_robot, y_robot, self.x_sample[start_idx], self.y_sample[start_idx])

        x_path, y_path, pan_path = [], [], []
        for i in range(start_idx, start_idx+self.path_steps):
            idx = i%self.path_steps
            x_path.append(self.x_sample[idx])
            y_path.append(self.y_sample[idx])
            pan_path.append(self.pan_sample[idx])

            # print(self.x_box, self.y_box, self.w_box, self.h_box)
            if self.x_sample[idx] > (self.x_box+self.h_box/2-self.r_box):
                print('stop', len(x_path))
                break

        return np.array(x_path), np.array(y_path), np.array(pan_path)


    def generate_path(self, path_steps=100, direction='clockwise'):
        self.path_steps = path_steps

        if direction == 'counterclockwise':
            x_sample, y_sample = sample_points_from_path(self.x_coords, self.y_coords, path_steps)  # counterclockwise
            pan_shift = math.pi/2
        else:
            x_sample, y_sample = sample_points_from_path(np.flip(self.x_coords), np.flip(self.y_coords), path_steps) # clockwise
            pan_shift = -math.pi/2

        # for x, y in zip(self.x_sample, self.y_sample):
        #     print(x, y)

        tangents_x, tangents_y = calculate_tangent_vectors(x_sample, y_sample)
        pan_sample = np.arctan2(tangents_y, tangents_x)

        self.x_sample = x_sample
        self.y_sample = y_sample
        self.pan_sample = pan_sample + pan_shift
        self.tangents_x = tangents_x
        self.tangents_y = tangents_y

        # plot(x_sample, y_sample, tangents_x, tangents_y)

        # return x_sample, y_sample, pan_sample

# get_trajectory()