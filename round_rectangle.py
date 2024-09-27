import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def create_rounded_rectangle_path(x_box, y_box, w_box, h_box, r_box, num_points_per_arc=20):
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
    # 创建路径的顶点和代码
    verts = []
    codes = []
    
    # 四个角的圆弧生成
    # 1. 从右上角圆弧开始 (0到90度)
    angles = np.linspace(0, np.pi / 2, num_points_per_arc)
    x_right_top_arc = (x_box + w_box / 2 - r_box) + r_box * np.cos(angles)
    y_right_top_arc = (y_box + h_box / 2 - r_box) + r_box * np.sin(angles)
    
    # 2. 上边直线
    x_top_line = np.linspace(x_box + w_box / 2 - r_box, x_box - w_box / 2 + r_box, num_points_per_arc)
    y_top_line = np.full_like(x_top_line, y_box + h_box / 2)
    
    # 3. 左上角圆弧 (90到180度)
    angles = np.linspace(np.pi / 2, np.pi, num_points_per_arc)
    x_left_top_arc = (x_box - w_box / 2 + r_box) + r_box * np.cos(angles)
    y_left_top_arc = (y_box + h_box / 2 - r_box) + r_box * np.sin(angles)
    
    # 4. 左边直线
    y_left_line = np.linspace(y_box + h_box / 2 - r_box, y_box - h_box / 2 + r_box, num_points_per_arc)
    x_left_line = np.full_like(y_left_line, x_box - w_box / 2)
    
    # 5. 左下角圆弧 (180到270度)
    angles = np.linspace(np.pi, 3 * np.pi / 2, num_points_per_arc)
    x_left_bottom_arc = (x_box - w_box / 2 + r_box) + r_box * np.cos(angles)
    y_left_bottom_arc = (y_box - h_box / 2 + r_box) + r_box * np.sin(angles)
    
    # 6. 下边直线
    x_bottom_line = np.linspace(x_box - w_box / 2 + r_box, x_box + w_box / 2 - r_box, num_points_per_arc)
    y_bottom_line = np.full_like(x_bottom_line, y_box - h_box / 2)
    
    # 7. 右下角圆弧 (270到360度)
    angles = np.linspace(3 * np.pi / 2, 2 * np.pi, num_points_per_arc)
    x_right_bottom_arc = (x_box + w_box / 2 - r_box) + r_box * np.cos(angles)
    y_right_bottom_arc = (y_box - h_box / 2 + r_box) + r_box * np.sin(angles)
    
    # 8. 右边直线
    y_right_line = np.linspace(y_box - h_box / 2 + r_box, y_box + h_box / 2 - r_box, num_points_per_arc)
    x_right_line = np.full_like(y_right_line, x_box + w_box / 2)
    
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

def get_trajectory(x_box, y_box, w_box, h_box, r_box, path_steps=100, direction='clockwise')
    # # 设置矩形和圆角参数
    # x_box, y_box = 0, 0  # 矩形的中心
    # w_box, h_box = 10, 6  # 矩形的宽度和高度
    # r_box = 2  # 圆角半径

    # 生成圆角矩形的所有点
    x_coords, y_coords = create_rounded_rectangle_path(x_box, y_box, w_box, h_box, r_box, num_points_per_arc=50)

    # 从轨迹中采样点
    num_sample_points = path_steps
    if direction == 'clockwise':
        x_sample, y_sample = sample_points_from_path(x_coords, y_coords, num_sample_points)  # clockwise
    else:
        x_sample, y_sample = sample_points_from_path(np.flip(x_coords), np.flip(y_coords), num_sample_points) # counterclockwise

    tangents_x, tangents_y = calculate_tangent_vectors(x_sample, y_sample)
    tangent_angles = np.arctan2(tangents_y, tangents_x)
    # tangent_angles_degrees = np.degrees(tangent_angles)

    # # 绘制采样后的点
    # plt.figure(figsize=(10, 10))
    # plt.plot(x_coords, y_coords, label="原始轨迹")
    # # plt.scatter(x_sample[70:], y_sample[70:], color='red', s=10, label="采样点")
    # # plt.scatter(x_sample_cc[70:], y_sample_cc[70:], color='b', s=10, label="采样点")
    # plt.quiver(x_sample, y_sample, tangents_x, tangents_y, scale=20, color='blue', label="切线方向")  # 绘制切线方向
    # # for i in range(0, len(x_sample), 5):  # 每隔 10 个点显示角度
    #     # plt.text(x_sample[i], y_sample[i], f'{tangent_angles_degrees[i]:.1f}°', fontsize=8, color='green')

    # plt.title("从圆角矩形中均匀采样点")
    # plt.xlabel("X坐标")
    # plt.ylabel("Y坐标")
    # plt.legend()
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.grid(True)
    # plt.show()

    return x_sample, y_sample, tangent_angles