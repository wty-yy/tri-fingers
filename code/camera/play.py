import cv2
import numpy as np
 
 
def display_vertices_uv(vertices_2d, win_name='vertices', wait_key=0, canvas_size=(320, 320)):
    """Show the vertices on uv-coordinates"""
    img = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)
    edges = np.array([
        [0, 1], [0, 2], [0, 4],
        [1, 3], [1, 5],
        [2, 3], [2, 6],
        [4, 5], [4, 6],
        [7, 5], [7, 6], [7, 3]])
 
    for edge in edges:
        pt1 = tuple(vertices_2d[edge[0]])
        pt2 = tuple(vertices_2d[edge[1]])
        cv2.line(img, pt1, pt2, (0, 255, 255), 2)
 
    cv2.imshow(win_name, img)
    cv2.waitKey(wait_key)
 
 
def perspective_projection(vertices, K):
    """use perspective projection"""
    vertices_2d = np.matmul(K, vertices.T).T
    vertices_2d[:, 0] /= vertices_2d[:, 2]
    vertices_2d[:, 1] /= vertices_2d[:, 2]
    vertices_2d = vertices_2d[:, :2].astype(np.int32)
 
    return vertices_2d
 
 
def construct_extrinsic_matrix_R(yaw_angle, roll_angle, pitch_angle):
    """Construct the camera external parameter rotation matrix R"""
    yaw = np.deg2rad(yaw_angle)
    roll = np.deg2rad(roll_angle)
    pitch = np.deg2rad(pitch_angle)
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    R_pitch = np.array([
        [0, np.cos(pitch), -np.sin(pitch)],
        [1, 0, 0],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    R = np.matmul(R_pitch, np.matmul(R_yaw, R_roll))
 
    return R
 
 
def sample_a():
    # 定义方形画布像素坐标长度
    canvas_square_size = 320
    # 定义立方体的边长
    length = 1
 
    # 定义立方体的8个顶点坐标 使用世界坐标作为表达
    vertices_w = np.array([
        [-length / 2, -length / 2, -length / 2],
        [-length / 2, -length / 2, length / 2],
        [-length / 2, length / 2, -length / 2],
        [-length / 2, length / 2, length / 2],
        [length / 2, -length / 2, -length / 2],
        [length / 2, -length / 2, length / 2],
        [length / 2, length / 2, -length / 2],
        [length / 2, length / 2, length / 2]])
    print("世界坐标系顶点集合: ", vertices_w.shape)
 
    # 定义一个角度
    a = 45
    # 转换为弧度制
    a = np.deg2rad(a)
 
    # 手动定一个相机外参R旋转矩阵，并设置让其绕roll轴旋转a度
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(a), -np.sin(a)],
        [0, np.sin(a), np.cos(a)]
    ])
    # 手动定一个相机外参的偏移向量t 即在x, y, z的位置看面朝单位
    t1 = 0
    t2 = 0
    t3 = 5  # 数值越大则距离观测目标距离则越长
    T = np.array([t1, t2, t3])
 
    # 求基于相机坐标系的顶点集
    vertices_c = np.matmul(R_roll, vertices_w.T).T + T
 
    # 手动定一组相机内参K
    fx = 800
    fy = 800
    cx = canvas_square_size // 2
    cy = canvas_square_size // 2
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
 
    # 使用透视投影解出像素坐标的顶点集
    vertices_uv = perspective_projection(vertices_c, K)
    print("像素坐标系顶点集合: ", vertices_uv.shape)
 
    # 显示求解后的uv顶点集
    display_vertices_uv(vertices_uv, canvas_size=(canvas_square_size, canvas_square_size))
 
    # 再定义一组旋转矩阵，以pitch轴进行b度旋转
    b = 25
    b = np.deg2rad(b)
    R_pitch = np.array([
        [0, np.cos(b), -np.sin(b)],
        [1, 0, 0],
        [0, np.sin(b), np.cos(b)]
    ])
    # 重新调整一下外参的旋转矩阵R
    R = np.matmul(R_roll, R_pitch)
    # 重新求基于相机坐标系的顶点集 加入yaw旋转角
    vertices_c_pitch = np.matmul(R, vertices_w.T).T + T
    # 继续使用内参K透视投影解出像素坐标的顶点集
    vertices_uv_pitch = perspective_projection(vertices_c_pitch, K)
    # 显示求解后的uv顶点集
    display_vertices_uv(vertices_uv_pitch)
 
    # 使用solvePnP尝试解出相机外参
    rvec = np.zeros((3, 1))
    tvec = np.zeros((3, 1))
 
    retval, rvec, tvec = cv2.solvePnP(vertices_w.astype(np.float32), vertices_uv_pitch.astype(np.float32),
                                      K.astype(np.float32),
                                      None, rvec, tvec, False, cv2.SOLVEPNP_ITERATIVE)
    R_solved, _ = cv2.Rodrigues(rvec)
    print("解PnP得出的R Matrix: ", -R_solved)   # 解出的坐标系是反着需要自行调整
    print("自己定的R Matrix: ", R)
 
 
def sample_b():
    # 定义方形画布像素坐标长度
    canvas_square_size = 320
    # 定义立方体的边长
    length = 1
 
    # 定义立方体的8个顶点坐标
    vertices_w = np.array([
        [-length / 2, -length / 2, -length / 2],
        [-length / 2, -length / 2, length / 2],
        [-length / 2, length / 2, -length / 2],
        [-length / 2, length / 2, length / 2],
        [length / 2, -length / 2, -length / 2],
        [length / 2, -length / 2, length / 2],
        [length / 2, length / 2, -length / 2],
        [length / 2, length / 2, length / 2]])
 
    # 手动定一组相机内参K
    fx = 800
    fy = 800
    cx = canvas_square_size // 2
    cy = canvas_square_size // 2
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
 
    # 初始化角度
    a = 0
    while True:
        # 手动定一个相机外参R旋转矩阵，并设置让三个轴旋转a度
        R = construct_extrinsic_matrix_R(a, a, a)
 
        # 手动定一个相机外参的偏移向量t 即在x, y, z的位置看面朝单位
        t1 = 0
        t2 = 0
        t3 = 5  # 数值越大则距离观测目标距离则越长
        T = np.array([t1, t2, t3])
        # 求基于相机坐标系的顶点集
        vertices_c = np.matmul(R, vertices_w.T).T + T
        # 使用透视投影解出像素坐标的顶点集
        vertices_uv = perspective_projection(vertices_c, K)
        # 显示求解后的uv顶点集
        display_vertices_uv(vertices_uv, wait_key=30, canvas_size=(canvas_square_size, canvas_square_size))
        a += 1
 
 
if __name__ == '__main__':
    # sample_a()  # 示例1
    sample_b()  # 示例2
