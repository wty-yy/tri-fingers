import numpy as np
import cv2

camera_matrix = np.array([[924.5472412109375, 0, 635.368896484375],  #fx 0 cx; 0 fy cy; 0 0 1;
                          [0, 924.8556518554688, 364.8779296875],
                          [0, 0, 1]])

dist_coeffs = np.array([0.10927442315826, -0.23448282190486733, 0.0013576548645471817, 0.0010983563521488342, 0])

def image_to_camera(image_point, depth_value):
    
    # 将图像坐标转换为归一化坐标
    normalized_point = np.linalg.inv(camera_matrix).dot(np.array([image_point[0], image_point[1], 1]))

    # 计算深度方向的相机坐标
    #camera_point = normalized_point * depth_value/1000
    print(normalized_point * depth_value/1000)
    # 应用畸变校正
    x = normalized_point[0]
    y = normalized_point[1]
    r2 = x**2 + y**2
    r4 = r2**2
    r6 = r2*r4
    x_distorted = x * (1 + dist_coeffs[0]*r2 + dist_coeffs[1]*r4 + dist_coeffs[4]*r6) + \
                  2*dist_coeffs[2]*x*y + dist_coeffs[3]*(r2 + 2*x**2)
    y_distorted = y * (1 + dist_coeffs[0]*r2 + dist_coeffs[1]*r4 + dist_coeffs[4]*r6) + \
                  dist_coeffs[2]*(r2 + 2*y**2) + 2*dist_coeffs[3]*x*y
    camera_point=np.asarray([x_distorted,y_distorted,1])*(depth_value/1000)

    return np.array(camera_point)


image_point=[345.23,334.32] 
print(image_to_camera(image_point,300))

#[-9.42037480e-02 -1.00111353e-02  3.00000000e+02]
#-0.09484457 -0.01005336  0.3 