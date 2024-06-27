from ultralytics import YOLO
import time
import pyrealsense2 as rs
import numpy as np
import cv2
from get_extri import image_to_earth

model = YOLO("/home/wq/camera/best.pt")
model.to('cpu')
class Camera(object):
    '''
    realsense相机处理类
    '''
    def __init__(self, width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16,  fps)
        # self.align = rs.align(rs.stream.color) # depth2rgb
        self.pipeline.start(self.config)  # 开始连接相机
 
 
    def get_frame(self):
        frames = self.pipeline.wait_for_frames() # 获得frame (包括彩色，深度图)
        # 创建对齐对象
        align_to = rs.stream.color            # rs.align允许我们执行深度帧与其他帧的对齐
        align = rs.align(align_to)            # “align_to”是我们计划对齐深度帧的流类型。
        aligned_frames = align.process(frames)
        # 获取对齐的帧
        depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame是对齐的深度图
        color_frame = aligned_frames.get_color_frame()
        #colorizer = rs.colorizer()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        #colorizer_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
        return color_image, depth_image                         #,colorizer_depth
 
    def release(self):
        self.pipeline.stop()


# 框（rectangle）可视化配置
bbox_color = (150, 0, 0)             # 框的 BGR 颜色
bbox_thickness = 2                   # 框的线宽
 
# 框类别文字
bbox_labelstr = {
    'font_size':1,         # 字体大小
    'font_thickness':2,   # 字体粗细
    'offset_x':0,          # X 方向，文字偏移距离，向右为正
    'offset_y':-10,       # Y 方向，文字偏移距离，向下为正
}


# 点类别文字
kpt_labelstr = {
    'font_size':0.5,             # 字体大小
    'font_thickness':1,       # 字体粗细
    'offset_x':10,             # X 方向，文字偏移距离，向右为正
    'offset_y':0,            # Y 方向，文字偏移距离，向下为正

}


if __name__=='__main__':
    fps, w, h = 30, 1280, 720
    cam = Camera(w, h, fps)
    i=0
    while True:
        img_bgr, depth_image = cam.get_frame() # 读取图像帧，包括RGB图和深度图  
        #print(img_bgr.shape,depth_image.shape)
        gray = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
        depth_image = depth_image.astype(np.uint8)
        gray_depth = np.concatenate((gray,depth_image),axis=1)

        cv2.imshow('gray_depth',gray_depth)

        mix = cv2.addWeighted(gray,0.5,depth_image,0.5,0)

        cv2.imshow('mix',mix)

        
        cv2.imshow('RealSense', img_bgr)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            i=i+1
            cv2.imwrite('/home/wq/savefig/rgb/image_r_{}.png'.format(str(i).zfill(5)), img_bgr)
            cv2.imwrite('/home/wq/savefig/depth/Tbimage_d_{}.png'.format(str(i).zfill(5)), np.asarray(depth_image,np.uint16))
            print('保存成功')
        elif key & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    cam.release()
 