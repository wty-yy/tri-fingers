from ultralytics import YOLO
import time
import pyrealsense2 as rs
import numpy as np
import cv2
from get_extri import image_to_earth
import torchvision
import tcp
from scipy.spatial import ConvexHull
from scipy.spatial import distance
from threading import Thread
model = YOLO("./best3.pt")
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

tcp_init=False
if __name__=='__main__':
    fps, w, h = 30, 1280, 720
    cam = Camera(w, h, fps)
    j=0
    if tcp_init:
        client=tcp.tcp_init()
    while True:
        img_bgr, depth_image = cam.get_frame() # 读取图像帧，包括RGB图和深度图  
        #print(img_bgr.shape,depth_image.shape)
       
        results = model(img_bgr, verbose=False)  # verbose设置为False，不单独打印每一帧预测结果
        # 预测框的个数
        num_bbox = len(results[0].boxes.cls)
    
        # 预测框的 xyxy 坐标
        bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')
    

        earth_point=[]
        sent_data=''
        for idx in range(num_bbox):  # 遍历每个框
    
            # 获取该框坐标
            bbox_xyxy = bboxes_xyxy[idx]
    
            # 获取框的预测类别（对于关键点检测，只有一个类别）
            conf = results[0].boxes.conf[idx].cpu().numpy()
            bbox_label=str(conf)
            # 画框
            
            if conf>0.55:
                # img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color,
                #                         bbox_thickness)
                xys=results[0].masks.xy[idx].astype(int)
                # print(xys)
                for x,y in xys:
                    cv2.circle(img_bgr,(x,y),2,(255,0,0))
                hull=ConvexHull(xys)
                hull_points=xys[hull.vertices]
                i=0
                for x,y in hull_points:
                    cv2.circle(img_bgr,(x,y),4,(0,0,255))
                    cv2.putText(img_bgr, str(i),
                                (x,y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color,
                                bbox_labelstr['font_thickness'])
                    i+=1
                min_dist=np.full(len(xys),np.inf)
                for i in range(len(hull_points)):
                    p1=hull_points[i]
                    p2=hull_points[(i+1)%len(hull_points)]
                    current_dist=distance.cdist(xys,[p1,p2],'euclidean').min(axis=1)
                    min_dist=np.minimum(min_dist,current_dist)
                max_index=np.argmax(min_dist)
                cv2.circle(img_bgr,xys[max_index],5,(0,255,255))
                # print("minist:",min_dist)
                
                
                
                # 求外切圆圆心
                img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color,
                                        bbox_thickness)
                xys=results[0].masks.xy[idx].astype(int)
                maxDistPt, radius=cv2.minEnclosingCircle(xys)
                maxDistPt=list(map(int, maxDistPt))
                #print(maxDistPt)
                depth_val=depth_image[int(maxDistPt[1]),int(maxDistPt[0])]
                earth_point.append(image_to_earth(maxDistPt,depth_val).reshape(1,3))
                # 绘制圆
                cv2.circle(img_bgr, maxDistPt, int(radius), (0, 255, 0), 2, 1, 0)
                # 绘制圆心
                cv2.circle(img_bgr, maxDistPt, 1, (0, 255, 0), 2, 1, 0)
                cv2.putText(img_bgr, str(image_to_earth(maxDistPt,depth_val).reshape(3)),
                                maxDistPt,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color,
                                bbox_labelstr['font_thickness'])
                
                '''
                求内切圆圆心
                data=results[0].masks.data[idx].numpy().astype(np.uint8)
                mask_gray=np.multiply(data,255)
                #cv2.imshow("mask_gray",mask_gray)
            
                contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #计算到轮廓的距离
                raw_dist = np.empty(mask_gray.shape, dtype=np.float32)
                for i in range(mask_gray.shape[0]):
                    for j in range(mask_gray.shape[1]):
                        raw_dist[i, j] = cv2.pointPolygonTest(contours[0], (j, i), True)

                # 获取最大值即内接圆半径，中心点坐标
                minVal, maxVal, _, maxDistPt = cv2.minMaxLoc(raw_dist)
                # 半径：maxVal  圆心：maxDistPt
                # 转换格式
                maxVal = abs(maxVal)
                radius = int(maxVal)
                # 绘制圆
                cv2.circle(img_bgr, maxDistPt, int(radius), (0, 0, 255), 2, 1, 0)
                # 绘制圆心
                cv2.circle(img_bgr, maxDistPt, 1, (0, 0, 255), 2, 1, 0)
                for x,y in xys:
                    cv2.circle(img_bgr, (x, y), radius=2, color=(0,0,50*idx%255), thickness=4)
                '''
            # 写框类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
            '''
            img_bgr = cv2.putText(img_bgr, bbox_label,
                                (bbox_xyxy[0] + bbox_labelstr['offset_x'], bbox_xyxy[1] + bbox_labelstr['offset_y']),
                                cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color,
                                bbox_labelstr['font_thickness'])
            '''
        if len(earth_point)!=0:
            # print(earth_point)
            earth_point=np.mean(earth_point,axis=0)
            # print(earth_point)
            if tcp_init:
                sent_data=f"X:{earth_point[0][0]:.4f} Y:{earth_point[0][1]:.4f} Z:{earth_point[0][2]:.4f}"
                tcp.send_data(client, 'SEND_DATA',sent_data)

        cv2.imshow('RealSense', img_bgr)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            j=j+1
            cv2.imwrite('image_r_{}.png'.format(str(j).zfill(5)), img_bgr)
            # cv2.imwrite('/home/wq/savefig/depth/Tbimage_d_{}.png'.format(str(i).zfill(5)), np.asarray(depth_image,np.uint16))
            print('保存成功')
        elif key & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            client.close()
            break
    cam.release()
 