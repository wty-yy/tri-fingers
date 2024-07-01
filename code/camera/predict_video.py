from ultralytics import YOLO
import time
import pyrealsense2 as rs
import numpy as np
import cv2
from get_extri import image_to_earth ,earth_to_image
import torchvision
import tcp
from scipy.spatial import ConvexHull
from scipy.spatial import distance
from pathlib import Path
from threading import Thread

path_video = Path(__file__).parent / "vidoes"
path_video.mkdir(exist_ok=True)
model = YOLO("./best3.pt")  # YOLO模型权重文件
model.to('cpu')
tcp_init = True  # TCP 通讯开关（先开win服务器，再开这个）
target_init = True  # 绘制红色目标

class Camera(object):  
    '''
    realsense相机处理类
    '''
    def __init__(self, width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        # self.config = rs.config()
        # self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, fps)
        # self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16,  fps)


        # self.align = rs.align(rs.stream.color) # depth2rgb

        connect_device=[]
        self.connect_device_name=[]
        for d in rs.context().devices:
            print('found device:',d.get_info(rs.camera_info.name),d.get_info(rs.camera_info.serial_number))
            connect_device.append(d.get_info(rs.camera_info.serial_number))
            self.connect_device_name.append(d.get_info(rs.camera_info.name))
        if len(connect_device)<2:
            print("相机数量不足")
            
        self.config_list=[rs.config() for i in range(len(connect_device))]
        self.pipeline_list=[rs.pipeline() for i in range(len(connect_device))]
        for n, config in enumerate(self.config_list):
            config.enable_device(connect_device[n])
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, fps)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16,  fps)
        for n,pipeline in enumerate(self.pipeline_list):
            pipeline.start(self.config_list[n])
        # self.pipeline1 = rs.pipeline()
        # rs.config.enable_device(rs.config(),connect_device[0])
        # self.pipeline1.start(self.config)  # 开始连接相机

        # self.pipeline2 = rs.pipeline()
        # rs.config.enable_device(rs.config(),connect_device[1])
        # self.pipeline2.start(self.config)  # 开始连接相机
 
 
    def get_frame(self):
        Color_image=[]
        Depth_image=[]
        for pipeline in self.pipeline_list:
            frames = pipeline.wait_for_frames() # 获得frame (包括彩色，深度图)
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
            Color_image.append(color_image)
            Depth_image.append(depth_image)
            #colorizer_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
        return Color_image, Depth_image                     #,colorizer_depth
 
    def release(self):

        self.pipeline1.stop()
        self.pipeline2.stop()


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
from scipy.spatial import distance
import time

vid_writer = None
def save_vid():
    global vid_writer
    if vid_writer is not None:
        vid_writer.release()
        vid_writer = None
        print("Record release, save video!")

if __name__=='__main__':
    fps, w, h = 30, 1280, 720
    scale_ratio = 0.8
    sw, sh = int(w*scale_ratio), int(h*scale_ratio)
    cam = Camera(w, h, fps)
    j=0
    if tcp_init:  # TCP 接受使用多进程
        client=tcp.tcp_init()
        tread=Thread(target=tcp.get_target,args=(client,))
        tread.setDaemon(True)
        tread.start()

    while True:
        sent_data=f"X:2.0 Y:0.0 Z:1.0"
        img_bgrs, depth_images = cam.get_frame() # 读取图像帧，包括RGB图和深度图  
        cube_pos=[]  # 方块上表面中心的世界坐标
        imgs=[]
        for img_bgr,depth_image,cam_name in zip(img_bgrs,depth_images,cam.connect_device_name):
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
                    # hull=ConvexHull(xys)
                    # hull_points=xys[hull.vertices]
                    # for x,y in hull_points:
                    #     depth_val=depth_image[int(y),int(x)]
                    #     cube_points.append(image_to_earth([x,y],depth_val).reshape(1,3))
                    #     cv2.circle(img_bgr,(x,y),2,(255,0,0))

                    # min_dist=np.full(len(xys),np.inf)
                    # for i in range(len(hull_points)):
                    #     p1=hull_points[i]
                    #     p2=hull_points[(i+1)%len(hull_points)]
                    #     current_dist=distance.cdist(xys,[p1,p2],'euclidean').min(axis=1)
                    #     min_dist=np.minimum(min_dist,current_dist)
                    # max_index=np.argmax(min_dist)
                    # cv2.circle(img_bgr,xys[max_index],5,(0,255,255))
                    # print("minist:",min_dist)
                    
                    
                    
                    # 求外切圆圆心就是正方形上表面中心（如果有两个则再取一次平均）
                    img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color,
                                            bbox_thickness)
                    xys=results[0].masks.xy[idx].astype(int)
                    maxDistPt, radius=cv2.minEnclosingCircle(xys)
                    maxDistPt=list(map(int, maxDistPt))
                    #print(maxDistPt)
                    depth_val=depth_image[int(maxDistPt[1]),int(maxDistPt[0])]
                    earth_point.append(image_to_earth(maxDistPt,depth_val,cam_name).reshape(1,3))
                    # 绘制圆
                    # cv2.circle(img_bgr, maxDistPt, int(radius), (0, 255, 0), 2, 1, 0)
                    # 绘制圆心
                    cv2.circle(img_bgr, maxDistPt, 1, (0, 255, 0), 2, 1, 0)
                    cv2.putText(img_bgr, str(image_to_earth(maxDistPt,depth_val,cam_name).reshape(3)),
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
                earth_point=np.mean(earth_point,axis=0)
                # print(earth_point)
                if distance.euclidean(earth_point[0],[0,0,0.064])<0.2:  # 排除距离原点非常的点
                    cube_pos.append(earth_point)
            #     # print(earth_point)
            #     if tcp_init:
            #         sent_data=f"X:{earth_point[0][0]:.4f} Y:{earth_point[0][1]:.4f} Z:{earth_point[0][2]:.4f}"
            #         tcp.send_data(client, 'SEND_DATA',sent_data)
            if all(tcp.target_pos) !=None and target_init:  # 打印出来的正方形块的边长为6.4cm，TCP传输的是正方形的重心
                # print(tcp.target_pos)
                # 分别找到8个顶点的世界坐标
                p1=np.add(tcp.target_pos,[0.032,0.032,-0.032])
                p2=np.add(tcp.target_pos,[-0.032,0.032,-0.032])
                p3=np.add(tcp.target_pos,[-0.032,-0.032,-0.032])
                p4=np.add(tcp.target_pos,[0.032,-0.032,-0.032])
                p5=np.add(tcp.target_pos,[0.032,0.032,0.032])
                p6=np.add(tcp.target_pos,[-0.032,0.032,0.032])
                p7=np.add(tcp.target_pos,[-0.032,-0.032,0.032])
                p8=np.add(tcp.target_pos,[0.032,-0.032,0.032])
                ps=[p1,p2,p3,p4,p5,p6,p7,p8]
                image_points=[earth_to_image(p,cam_name) for p in ps]
                for i in range(4):
                    cv2.line(img_bgr,image_points[i],image_points[(i+1)%4],color=(0, 0, 255),thickness=2)
                    cv2.line(img_bgr,image_points[4+i],image_points[(i+5) if i!=3 else 4],color=(0, 0, 255),thickness=2)
                    cv2.line(img_bgr,image_points[i],image_points[(i+4)],color=(0, 0, 255),thickness=2)
            # cv2.imshow('RealSense', img_bgr)
            imgs.append(img_bgr)
        imgs = [cv2.resize(img, (sw, sh)) for img in imgs]
        if len(imgs)>1:
            img=np.hstack([imgs[0],imgs[1]])
            cv2.imshow('realsense',img)
        else:
            cv2.imshow('realsense',imgs[0])

        if len(cube_pos)>0:
            
            pos=np.mean(cube_pos,axis=0)
            # print(pos,distance.euclidean(pos[0],[0,0,0.064]))
            if distance.euclidean(pos[0],[0,0,0.064])<0.2:
                if tcp_init:
                    sent_data=f"X:{pos[0][0]:.4f} Y:{pos[0][1]:.4f} Z:{pos[0][2]:.4f}"
                    tcp.send_data(client, 'SEND_DATA',sent_data)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            j=j+1
            cv2.imwrite('image_r_{}.png'.format(str(j).zfill(5)), img_bgr)
            # cv2.imwrite('/home/wq/savefig/depth/Tbimage_d_{}.png'.format(str(i).zfill(5)), np.asarray(depth_image,np.uint16))
            print('保存成功')
        elif key & 0xFF == ord('q'):
            save_vid()
            cv2.destroyAllWindows()
            client.close()
            break
        elif key & 0xFF == ord('r'):
            if vid_writer is not None:
                save_vid()
            else:
                path_save = path_video / f"{time.strftime('%m%d_%H%M%S')}.mp4"
                vid_writer = cv2.VideoWriter(str(path_save), cv2.VideoWriter_fourcc(*'mp4v'), fps//3, (sw, sh))
                print(f"Start record to {path_save}")
        if vid_writer is not None:
            vid_writer.write(imgs[0])
    cam.release()
 