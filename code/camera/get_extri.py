import numpy as np
import time
import pyrealsense2 as rs
import numpy as np
import cv2
import math

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
'''
Calculation Pose Matrix from other geometric elements.
'''
def calPoseFrom3Points(Oab, Pxb, Pyb):
    '''
    作者：xyq；
    日期：2022.1.18；
    功能：已知坐标系a的原点、x轴正半轴上任一点和y轴正半轴上任一点在坐标系b下的坐标，
        求解坐标系a到坐标系b的旋转矩阵R和平移矩阵T；
    输入：坐标系a的原点在坐标系b下的坐标:Oab(x1,y1,z1);
        坐标系a的x轴正半轴上任一点在坐标系b下的坐标:Pxb(x2,y2,z2);
        坐标系a的y轴正半轴上任一点在坐标系b下的坐标:Pyb(x3,y3,z3);
    输出：坐标系n到坐标系s的旋转矩阵Rns，输出格式为矩阵;
    DEMO：
            import geomeas as gm
            import numpy as np

            Oab = np.array([-37.84381632, 152.36389864, 41.68600167])
            Pxb = np.array([-19.59820338, 139.58818292, 45.55380309])
            Pyb = np.array([-38.23270656, 157.3130709, 59.86810327])

            print(gm.Pose().calPoseFrom3Points(Oab, Pxb, Pyb))
    '''
    print(Oab,Pxb,Pyb)
    x = (Pxb - Oab) / np.linalg.norm(Pxb - Oab)
    y = (Pyb - Oab) / np.linalg.norm(Pyb - Oab)
    z = np.cross(x, y)
    length = np.linalg.norm(z)
    z = z / length
    Rab = np.matrix([x, y, z]).transpose()
    Tab = np.matrix(Oab).transpose()
    return Rab, Tab                

def calOrientationFrom2Vectors( Vs1, Vs2, Vn1, Vn2):
    '''
    作者：xyq；
    日期：2022.1.18；
    功能：根据两个不共线的矢量分别在两个坐标系下的矢量坐标求解这两个坐标系之间的旋转矩阵；
    输入：两个矢量在坐标系s下的坐标:Vs1(xs1,ys1,zs1),Vs2(xs2,ys2,zs2);
        在坐标系n下的坐标:Vn1(xn1,yn1,zn1),Vn2(xn2,yn2,zn2);
    输出：坐标系n到坐标系s的旋转矩阵Rns，输出格式为矩阵;
    DEMO：
            import geomeas as gm
            import numpy as np

            Vs1 = np.array([0.55397988, 0.82791962, -0.08749517])
            Vs2 = np.array([0.02063334, -0.26258813, -0.96468738])
            Vn1 = np.array([0.97066373, 0.20744552, 0.12156592])
            Vn2 = np.array([0, 0, -1])

            print(gm.Pose().calOrientationFrom2Vectors(Vs1, Vs2, Vn1, Vn2))
    '''
    # frame s
    a = Vs1
    b = np.cross(Vs1, Vs2) / np.linalg.norm(np.cross(Vs1, Vs2))
    c = np.cross(a, b)
    # 参考坐标系frame d
    Rsd = np.array([a, b, c])

    # frame n
    A = Vn1
    B = np.cross(Vn1, Vn2) / np.linalg.norm(np.cross(Vn1, Vn2))
    C = np.cross(A, B)
    Rnd = np.array([A, B, C])
    Rns = np.dot(np.linalg.inv(Rsd), Rnd)

    return Rns

# 已知相机的内参矩阵和畸变矩阵（示例值，需要替换为实际值）
camera_matrix = np.array([[914.9191276677989, 0, 643.1500962950337],  #fx 0 cx; 0 fy cy; 0 0 1;
                          [0, 915.1587165967329, 359.53871968201054],
                          [0, 0, 1]])

dist_coeffs = np.array([0.10927442315826, -0.23448282190486733, 0.0013576548645471817, 0.0010983563521488342, 0])

def image_to_camera(image_point, depth_value):
    
    # 将图像坐标转换为归一化坐标
    normalized_point = np.linalg.inv(camera_matrix).dot(np.array([image_point[0], image_point[1], 1]))

    # 计算深度方向的相机坐标
    #camera_point = normalized_point * depth_value/1000
    #print(normalized_point * depth_value/1000)
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

#model = YOLO("/home/amov/temptest/orange_s_v2.pt")
'''
extri_R=np.asarray([[-0.47745941,  0.83653434,  0.0111403 ],
                    [ 0.66286799,  0.4215384,  -0.65178018],
                    [-0.57674825, -0.35002241, -0.75832611]])
extri_T=np.asarray([[ 0.03260845],
                    [-0.01863098],
                    [ 0.501     ]])
'''
extri_R=np.asarray([[-0.45387007,  0.84991783,  0.00676841],
                    [ 0.67944918,  0.40663435, -0.64432766],
                    [-0.57649873, -0.33509429, -0.7647196 ]])
extri_T=np.asarray([[ 0.02923178],
                    [-0.01699791],
                     [ 0.501     ]])
def image_to_earth(image_point,depth_value):
    camera_point=image_to_camera(image_point,depth_value).reshape(3,1)
    earth_point=np.linalg.inv(extri_R).dot(camera_point.reshape(3,1)-extri_T)
    return np.array(earth_point)




if __name__=='__main__':
    fps, w, h = 30, 1280, 720
    cam = Camera(w, h, fps)
    i=0
    while True:
        color_image, depth_image = cam.get_frame() # 读取图像帧，包括RGB图和深度图 
        result = cv2.blur(color_image, (4,4))
        gray=cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
        # ret,gray=cv2.threshold(gray,180,255,cv2.THRESH_BINARY)  # 阈值二值化: 180
        ret,gray=cv2.threshold(gray,230,255,cv2.THRESH_BINARY)
        cv2.imshow('gray', gray)
        circles= cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,50,param1=80,param2=10,minRadius=1,maxRadius=40)
        
        if circles is not None:
            print(len(circles[0]))

            for circle in circles[0]:

                #坐标行列(就是圆心)
                x=int(circle[0])
                y=int(circle[1])
                #半径
                r=int(circle[2])
                #在原图用指定颜色圈出圆，参数设定为int所以圈画存在误差
                color_image=cv2.circle(color_image,(x,y),r,(0,0,255),1,8,0)
        cv2.imshow('RealSense', color_image)

        key = cv2.waitKey(1)

        if key & 0xFF == ord('s'):
            cv2.imwrite(f"/home/wq/savefig/rgb/color_image{i}.png",color_image)
            cv2.imwrite(f"/home/wq/savefig/depth/depth_image{i}.png",color_image)
            i=i+1
            print("save success")
        elif key&0xFF==ord('c'):
            if len(circles[0])==3:
                points=np.zeros([3,3])
                camera_points=np.zeros([3,3])
                points[:,:2]=circles[0][:,0:2]
                for idx in range(3):
                    points[idx,2]=depth_image[int(points[idx,1]),int(points[idx,0])]
                    camera_points[idx]=image_to_camera(points[idx,0:2], points[idx,2])
                print("图像坐标")
                print(points)
                print("相机坐标")
                print(camera_points)
                Opoint=0
                max_dis=0
                for idx in range(3):
                    dis=math.dist(points[idx-1],points[idx])
                    if dis>max_dis:
                        max_dis=dis
                        Opoint=(idx+1)%3
                cv2.putText(color_image,"OO",(int(points[Opoint][0]),int(points[Opoint][1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,225,0),2)
                cv2.putText(color_image,"11",(int(points[(Opoint+1)%3][0]),int(points[(Opoint+1)%3][1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,225,0),2)
                cv2.putText(color_image,"22",(int(points[(Opoint+2)%3][0]),int(points[(Opoint+2)%3][1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,225,0),2)
                cv2.imwrite("calibrate.png",color_image)
                time.sleep(0.1)
                a=int(input("输入x坐标序号 1 2\n"))
                if a==2:
                    b=1
                else:
                    b=2
                print(a,b)
                R,T=calPoseFrom3Points(camera_points[Opoint], camera_points[(Opoint+a)%3], camera_points[(Opoint+b)%3])
                print(R,T)
                break
        elif key & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    cam.release()

    '''
 图像坐标
[[658.5 384.5 476. ]
 [760.5 361.5 487. ]
 [695.5 330.5 501. ]]
相机坐标
[[ 0.01191227  0.010102    0.476     ]
 [ 0.06606768 -0.00177053  0.487     ]
 [ 0.03260845 -0.01863098  0.501     ]]
输入x坐标序号 1 2
1
1 2
[ 0.03260845 -0.01863098  0.501     ] [0.01191227 0.010102   0.476     ] [ 0.06606768 -0.00177053  0.487     ]
[[-0.47745941  0.83653434  0.0111403 ]
 [ 0.66286799  0.4215384  -0.65178018]
 [-0.57674825 -0.35002241 -0.75832611]] [[ 0.03260845]
 [-0.01863098]
 [ 0.501     ]]


图像坐标
[[661.5 383.5 476. ]
 [696.5 328.5 501. ]
 [764.5 359.5 487. ]]
相机坐标
[[ 9.54960483e-03  1.24665575e-02  4.76000000e-01]
 [ 2.92317813e-02 -1.69979070e-02  5.01000000e-01]
 [ 6.47407402e-02 -9.01732536e-06  4.87000000e-01]]
输入x坐标序号 1 2
2
2 1
[ 0.02923178 -0.01699791  0.501     ] [0.0095496  0.01246656 0.476     ] [ 6.47407402e-02 -9.01732536e-06  4.87000000e-01]
[[-0.45387004  0.84991785  0.00676841]
 [ 0.67944913  0.4066343  -0.64432773]
 [-0.5764988  -0.33509431 -0.76471954]] [[ 0.02923178]
 [-0.01699791]
 [ 0.501     ]]
    '''

