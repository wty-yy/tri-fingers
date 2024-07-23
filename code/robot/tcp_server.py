import socket  # 导入 socket 模块
from threading import Thread
import time
import json
import math
import numpy as np
import Gcan
import math
ADDRESS = ('192.168.43.84',  123)  # 本机在局域网下的ipv4地址，端口随便给一个空置的

g_socket_server = None  # 负责监听的socket
 
g_conn_pool = {}  # 连接池
client_type='trifinger'  #主机名
def tcp_init():
    """
    初始化服务端
    """
    global g_socket_server
    g_socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
    g_socket_server.bind(ADDRESS)
    g_socket_server.listen(5)  # 最大等待数（有很多人理解为最大连接数，其实是错误的）
    print("server start，wait for client connecting...")

def accept_client():
    """
    接收新连接
    """
    while True:
        client, info = g_socket_server.accept()  # 阻塞，等待客户端连接
        # 给每个客户端创建一个独立的线程进行管理
        thread = Thread(target=message_handle, args=(client, info))
        # 设置成守护线程
        thread.setDaemon(True)
        thread.start()
def send_data(client,cmd,kv):
    global client_type
    jd={}
    jd['COMMAND']= cmd
    jd['client_type']= client_type
    jd['data']=kv
    jsonstr=json.dumps(jd)
    client.sendall(jsonstr.encode('utf8'))

offset_buffer=[]
def message_handle(client, info):
    """
    消息处理
    """
    client.sendall("connect server successfully!".encode(encoding='utf8'))
    
    while True:
        try:
            bytes = client.recv(1024)
            msg = bytes.decode(encoding='utf8')
            if len(msg)>120:
                continue
            jd = json.loads(msg)
            cmd = jd['COMMAND']
            client_type = jd['client_type']
            if 'CONNECT' == cmd:
                g_conn_pool[client_type] = client
                print('on client connect: ' + client_type, info)
            elif 'SEND_DATA' == cmd:
                #print('recv client msg: ' + client_type, jd['data'])
                if client_type=='euler':
                    parts = jd['data'].split('\t')
                    parts = [part for part in parts if part]
                    if len(parts)==6:
                        parts =parts[3:6]  #有时会一次收到两个数据包
                    #print(parts)
                    
                    trifinger_state.euler = [float(part.split(":")[1]) if part.split(":")[1].replace('.', '', 1).replace('-', '', 1).isdigit() else 0 for part in parts]
                    if (trifinger_state.euler_offset_init==False)&(trifinger_state.euler[2]!=0):
                        offset_buffer.append(trifinger_state.euler)
                        if len(offset_buffer)>20:
                            trifinger_state.euler_offset=np.mean(offset_buffer,axis=0)
                            trifinger_state.euler_offset_init=True
                            print("euler_offset 标定成功：",trifinger_state.euler_offset)
                    if trifinger_state.euler_offset_init==True:
                        trifinger_state.euler=trifinger_state.euler-trifinger_state.euler_offset
                        if trifinger_state.euler[2]>180:    #将偏航角限制在-180~180
                            trifinger_state.euler[2]=trifinger_state.euler[2]-360
                        elif trifinger_state.euler[2]<-180:
                            trifinger_state.euler[2]=trifinger_state.euler[2]+360
                        trifinger_state.cube_state[0,3:7]=euler_to_quaternion(trifinger_state.euler[0],trifinger_state.euler[1],trifinger_state.euler[2])

                elif client_type=='pose':
                    parts = jd['data'].split(' ')
                    parts = [part for part in parts if part]
                    if len(parts)==6:
                        parts =parts[3:6]  #有时会一次收到两个数据包
                    #print(parts)
                    trifinger_state.cube_state[0,0:3] = [float(part.split(":")[1]) if part.split(":")[1].replace('.', '', 1).replace('-', '', 1).isdigit() else 0 for part in parts]
                    trifinger_state.cube_state[0,2]=0.0325
                    
                    send_data(client,'SEND_DATA',str(trifinger_state.target_cube_state[0][0:3])) 
                #print(trifinger_state.euler,trifinger_state.euler_offset)  
                elif client_type=='sim':
                    pos=trifinger_state.dof_pos[0].copy()
                    pos=pos[[3,4,5,6,7,8,0,1,2]]
                    data=np.append(pos,trifinger_state.cube_state[0])
                    data=np.append(data,trifinger_state.target_cube_state[0])
                    send_data(client,'SEND_DATA',str(data))   
        except Exception as e:
            print(e)
            remove_client(client_type)
            break

def remove_client(client_type):
    client = g_conn_pool[client_type]
    if None != client:
        client.close()
        g_conn_pool.pop(client_type)
        print("client offline: " + client_type)

class Trifinger_state:       #41
    def __init__(self) -> None:
        self.dof_pos=np.zeros([1,9])
        self.dof_vel=np.zeros([1,9])
        self.cube_state=np.zeros([1,7])
        self.target_cube_state=np.zeros([1,7])
        self.last_action=np.zeros([1,9])
        ##41
        self.sim_frict=np.array([0.7744,0.6816,-0.3928]*3,dtype=np.float32)
        self.real_frict=np.array([0.476,0.45,-0.25]*3,dtype=np.float32)
        self.torque_scale=self.sim_frict/self.real_frict
        #self.dof_pos_low=np.array([-0.33, 0.0, -2.7]*3,dtype=np.float32)
        #self.dof_pos_high=np.array([1.0, 1.57, 0.0]*3,dtype=np.float32)
        self.dof_pos_low=np.array([-0.50, 0.3, -2.7]*3,dtype=np.float32)
        self.dof_pos_high=np.array([0.50, 1.57, -0.0]*3,dtype=np.float32)
        self.dof_pos_default=np.array([0.0,0.9,-1.7]*3,dtype=np.float32)
        self.dof_vel_low=np.array([-3.14159268/4]*9,dtype=np.float32)
        self.dof_vel_high=np.array([3.14159268/4]*9,dtype=np.float32)
        self.cube_state_low=np.array([-0.3, -0.3, 0,-1,-1,-1,-1], dtype=np.float32),
        self.cube_state_high=np.array([0.3, 0.3, 0.3,1,1,1,1], dtype=np.float32),
        self.max_distance_to_center =0.1387083487540115  #m
        self.dof_pos_offset=np.array([0,math.pi/2,0,0,math.pi/2,0,0,math.pi/2,0])
        self.euler=np.zeros([1,3])  #["AngleX", "AngleY", "AngleZ"]
        self.euler_offset=np.zeros([1,3])
        self.action_scale = 1.8
        self.safe_flag=True
        self.euler_offset_init=False
        self.target_init=False
        self.caninit=False
    def target_set(self,x,y,z,Anglex,Angley,Anglez):
        quat=euler_to_quaternion(Anglez,Angley,Anglex)
        pose=np.array([x,y,z])
        self.target_cube_state[0]=np.append(pose,quat)
        self.target_init=True
        print("target:",self.target_cube_state)
    def motor_control(self,action:list):
        if len(action) !=9:
            print("action length wrong!")
        else:
            action=np.clip(action,self.dof_pos_low,self.dof_pos_high)
            
            

            action=np.multiply(np.subtract(action,self.dof_pos[0]),9/(2*math.pi))
            #print(action)
            
            
            Gcan.sendcan1(0,1,action[0])
            time.sleep(0.003)
            Gcan.sendcan1(0,2,action[1])
            time.sleep(0.003)
            Gcan.sendcan1(0,3,action[2])
            time.sleep(0.003)
            
            Gcan.sendcan1(0,4,action[3])
            time.sleep(0.003)
            Gcan.sendcan1(0,5,action[4])
            time.sleep(0.003)
            Gcan.sendcan2(0,6,action[5])
            time.sleep(0.003)
            
            Gcan.sendcan2(0,7,action[6])
            time.sleep(0.003)
            Gcan.sendcan2(0,8,action[7])
            time.sleep(0.003)
            Gcan.sendcan2(0,9,action[8])
            time.sleep(0.003)
            
    def input_normalize(self,val,high,low):
        #print(val,high,low)
        m=np.multiply(np.add(high,low),0.5)
        d=np.multiply(np.subtract(high,low),0.5)
        #print(m,d)
        return np.divide(np.subtract(val,m),d)
    def random_xy( self,max_com_distance_to_center: float=0.1387083487540115 ) :
        """Returns sampled uniform positions in circle (https://stackoverflow.com/a/50746409)"""
        # sample radius of circle
        radius = math.sqrt(np.random.rand())
        radius *= max_com_distance_to_center
        # sample theta of point
        theta = 2 * np.pi * np.random.rand()
        # x,y-position of the cube
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)

        return x, y

trifinger_state=Trifinger_state()

def euler_to_quaternion(roll, pitch, yaw):   #yaw (Z), pitch (Y), roll (X)
    """
    Convert Euler angles to quaternion.

    Args:
    - roll: Roll angle in degree
    - pitch: Pitch angle in degree
    - yaw: Yaw angle in degree

    Returns:
    - Quaternion as a numpy array [w, x, y, z]
    """
    roll = math.pi *roll/180
    pitch = math.pi *pitch/180
    yaw = math.pi *yaw/180
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return np.array([qx, qy, qz, qw])


if __name__ == '__main__':
    tcp_init()
    # 新开一个线程，用于接收新连接
    thread = Thread(target=accept_client)
    thread.setDaemon(True)
    thread.start()
    # 主线程逻辑
    while True:
        time.sleep(0.1)

