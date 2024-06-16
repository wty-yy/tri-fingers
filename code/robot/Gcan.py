import os,sys
import threading
import ctypes
from ctypes import *
import struct
import time
import numpy as np
import tcp_server
import math
DevType = c_uint

'''
    Device Type
'''
USBCAN1 = DevType(3)
USBCAN2 = DevType(4)
USBCANFD = DevType(6)
'''
    Device Index
'''
DevIndex = c_uint(0)  # 设备索引
'''
    Channel
'''
Channel1 = c_uint(0)  # CAN1
Channel2 = c_uint(1)  # CAN2
'''
    ECAN Status
'''
STATUS_ERR = 0
STATUS_OK = 1

'''
    Device Information
'''


class BoardInfo(Structure):
    _fields_ = [("hw_Version", c_ushort),  # 硬件版本号，用16进制表示
                ("fw_Version", c_ushort),  # 固件版本号，用16进制表示
                ("dr_Version", c_ushort),  # 驱动程序版本号，用16进制表示
                ("in_Version", c_ushort),  # 接口库版本号，用16进制表示
                ("irq_Num", c_ushort),  # 板卡所使用的中断号
                ("can_Num", c_byte),  # 表示有几路CAN通道
                ("str_Serial_Num", c_byte * 20),  # 此板卡的序列号，用ASC码表示
                ("str_hw_Type", c_byte * 40),  # 硬件类型，用ASC码表示
                ("Reserved", c_byte * 4)]  # 系统保留


class CAN_OBJ(Structure):
    _fields_ = [("ID", c_uint),  # 报文帧ID
                ("TimeStamp", c_uint),  # 接收到信息帧时的时间标识，从CAN控制器初始化开始计时，单位微秒
                ("TimeFlag", c_byte),  # 是否使用时间标识，为1时TimeStamp有效，TimeFlag和TimeStamp只在此帧为接收帧时有意义。
                ("SendType", c_byte),
                # 发送帧类型。=0时为正常发送，=1时为单次发送（不自动重发），=2时为自发自收（用于测试CAN卡是否损坏），=3时为单次自发自收（只发送一次，用于自测试），只在此帧为发送帧时有意义
                ("RemoteFlag", c_byte),  # 是否是远程帧。=0时为数据帧，=1时为远程帧
                ("ExternFlag", c_byte),  # 是否是扩展帧。=0时为标准帧（11位帧ID），=1时为扩展帧（29位帧ID）
                ("DataLen", c_byte),  # 数据长度DLC(<=8)，即Data的长度
                ("data", c_ubyte * 8),  # CAN报文的数据。空间受DataLen的约束
                ("Reserved", c_byte * 3)]  # 系统保留。


class INIT_CONFIG(Structure):
    _fields_ = [("acccode", c_uint32),  # 验收码。SJA1000的帧过滤验收码
                ("accmask", c_uint32),  # 屏蔽码。SJA1000的帧过滤屏蔽码。屏蔽码推荐设置为0xFFFF FFFF，即全部接收
                ("reserved", c_uint32),  # 保留
                ("filter", c_byte),  # 滤波使能。0=不使能，1=使能。使能时，请参照SJA1000验收滤波器设置验收码和屏蔽码
                ("timing0", c_byte),  # 波特率定时器0,详见动态库使用说明书7页
                ("timing1", c_byte),  # 波特率定时器1,详见动态库使用说明书7页
                ("mode", c_byte)]  # 模式。=0为正常模式，=1为只听模式，=2为自发自收模式。


#import _ctypes

cwdx = os.getcwd()


class ECAN(object):
    def __init__(self):
        #self.dll = cdll.LoadLibrary(cwdx + '/ECanVci64.dll')
        self.dll = cdll.LoadLibrary( R'D:\Code\Python\all\Gcan\ECanVci64.dll')
        if self.dll == None:
            print("DLL Couldn't be loaded")

    def OpenDevice(self, DeviceType, DeviceIndex):
        try:
            return self.dll.OpenDevice(DeviceType, DeviceIndex, 0)
        except:
            print("Exception on OpenDevice!")
            raise

    def CloseDevice(self, DeviceType, DeviceIndex):
        try:
            return self.dll.CloseDevice(DeviceType, DeviceIndex, 0)
        except:
            print("Exception on CloseDevice!")
            raise

    def InitCan(self, DeviceType, DeviceIndex, CanInd, Initconfig):
        try:
            return self.dll.InitCAN(DeviceType, DeviceIndex, CanInd, byref(Initconfig))
        except:
            print("Exception on InitCan!")
            raise

    def StartCan(self, DeviceType, DeviceIndex, CanInd):
        try:
            return self.dll.StartCAN(DeviceType, DeviceIndex, CanInd)
        except:
            print("Exception on StartCan!")
            raise

    def ReadBoardInfo(self, DeviceType, DeviceIndex):
        try:
            mboardinfo = BoardInfo()
            ret = self.dll.ReadBoardInfo(DeviceType, DeviceIndex, byref(mboardinfo))
            return mboardinfo, ret
        except:
            print("Exception on ReadBoardInfo!")
            raise

    def Receivce(self, DeviceType, DeviceIndex, CanInd, length):
        try:
            recmess = (CAN_OBJ * length)()
            ret = self.dll.Receive(DeviceType, DeviceIndex, CanInd, byref(recmess), length, 0)
            return length, recmess, ret
        except:
            print("Exception on Receive!")
            raise

    def Tramsmit(self, DeviceType, DeviceIndex, CanInd, mcanobj):
        try:
            # mCAN_OBJ=CAN_OBJ*2
            # self.dll.Transmit.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, POINTER(CAN_OBJ),
            # ctypes.c_uint16]
            
            return self.dll.Transmit(DeviceType, DeviceIndex, CanInd, byref(mcanobj), c_uint16(1))
        except:
            print("Exception on Tramsmit!")
            raise


#设置报文参数
baudcan1="1M"
baudcan2="1M"
musbcanopen = False   #是否打开设备
rec_CAN1 = 1         #can1接收报文数
rec_CAN2 = 1
rec_data_Can1 = {'pos':np.zeros(9),
                 'vel':np.zeros(9)
                 }
rec_data_Can1_last = {}
rec_data_Can2 = {}
rec_data_Can2_last = {}

Time=0
time_last=0
time_theta=0
time_theta_max=0
# 加载动态库
ecan = ECAN()

lock = threading.RLock()


'''
读取数据
'''
def ReadCAN():
    global musbcanopen, rec_CAN1, rec_CAN2,rec_data_Can1,rec_data_Can1_last,rec_data_Can2,rec_data_Can2_last,Time,time_last,time_theta,time_theta_max
    if (musbcanopen == True):
        scount = 0
        while (1):
            scount=scount+1
            #lock.acquire()
            
            len, rec, ret = ecan.Receivce(USBCAN2, DevIndex, Channel1, 1)
            if (len > 0 and ret == 1):
                rec_data_Can1_last=rec_data_Can1
                rec_CAN1 = rec_CAN1 + 1
                """
                if rec[0].TimeFlag == 1:
                    time_last=Time
                    Time=rec[0].TimeStamp
                    time_theta=Time-time_last
                    if time_last!=0:
                        time_theta_max=max(time_theta_max,time_theta)
                """
                if rec[0].ExternFlag == 0 & rec[0].RemoteFlag == 0:
                    if rec[0].ID == 1:
                        temp=[]
                        for i in range(0, rec[0].DataLen):
                            temp.append(rec[0].data[i])
                        binary_data = struct.pack('BBBB', *temp[4:8])
                        rec_data_Can1['pos'][temp[3]-1]=struct.unpack('>f', binary_data)[0]*(math.pi*2/9)  #//传过来的数据是以电机的圈数为单位
                    if rec[0].ID == 2:
                        temp=[]
                        for i in range(0, rec[0].DataLen):
                            temp.append(rec[0].data[i])
                        binary_data = struct.pack('BBBB', *temp[4:8])
                        rec_data_Can1['vel'][temp[3]-1]=struct.unpack('>f', binary_data)[0]*(1000*math.pi*2/60/9)    #传过来的电机速度单位 krpm 
                        
            #lock.release()
            '''
            len2, rec2, ret2 = ecan.Receivce(USBCAN2, DevIndex, Channel2, 1)
            if (len2 > 0 and ret2 == 1):
                rec_data_Can1_last=rec_data_Can1
                rec_CAN2 = rec_CAN2 + 1
                if rec2[0].ExternFlag == 0 & rec2[0].RemoteFlag == 0:
                    if rec2[0].ID == 1:
                        temp=[]
                        for i in range(0, rec2[0].DataLen):
                            temp.append(rec2[0].data[i])
                        binary_data = struct.pack('BBBB', *temp[4:8])
                        rec_data_Can1['pos'][temp[3]-1]=struct.unpack('>f', binary_data)[0]
                    if rec2[0].ID == 2:
                        temp=[]
                        for i in range(0, rec2[0].DataLen):
                            temp.append(rec2[0].data[i])
                        binary_data = struct.pack('BBBB', *temp[4:8])
                        rec_data_Can1['vel'][temp[3]-1]=struct.unpack('>f', binary_data)[0]
            '''

            '''
            #can2接收
            len2, rec2, ret2 = ecan.Receivce(USBCAN2, DevIndex, Channel2, 1)
            if (len2 > 0 and ret2 == 1):
                mstr = "Rec: "
                rec_data_Can2_last=rec_data_Can2
                rec_CAN2 = rec_CAN2 + 1
                if rec2[0].TimeFlag == 0:
                    pass
                if rec2[0].ExternFlag == 0:
                    rec_data_Can2[hex(rec2[0].ID).zfill(3)]=[]

                if rec2[0].RemoteFlag == 0:
                    for i in range(0, rec2[0].DataLen):
                        rec_data_Can2[hex(rec2[0].ID).zfill(3)].append(hex(rec2[0].data[i]).zfill(2))
                '''   
        #Time=time.time()
        #print(Time-time_last)
        #time_last=Time
        #print(rec_data_Can1["pos"][0],rec_data_Can1["vel"][0],rec_data_Can1["pos"][1],rec_data_Can1["vel"][1] ) 
            np.set_printoptions(precision=3, suppress=True)

            #print(rec_data_Can1["pos"])
            tcp_server.trifinger_state.dof_pos[0]=rec_data_Can1["pos"]+tcp_server.trifinger_state.dof_pos_offset
            tcp_server.trifinger_state.dof_vel[0]=rec_data_Can1["vel"]
        #t = threading.Timer(0.03, ReadCAN)
        #t.setDaemon(True)
        #t.start()


def ReadCAN2():
    global musbcanopen, rec_CAN1, rec_CAN2,rec_data_Can1,rec_data_Can1_last,rec_data_Can2,rec_data_Can2_last,Time,time_last,time_theta,time_theta_max
    if (musbcanopen == True):
        while (1):
            len2, rec2, ret2 = ecan.Receivce(USBCAN2, DevIndex, Channel2, 1)
            if (len2 > 0 and ret2 == 1):
                rec_data_Can1_last=rec_data_Can1
                rec_CAN2 = rec_CAN2 + 1
                if rec2[0].ExternFlag == 0 & rec2[0].RemoteFlag == 0:
                    if rec2[0].ID == 1:
                        temp=[]
                        for i in range(0, rec2[0].DataLen):
                            temp.append(rec2[0].data[i])
                        binary_data = struct.pack('BBBB', *temp[4:8])
                        rec_data_Can1['pos'][temp[3]-1]=struct.unpack('>f', binary_data)[0]*(math.pi*2/9)
                    if rec2[0].ID == 2:
                        temp=[]
                        for i in range(0, rec2[0].DataLen):
                            temp.append(rec2[0].data[i])
                        binary_data = struct.pack('BBBB', *temp[4:8])
                        rec_data_Can1['vel'][temp[3]-1]=struct.unpack('>f', binary_data)[0]*(1000*math.pi*2/60/9)

# python调用动态库默认参数为整型


def caninit():
    global musbcanopen, t, rec_CAN1, rec_CAN2
    if (musbcanopen == False):
        initconfig = INIT_CONFIG()
        initconfig.acccode = 0  # 设置验收码
        initconfig.accmask = 0xFFFFFFFF  # 设置屏蔽码
        initconfig.filter = 0  # 设置滤波使能
        mbaudcan1 = baudcan1
        mbaudcan2 = baudcan2
        # 打开设备
        if (ecan.OpenDevice(USBCAN2, DevIndex) != STATUS_OK):
            print("ERROR", "OpenDevice Failed!")
            return
        initconfig.timing0, initconfig.timing1 = getTiming(mbaudcan1)
        initconfig.mode = 0
        # 初始化CAN1
        if (ecan.InitCan(USBCAN2, DevIndex, Channel1, initconfig) != STATUS_OK):
            print("ERROR", "InitCan CAN1 Failed!")
            ecan.CloseDevice(USBCAN2, DevIndex)
            return
        # 初始化CAN2
        initconfig.timing0, initconfig.timing1 = getTiming(mbaudcan2)
        if (ecan.InitCan(USBCAN2, DevIndex, Channel2, initconfig) != STATUS_OK):
            print("ERROR", "InitCan CAN2 Failed!")
            ecan.CloseDevice(USBCAN2, DevIndex)
            return
        if (ecan.StartCan(USBCAN2, DevIndex, Channel1) != STATUS_OK):
            print("ERROR", "StartCan CAN1 Failed!")
            ecan.CloseDevice(USBCAN2, DevIndex)
            return
        if (ecan.StartCan(USBCAN2, DevIndex, Channel2) != STATUS_OK):
            print("ERROR", "StartCan CAN2 Failed!")
            ecan.CloseDevice(USBCAN2, DevIndex)
            return
        musbcanopen = True
        rec_CAN1 = 1
        rec_CAN2 = 1
        print("CAN init succes")
        tcp_server.trifinger_state.caninit=True
        t = threading.Thread(target=ReadCAN)
        t.setDaemon(True)
        t.start()
        t2 = threading.Thread(target=ReadCAN2)
        t2.setDaemon(True)
        t2.start()
    else:
        musbcanopen = False
        ecan.CloseDevice(USBCAN2, DevIndex)
        print("CAN close succes")

def getTiming(mbaud):
    if mbaud == "1M":
        return 0, 0x14
    if mbaud == "800k":
        return 0, 0x16
    if mbaud == "666k":
        return 0x80, 0xb6
    if mbaud == "500k":
        return 0, 0x1c
    if mbaud == "400k":
        return 0x80, 0xfa
    if mbaud == "250k":
        return 0x01, 0x1c
    if mbaud == "200k":
        return 0x81, 0xfa
    if mbaud == "125k":
        return 0x03, 0x1c
    if mbaud == "100k":
        return 0x04, 0x1c
    if mbaud == "80k":
        return 0x83, 0xff
    if mbaud == "50k":
        return 0x09, 0x1c


'''
读取SN号码
'''


def readmess():
    global musbcanopen
    if (musbcanopen == False):
        print("ERROR", "请先打开设备")
    else:
        mboardinfo, ret = ecan.ReadBoardInfo(USBCAN2, DevIndex)  # 读取设备信息需要在打开设备后执行
        if ret == STATUS_OK:
            mstr = ""
            for i in range(0, 10):
                mstr = mstr + chr(mboardinfo.str_Serial_Num[i])  # 结构体中str_Serial_Num内部存放存放SN号的ASC码
            

def sendcan1(Id:int,motor_id:int,torque_data:float):  #motor_id 0~255
    global musbcanopen
    if (musbcanopen == False):
        print("ERROR", "请先打开设备")
    else:
        canobj = CAN_OBJ()
        canobj.ID = Id
        canobj.DataLen = 8
        canobj.data[0] = 0
        canobj.data[1] = 0
        canobj.data[2] = 0
        canobj.data[3] = motor_id
        Bytes=list(struct.pack('>f',torque_data))
        canobj.data[4] = Bytes[0]
        canobj.data[5] = Bytes[1]
        canobj.data[6] = Bytes[2]
        canobj.data[7] = Bytes[3]
        canobj.RemoteFlag = 0
        canobj.ExternFlag = 0
        ecan.Tramsmit(USBCAN2, DevIndex, Channel1, canobj)

def sendcan2(Id:int,motor_id:int,torque_data:float):
    global musbcanopen
    if (musbcanopen == False):
        print("ERROR", "请先打开设备")
    else:
        canobj = CAN_OBJ()
        canobj.ID = Id
        canobj.DataLen = 8
        canobj.data[0] = 0
        canobj.data[1] = 0
        canobj.data[2] = 0
        canobj.data[3] = motor_id
        Bytes=list(struct.pack('>f',torque_data))
        canobj.data[4] = Bytes[0]
        canobj.data[5] = Bytes[1]
        canobj.data[6] = Bytes[2]
        canobj.data[7] = Bytes[3]
        canobj.RemoteFlag = 0
        canobj.ExternFlag = 0
        ecan.Tramsmit(USBCAN2, DevIndex, Channel2, canobj)



if __name__=='__main__':
    try:

        caninit()
        print(233)
        a=1
        while(1):
            #print(rec_data_Can1["pos"][6:9])
            print("-----------------------")
            print(tcp_server.trifinger_state.dof_vel[0])
            print(rec_data_Can1["pos"])
            time.sleep(0.1)
            #print(Time,time_last,time_theta,time_theta_max)
            

    except Exception as e:
        if e.__class__ ==KeyboardInterrupt:
            caninit()
            print(rec_data_Can1)
            raise KeyboardInterrupt
    