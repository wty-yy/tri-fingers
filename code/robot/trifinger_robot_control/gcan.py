import sys
from pathlib import Path
PATH_ROOT = Path(__file__).parents[1]
sys.path.append(str(PATH_ROOT))

import trifinger_robot_control.constants as const
from trifinger_robot_control.log_util import get_logger
logger = get_logger(__name__)

import threading
from ctypes import *
import struct
import numpy as np
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

class ECAN(object):
    def __init__(self):
        path_dll = str(const.path_gcan_dll)
        self.dll = cdll.LoadLibrary(path_dll)
        if self.dll == None:
            logger.error("DLL Couldn't be loaded")

    def OpenDevice(self, DeviceType, DeviceIndex):
        try:
            return self.dll.OpenDevice(DeviceType, DeviceIndex, 0)
        except:
            logger.exception("Exception on OpenDevice!")
            raise

    def CloseDevice(self, DeviceType, DeviceIndex):
        try:
            return self.dll.CloseDevice(DeviceType, DeviceIndex, 0)
        except:
            logger.exception("Exception on CloseDevice!")
            raise

    def InitCan(self, DeviceType, DeviceIndex, CanInd, Initconfig):
        try:
            return self.dll.InitCAN(DeviceType, DeviceIndex, CanInd, byref(Initconfig))
        except:
            logger.exception("Exception on InitCan!")
            raise

    def StartCan(self, DeviceType, DeviceIndex, CanInd):
        try:
            return self.dll.StartCAN(DeviceType, DeviceIndex, CanInd)
        except:
            logger.exception("Exception on StartCan!")
            raise

    def ReadBoardInfo(self, DeviceType, DeviceIndex):
        try:
            mboardinfo = BoardInfo()
            ret = self.dll.ReadBoardInfo(DeviceType, DeviceIndex, byref(mboardinfo))
            return mboardinfo, ret
        except:
            logger.exception("Exception on ReadBoardInfo!")
            raise

    def Receivce(self, DeviceType, DeviceIndex, CanInd, length):
        try:
            recmess = (CAN_OBJ * length)()
            ret = self.dll.Receive(DeviceType, DeviceIndex, CanInd, byref(recmess), length, 0)
            return length, recmess, ret
        except:
            logger.exception("Exception on Receive!")
            raise

    def Tramsmit(self, DeviceType, DeviceIndex, CanInd, mcanobj):
        try:
            return self.dll.Transmit(DeviceType, DeviceIndex, CanInd, byref(mcanobj), c_uint16(1))
        except:
            logger.exception("Exception on Tramsmit!")
            raise

class GCAN:
  """Send message and receive latest robot position or velocity"""
  baudcan1 = "1M"  # setup message parameter
  baudcan2 = "1M"
  musbcanopen = False  # whether CAN is opened
  rec_data = {'pos': np.zeros(9), 'vel': np.zeros(9)}  # receive message decode

  def __init__(self):
    self.ecan = ECAN()
    self.caninit()

  def read_can_loop(self, can_idx: c_uint):
    """Update data from can[1|2] in thread"""
    assert can_idx in [Channel1, Channel2]
    if self.musbcanopen == True:
      while True:
        len, rec, ret = self.ecan.Receivce(USBCAN2, DevIndex, can_idx, 1)
        if len > 0 and ret == 1:
          if rec[0].ExternFlag == 0 and rec[0].RemoteFlag == 0:
            temp = [rec[0].data[i] for i in range(rec[0].DataLen)]  # low=0|0|0|id, high=data (float32)
            if temp[3] in const.motor_id_inverse_map:  # DEBUG
              temp[3] = const.motor_id_inverse_map[temp[3]]
            motor_id = temp[3] - 1
            binary_data = struct.pack('BBBB', *temp[4:8])  # high data
            if rec[0].ID == 1:
              self.rec_data['pos'][motor_id] = struct.unpack('>f', binary_data)[0] * (math.pi*2/9)  # motor turns (9 motor turns = 1 arm turns)
            if rec[0].ID == 2:
              self.rec_data['vel'][motor_id] = struct.unpack('>f', binary_data)[0] * (1000*math.pi*2/60/9)  # motor velocity (krpm)

  def caninit(self):
    if self.musbcanopen == False:
      initconfig = INIT_CONFIG()
      initconfig.acccode = 0  # 设置验收码
      initconfig.accmask = 0xFFFFFFFF  # 设置屏蔽码
      initconfig.filter = 0  # 设置滤波使能
      mbaudcan1 = self.baudcan1
      mbaudcan2 = self.baudcan2
      # open device
      if self.ecan.OpenDevice(USBCAN2, DevIndex) != STATUS_OK:
        logger.error("ERROR", "OpenDevice Failed! Make sure GCan APP is closed!")
        return
      initconfig.timing0, initconfig.timing1 = self.getTiming(mbaudcan1)
      initconfig.mode = 0
      # init CAN1
      if self.ecan.InitCan(USBCAN2, DevIndex, Channel1, initconfig) != STATUS_OK:
        logger.error("ERROR", "InitCan CAN1 Failed!")
        self.ecan.CloseDevice(USBCAN2, DevIndex)
        return
      if self.ecan.StartCan(USBCAN2, DevIndex, Channel1) != STATUS_OK:
        logger.error("ERROR", "StartCan CAN1 Failed!")
        self.ecan.CloseDevice(USBCAN2, DevIndex)
        return
      # init CAN2
      initconfig.timing0, initconfig.timing1 = self.getTiming(mbaudcan2)
      if self.ecan.InitCan(USBCAN2, DevIndex, Channel2, initconfig) != STATUS_OK:
        logger.error("ERROR", "InitCan CAN2 Failed!")
        self.ecan.CloseDevice(USBCAN2, DevIndex)
        return
      if self.ecan.StartCan(USBCAN2, DevIndex, Channel2) != STATUS_OK:
        logger.error("ERROR", "StartCan CAN2 Failed!")
        self.ecan.CloseDevice(USBCAN2, DevIndex)
        return
      self.musbcanopen = True
      logger.info("CAN init succes")
      self.t1 = threading.Thread(target=self.read_can_loop, args=(Channel1,))
      self.t1.setDaemon(True)
      self.t1.start()
      self.t2 = threading.Thread(target=self.read_can_loop, args=(Channel2,))
      self.t2.setDaemon(True)
      self.t2.start()
    else:
      self.musbcanopen = False
      self.ecan.CloseDevice(USBCAN2, DevIndex)
      logger.info("CAN close succes")

  def getTiming(self, mbaud):
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

  def send_can(self, motor_id: int, data: float):
    """ Send CAN message, auto select the CAN id to match the motor id

    | can_id | motor_id  |
    |--------|-----------|
    |    1   | 1,2,3,4,5 |
    |    2   | 6,7,8,9   |

    Args:
      motor_id: [int] The id number of the motor, range in 1~9
      data: [float] The position or torque data, send to CAN bus
    """
    assert motor_id in list(range(1, 10))
    can_id = 1 if motor_id in [1,2,3,4,5] else 2
    if (self.musbcanopen == False):
      logger.error("[ERROR] Please open device first!")
      return
    if motor_id in const.motor_id_map:  # DEBUG
      motor_id = const.motor_id_map[motor_id]
    canobj = CAN_OBJ()
    canobj.ID = 0  # message id always be 0, since MBOX16 on DSP monitor message id 0
    canobj.DataLen = 8
    canobj.data[0] = 0
    canobj.data[1] = 0
    canobj.data[2] = 0
    canobj.data[3] = motor_id
    Bytes=list(struct.pack('>f',data))
    canobj.data[4] = Bytes[0]
    canobj.data[5] = Bytes[1]
    canobj.data[6] = Bytes[2]
    canobj.data[7] = Bytes[3]
    canobj.RemoteFlag = 0
    canobj.ExternFlag = 0
    self.ecan.Tramsmit(USBCAN2, DevIndex, Channel1 if can_id == 1 else Channel2, canobj)

if __name__ == '__main__':
   gcan = GCAN()