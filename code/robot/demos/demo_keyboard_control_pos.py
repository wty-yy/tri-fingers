"""
A demo uses keyboard to control the robot
"""
from pathlib import Path
import sys
PATH_ROOT = Path(__file__).parents[1]
sys.path.append(str(PATH_ROOT))

from trifinger_robot_control.trifinger_robot_env import TrifingerRobotEnv
import trifinger_robot_control.constants as const
from trifinger_robot_control.log_util import get_logger
logger = get_logger(__name__)

import keyboard
import numpy as np
import time

class Control:
  def __init__(self, motor_id=9):
    self.motor_id = motor_id
    self.init_target = np.array([0.0,0.9,-1.7]*3, np.float64)  # 目标角度为90度
    self.target = self.init_target.copy()
    self.degree = 10
    self.send_single = False
    self.avg_time_used, self.total_send = 0, 0
    self.env = TrifingerRobotEnv('pos')
  
  @property
  def delta(self):
    return np.pi / 180 * self.degree
  
  def send(self):
    self.target = np.clip(self.target, const.dof_pos_low, const.dof_pos_high)
    start_send = time.time()
    self.env.step(self.target)
    time_used = time.time() - start_send
    self.total_send += 1
    self.avg_time_used += (time_used - self.avg_time_used) / self.total_send
    logger.info(f"change target to {self.target}, time used={time_used:.6f}, avg time used={self.avg_time_used:.6f}, total send={self.total_send}")
  
  def execute(self, cmd):
    if cmd == 'c':
      self.motor_id = int(input("Change Motor id="))
      print(f"Change motor id={self.motor_id}")
    elif cmd == 'k':
      print("Start update (push up)")
      self.target[self.motor_id-1] += self.delta
      self.send()
    elif cmd == 'j':
      print("Start update (push down)")
      self.target[self.motor_id-1] -= self.delta
      self.send()
    elif cmd == 'r':
      print("Random change target (push up)")
      rand = np.random.randint(0, 11, 9).astype(np.float32) / 180 * np.pi
      self.target += rand
      self.send()
    elif cmd == 'e':
      print("Random change target (push down)")
      rand = np.random.randint(0, 11, 9).astype(np.float32) / 180 * np.pi
      self.target -= rand
      self.send()
    elif cmd == 'i':
      print("Reset to initial target")
      self.target = self.init_target.copy()
      self.send()
    elif cmd == 'h':
      print("Reset to high")
      self.target = const.dof_pos_high
      self.send()

  def play(self):
    while 1:
      while 1:
        flag = False
        for key in ['c', 'k', 'j', 'r', 'e', 'i', 'h']:
          if keyboard.is_pressed(key):
            self.execute(key)
            flag = True
            break
        if flag: break
      print("press space to continue...", end="", flush=True)
      keyboard.wait('space')
      print(f"wait for operation: (current motor id={self.motor_id})")
  
  def auto_test(self):
    while 1:
      self.execute('i')
      for _ in range(5):
        self.execute('e')
        self.execute('e')
        self.execute('r')
        self.execute('r')
        self.execute('r')
      for _ in range(25):
        self.execute('r')

if __name__ == '__main__':
  control = Control()
  # control.play()
  control.auto_test()

