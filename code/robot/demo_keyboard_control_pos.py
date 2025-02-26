"""
轮询can通讯速度: 
- 1个: avg time used=0.000955, total send=137
- 9个: avg time used=0.009810, total send=300
中断can通讯速度: 
- 1个: avg time used=0.000978, total send=134
- 9个: avg time used=0.009913, total send=233
"""
import keyboard
import numpy as np
import Gcan
import tcp_server
import time

class Control:
  def __init__(self, motor_id):
    self.motor_id = motor_id
    self.init_target = np.array([0.0,0.9,-1.7]*3, np.float64)  # 目标角度为90度
    self.target = self.init_target.copy()
    self.degree = 10
    self.send_single = False
    self.avg_time_used, self.total_send = 0, 0
    Gcan.caninit()
  
  @property
  def delta(self):
    return np.pi / 180 * self.degree
  
  def send(self):
    start_send = time.time()
    if self.send_single:
      action = tcp_server.trifinger_state.rad_action2real_action(self.target)
      Gcan.sendcan2(0,self.motor_id,action[self.motor_id-1])
    else:
      tcp_server.trifinger_state.motor_control(self.target)
    time_used = time.time() - start_send
    self.total_send += 1
    self.avg_time_used += (time_used - self.avg_time_used) / self.total_send
    print(f"change target to {self.target}, time used={time_used:.6f}, avg time used={self.avg_time_used:.6f}, total send={self.total_send}")

  def run(self):
    while 1:
      while 1:
        # print("wait for operator...")
        if keyboard.is_pressed('c'):
          self.motor_id = int(input("Change Motor id="))
          print(f"Change motor id={self.motor_id}")
          break
        if keyboard.is_pressed('k'):
          print("Start update (push up)")
          self.target[self.motor_id-1] += self.delta
          self.send()
          break
        if keyboard.is_pressed('j'):
          print("Start update (push down)")
          self.target[self.motor_id-1] -= self.delta
          self.send()
          break
        if keyboard.is_pressed('r'):
          print("Random change target (push up)")
          rand = np.random.randint(0, 11, 9).astype(np.float32) / 180 * np.pi
          self.target += rand
          self.target = np.clip(self.target, tcp_server.trifinger_state.dof_pos_low, tcp_server.trifinger_state.dof_pos_high)
          self.send()
          break
        if keyboard.is_pressed('e'):
          print("Random change target (push down)")
          rand = np.random.randint(0, 11, 9).astype(np.float32) / 180 * np.pi
          self.target -= rand
          self.target = np.clip(self.target, tcp_server.trifinger_state.dof_pos_low, tcp_server.trifinger_state.dof_pos_high)
          self.send()
          break
        if keyboard.is_pressed('i'):
          print("Reset to initial target")
          self.target = self.init_target.copy()
          self.send()
          break
        if keyboard.is_pressed('h'):
          print("Reset to high")
          self.target = tcp_server.trifinger_state.dof_pos_high
          self.send()
          break
      print("press space to continue...", end="", flush=True)
      keyboard.wait('space')
      print(f"wait for operation: (current motor id={self.motor_id})")

if __name__ == '__main__':
  control = Control(9)
  control.run()
