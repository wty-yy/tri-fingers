import sys
from pathlib import Path
PATH_ROOT = Path(__file__).parents[1]
sys.path.append(str(PATH_ROOT))

from trifinger_robot_control.gcan import GCAN
import trifinger_robot_control.constants as const
from trifinger_robot_control.log_util import get_logger
logger = get_logger(__name__)

import numpy as np
from multiprocessing.connection import PipeConnection
from threading import Thread
import time


class CANManager:
  """ Running as multiprocessing.Process
  Receive cmd from main console, send message through GCAN,
  and check robot has reached target position.
  """
  def __init__(self, cmd_pipe: PipeConnection):
    self.cmd_pipe = cmd_pipe
    self.gcan = GCAN()
    self.run()
  
  def run(self):
    """Process main loop"""
    while 1:
      name, *args = self.cmd_pipe.recv()
      if name == "step_pos":
        rad_target = args[0]
        assert len(rad_target) == 9, f"The action length must be 9, but get {len(rad_target)}"
        # Step
        self.execute_rad_target(rad_target)
        step_start_time = start_time = time.time()
        # Check
        max_error = lambda: np.abs(self.get_pos()-rad_target)[const.motor_control_mask].max()
        last_info_time = time.time()
        fine_tune_count = 0
        while max_error() > const.max_rad_error:
          if time.time() - last_info_time > const.max_wait_time / 5:
            last_info_time = time.time()
            fine_tune_count += 1
            logger.info(
              f"max rad error={max_error():.4f} > {const.max_rad_error=:.4f}, "
              f"fine-tuning times {fine_tune_count}...")
            self.execute_rad_target(rad_target)
          time_used = time.time() - start_time
          if time_used > const.max_wait_time:
            current_pos = self.get_pos()
            error_motor_ids = np.argwhere(
              (np.abs(current_pos - rad_target) > const.max_rad_error)
              * const.motor_control_mask
            ).astype(np.int32).reshape(-1)
            reset_arm_ids = np.unique(error_motor_ids // 3)
            for arm_id in reset_arm_ids:
              st = arm_id*3; end=(arm_id+1)*3
              rad_target[st:end] = const.dof_pos_default[st:end]
            self.execute_rad_target(rad_target)
            start_time = time.time()
            logger.warning(
              "Position change TIME OUT! "
              f"time used={time_used:.4f}s > max wait time={const.max_wait_time:.4f}s, "
              f"error motor ids (start from 1)={error_motor_ids+1}, "
              f"reset arm ids (start from 1)={reset_arm_ids+1}, "
              f"current pos={list(current_pos)}, target pos={rad_target}")
        logger.info(f"Complete position action! step time used={time.time()-step_start_time:.4f}s")

      if name in ["get_obs"]:
        cmd = ("obs", self.get_obs().copy())
        self.cmd_pipe.send(cmd)
  
  def execute_rad_target(self, rad_target):
    action = self.rad_pos2relative_pos(rad_target)
    threads: list[Thread] = []
    start_time = time.time()
    for i in range(len(action)):
      if not const.motor_control_mask[i]: continue
      threads.append(Thread(target=self.gcan.send_can, args=(i+1, action[i])))
      threads[-1].start()
    for t in threads:
      t.join()
    time_used = time.time() - start_time
    logger.info(f"Send 9 can message time used: {time_used:.6f}s")
    

  def get_obs(self) -> np.ndarray:
    return np.r_[self.get_pos(), self.gcan.rec_data['vel']]
  
  def get_pos(self) -> np.ndarray:
    return self.gcan.rec_data['pos'] + const.dof_pos_offset

  def rad_pos2relative_pos(self, rad_action):
    """convert rad position to relative position"""
    action = np.clip(rad_action, const.dof_pos_low, const.dof_pos_high)
    return np.multiply(np.subtract(action, self.get_pos()), 9/(2*np.pi))  # 电机转九圈关节转一圈

if __name__ == '__main__':
  cmd, *args = ('step', (1,2,3))
  print(cmd, args, args[0])
  # print(np.r_[1,3,[45,5]])
