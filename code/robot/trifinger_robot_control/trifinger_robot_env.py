import sys
from pathlib import Path
PATH_ROOT = Path(__file__).parents[1]
sys.path.append(str(PATH_ROOT))

import trifinger_robot_control.constants as const
from trifinger_robot_control.main_console import MainConsole

import numpy as np
import gymnasium as gym

def normalize_min_max(x: np.ndarray, low, high):
  """[low,high]->[-1,1]"""
  mid = (high + low) / 2
  radius = (high - low) / 2
  return np.clip((x - mid) / radius, -1, 1)

class TrifingerRobotEnv(gym.Env):
  """Trifinger Robot Control Environment"""
  def __init__(self, control_type='pos'):
    assert control_type in ['pos', 'torque']
    self.control_type = control_type
    # obs: pos + vel (normalization)
    self.observation_space = gym.spaces.Box(-1, 1, (18,), np.float32)
    self.action_space = gym.spaces.Box(const.dof_pos_low, const.dof_pos_high, (9,), np.float32)
    self.main_console = MainConsole()
  
  def reset(self, seed=None, options=None):
    self.step(const.dof_pos_default)
    obs = self.get_obs()
    info = {}
    return obs, info
  
  def step(self, action):
    if self.control_type == 'pos':
      self.main_console.execute('step_pos', action)
    obs = self.get_obs()
    reward = 0
    terminal = truncated = False
    info = {}
    return obs, 0, terminal, truncated, info
  
  def get_obs(self):
    """Position action"""
    if self.control_type == 'pos':
      obs = self.main_console.execute('get_robot_obs')
      obs[:9] = normalize_min_max(obs[:9], const.dof_pos_low, const.dof_pos_high)
      obs[9:] = 0  # vel is 0
    return obs

if __name__ == '__main__':
  robot_env = TrifingerRobotEnv()
  robot_env.step(const.dof_pos_default)
