"""
The main console is used to manage different processes or thread. (There is only one instantiation)
Subprocesses include:
- Active
  - TrifingerRobot[Pos|Vel]Env -> action -> CANManager -> wait latest obs
- Passive
  - CANManger -> receive action from Env -> send to CAN bus -> check available -------------------------> return latest obs
                                                                      |- not -->   initialize the error arm  -->-|
"""
import sys
from pathlib import Path
PATH_ROOT = Path(__file__).parents[1]
sys.path.append(str(PATH_ROOT))

from trifinger_robot_control.can_manager import CANManager
from trifinger_robot_control.log_util import get_logger
logger = get_logger(__name__)

from multiprocessing import Process, Pipe

class MainConsole:
  def __init__(self):
    self.can_manager_pipe, child_pipe = Pipe()
    self.can_manager_process = Process(target=CANManager, args=(child_pipe,))
    self.can_manager_process.start()
  
  def execute(self, cmd, *args):
    if cmd == 'step_pos':  # args[0]=rad_target
      self.can_manager_pipe.send((cmd, *args))
    elif cmd == 'get_robot_obs':
      self.can_manager_pipe.send(('get_obs',))
      name, *args = self.can_manager_pipe.recv()
      assert name == 'obs'
      return args[0]
    else:
      logger.error(
        f"Unable to resolve {cmd=}, available"
        "[step_pos|get_robot_obs]")
      raise ValueError
