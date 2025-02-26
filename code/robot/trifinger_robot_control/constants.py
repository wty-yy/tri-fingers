import math
import numpy as np
from pathlib import Path
import time

path_root = Path(__file__).parents[1]
path_gcan_dll = path_root / "Gcan/ECanVci64.dll"  # ECanVci64.dll
path_log_parent_dir = path_root / "logs"
path_log_parent_dir.mkdir(exist_ok=True)
# path_log_dir = path_log_parent_dir / f"{len(list(path_log_parent_dir.glob('*'))):04}_{time.strftime(r'%Y%m%d_%H%M%S')}"
path_log_dir = path_log_parent_dir / f"{time.strftime(r'%Y%m%d_%H%M')}"
path_log_dir.mkdir(exist_ok=True)
path_log_message = path_log_dir / "logging.log"

# dof rad position
dof_pos_low = np.array([-0.50, 0.3, -2.7]*3, dtype=np.float32)
dof_pos_high = np.array([0.50, 1.57, -0.0]*3, dtype=np.float32)
dof_pos_default = np.array([0.0, 0.9, -1.7]*3, dtype=np.float32)
dof_pos_offset = np.array([0, math.pi/2, 0]*3, dtype=np.float32)

# CAN Manger
max_degree_error = 5
max_rad_error = max_degree_error / 180 * math.pi
max_wait_time = 3.5  # (sec) maximum time to reach the target position 

# DEBUG
# motor_control_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1]  # motor control mask
motor_control_mask = [0, 0, 0, 0, 0, 0, 0, 1, 1]
motor_control_mask = np.array(motor_control_mask, np.bool_)
assert len(motor_control_mask) == 9
motor_id_map = {
  # 7: 8,
  # 8: 7,
  8: 8,
  9: 7,
}
motor_id_inverse_map = {value: key for key, value in motor_id_map.items()}

