import sys
from pathlib import Path
PATH_ROOT = Path(__file__).parents[1]
sys.path.append(str(PATH_ROOT))

import trifinger_robot_control.constants as const

import sys
import logging
from termcolor import colored

# logging.basicConfig(
#   filename=const.path_log_message, level=logging.DEBUG,
#   format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
#   datefmt="%Y-%m-%d %H:%M:%S",
# )

log_file_handler = logging.FileHandler(const.path_log_message)
log_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

class TermColorFormatter(logging.Formatter):
  COLORS = {
    'DEBUG': 'white',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bold_red',
  }

  def format(self, record):
    log_message = super().format(record)
    color = self.COLORS.get(record.levelname, 'white')
    return colored(log_message, color)

log_console_handler = logging.StreamHandler(sys.stdout)
log_console_handler.setFormatter(TermColorFormatter(
    "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

def get_logger(name: str):
  logger = logging.getLogger(name)
  logger.setLevel(logging.DEBUG)
  logger.addHandler(log_file_handler)
  logger.addHandler(log_console_handler)
  return logger

if __name__ == '__main__':
  logger = get_logger(__name__)
  logger.debug("Hi")
  logger.info("Hi")
  logger.warning("not good")
  logger.error("GG")
  const.test_log()
