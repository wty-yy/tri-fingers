import hydra
import os
import logging
from omegaconf import DictConfig, OmegaConf, open_dict
from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Dict, List, Any
from hydra.core.config_store import ConfigStore


import torch
import numpy as np
# python
import os
import argparse
import yaml
from datetime import datetime
from copy import deepcopy
from rl_games.algos_torch import model_builder

import tcp_server
from tcp_server import trifinger_state
import Gcan
import threading
from scipy.spatial import distance
import keyboard
import time
@dataclass
class SimConfig:
    """Configuration for the IsaacGym simulator."""
    dt: float = 0.02
    substeps: int =  4
    up_axis: str = "z"
    use_gpu_pipeline: bool = MISSING
    num_client_threads: int = 0
    gravity: List[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    physx: Dict[str, Any] = field(default_factory = lambda: {
        "num_threads": 4,
        "solver_type": 1,
        "use_gpu": False, # set to False to run on CPU
        "num_position_iterations": 8,
        "num_velocity_iterations": 0,
        "contact_offset": 0.002,
        "rest_offset": 0.0,
        "bounce_threshold_velocity": 0.5,
        "max_depenetration_velocity": 1000.0,
        "default_buffer_size_multiplier": 5.0,
    })
    flex: Dict[str, Any] = field(default_factory=lambda: {
        "num_outer_iterations": 5,
        "num_inner_iterations": 20,
        "warm_start": 0.8,
        "relaxation": 0.75,
    })

@dataclass
class EnvConfig:
    """Configuration for all instances of `EnvBase`."""
    env_name: str = MISSING
    # general env settings
    num_instances: int = MISSING
    seed: int = MISSING
    spacing: float = 1.0
    aggregate_mode: bool = True
    # command settings
    control_decimation: int = 1
    # physics settings
    physics_engine: str = MISSING
    sim: SimConfig = SimConfig()

@dataclass
class Trifinger(EnvConfig):
    """Configuration for all instances of `TrifingerEnv`."""
    env_name: str = "Trifinger"
    episode_length: int = 750
    task_difficulty: int = MISSING
    enable_ft_sensors: bool = False
    # observation settings
    asymmetric_obs: bool = False
    normalize_obs: bool = True
    # reset distribution settings
    apply_safety_damping: bool = True
    command_mode: str = "torque"
    normalize_action: bool = True
    reset_distribution:  Dict[str, Any] = field(default_factory=lambda:{
      "object_initial_state": {
        "type": "random"
      },
      "robot_initial_state": {
        "dof_pos_stddev": 0.4,
        "dof_vel_stddev": 0.2,
        "type": "default"
      }
    })
    # reward settings
    reward_terms:Dict[str, Any] = field(default_factory=lambda: {
      "finger_move_penalty":{
        "activate": True,
        "weight": -0.05,#-0.1,
      },
      "finger_reach_object_rate": {
        "activate": True,
        "norm_p": 2,
        "weight": -1750,
        #"thresh_sched_start":0,
        #"thresh_sched_end":5e3,
      },
      "object_dist": {
        "activate": True,
        "weight": 5000, #2000,
      },
      "object_rot": {
        "activate": False,#False
        "weight": 300
      },
      "object_rot_delta": {
        "activate": False,
        "weight": -250
      },
      "object_move": { 
        "activate": False,
        "weight": -750,
      }
    })
    # # termination settings
    termination_conditions: Dict[str, Any] = field(default_factory=lambda: {
      "success": {
        "activate": True, #False,
        "bonus": 5000.0,
        "orientation_tolerance": 0.1, #rad
        "position_tolerance": 0.01,  #m
      }
    })

@dataclass
class TrifingerDifficulty1(Trifinger):
    """Trifinger Difficulty 1 Configuration."""
    task_difficulty = 1

@dataclass
class TrifingerDifficulty2(Trifinger):
    """Trifinger Difficulty 2 Configuration."""
    task_difficulty = 2

@dataclass
class TrifingerDifficulty3(Trifinger):
    """Trifinger Difficulty 3 Configuration."""
    task_difficulty = 3

@dataclass
class TrifingerDifficulty4(Trifinger):
    """Mode for testing to try to get the rotation reward up and running."""
    task_difficulty = 4
    episode_length = 750
    reward_terms:Dict[str, Any] = field(default_factory=lambda: {
        "finger_move_penalty":{
            "activate": True,
            "weight": -0.1,
        },
        "finger_reach_object_rate": {
            "activate": True,
            "norm_p": 2,
            "weight": -250,
            "thresh_sched_start": 0,
            "thresh_sched_end": 1e7,
        },
        "object_dist": {
            "activate": True,
            "weight": 2000,
            "thresh_sched_start": 0,
            "thresh_sched_end": 10e10,
        },
        "object_rot": {
            "activate": True,
            "weight": 2000,
            "epsilon": 0.01,
            "scale": 3.0,
            "thresh_sched_start": 1e7,
            "thresh_sched_end": 1e10, # dont end!
        },
        "object_rot_delta": {
            "activate": False,
            "weight": -250
        },
        "object_move": {
            "activate": False,
            "weight": -750,
        }
    })
    termination_conditions: Dict[str, Any] = field(default_factory=lambda: {
        "success": {
            "activate": False,
            "bonus": 5000.0,
            "orientation_tolerance": 0.25,
            "position_tolerance": 0.02,
        }
    })

@dataclass
class RLGConfig:
    """Configuration for RLGames."""
    asymmetric_obs: bool = MISSING  # argument specifying whether this config requires asymmetric states
    params: Dict[Any, Any] = MISSING

@dataclass
class RLGArgs:
    verbose: bool = False

@dataclass
class Args:
    """Items which must be propogated down to other areas of configuration or which are expected as args in other places in the config."""

    # configuration about the env creation
    cfg_env:str = 'Base'
    cfg_train:str = 'Base'
    # todo - maybe do some validation on this? should match up to the config selected
    task:str = 'Trifinger'
    task_type:str = 'Python'  # Choose Python or C++
    experiment_name:str = 'Base'  # used in RLGames

    # confuguration about the env itself
    num_envs:int = 256  # overrides the default number of envs
    # todo - investigate interaction between this and pyDR
    randomize:bool = False  # whether to apply physics domain randomisation

    # other misc rags
    seed:int = 7 # random seed
    verbose:bool = False
    logdir:str = 'logs/'  # backs up the configs for this run

    # devcie config
    physics_engine: Any = 'physx' # field(default_factory=lambda: gymapi.SIM_PHYSX) # 'physx' or 'flex'
    device:str = 'GPU'  # CPU or GPU for running physics
    ppo_device:str = 'GPU'  # whether to use GPU for inference with PPO

    # RLGames Arguments
    play:bool = False  # if set runs trained policy (for use with rl games)
    train: bool = MISSING # opposite of play
    checkpoint:str = ''  # used to set checkpoint path

    # Common Gym Arguments
    headless:bool = False  # disables rendering
    compute_device_id:int = 0  # for CUDA
    graphics_deice_id:int = 0  # graphics device id

    wandb_project_name: str = 'trifinger-manip'
    wandb_log: bool = True


@dataclass
class Config:
    """Base config class."""
    gym: EnvConfig = MISSING
    rlg: Dict[str, Any] = MISSING
    args: Args = Args()

    output_root: str = MISSING

@dataclass
class ConfigSlurm:
    """Base config class."""
    gym: EnvConfig = MISSING
    rlg: Dict[str, Any] = MISSING
    args: Args = Args()

def update_cfg(cfg):
    """Modifies cfg by copying key arguments to the correct places.

    Args:
        cfg: Hydra config to modify
    """

    cfg.args.train = not cfg.args.play
    # Override settings if passed by the command line
    # Override number of environments
    cfg.gym.num_instances = cfg.args.num_envs
    # Override the phyiscs settings
    cfg.gym.sim.use_gpu_pipeline = cfg.args.device == "GPU"
    cfg.gym.sim.physx.use_gpu = cfg.args.device == "GPU"
    cfg.gym.physics_engine = cfg.args.physics_engine

    # Set cfg to enable asymmetric training
    cfg.gym.asymmetric_obs = cfg.rlg.asymmetric_obs

    # %% RLG config
    #  Override experiment name
    if cfg.args.experiment_name != 'Base':
        cfg.rlg.params.config.name = f"{cfg.args.experiment_name}_{cfg.args.task_type}_{cfg.args.device}_{str(cfg.args.physics_engine).split('_')[-1]}"

    cfg.rlg.params.load_checkpoint = cfg.args.checkpoint != ''
    cfg.rlg.params.load_path = cfg.args.checkpoint

    # Set number of environment instances
    with open_dict(cfg):
        cfg.rlg["params"]["config"]["minibatch_size"] = cfg.args.num_envs
        cfg.rlg["params"]["config"]["num_actors"] = cfg.args.num_envs
        # Set minibatch size for central value config
        if "central_value_config" in cfg.rlg["params"]["config"]:
            cfg.rlg["params"]["config"]["central_value_config"]["minibatch_size"] = cfg.args.num_envs
        cfg.gym.seed = cfg.args.seed
        cfg.rlg.seed = cfg.args.seed

def get_config_store():
    # Instantiates the different configurations in the correct groups.
    cs = ConfigStore.instance()
    cs.store(group="gym", name="trifinger_difficulty_1", node=TrifingerDifficulty1)
    cs.store(group="gym", name="trifinger_difficulty_2", node=TrifingerDifficulty2)
    cs.store(group="gym", name="trifinger_difficulty_3", node=TrifingerDifficulty3)
    cs.store(group="gym", name="trifinger_difficulty_4", node=TrifingerDifficulty4)

    # Don't need to instantiate the RLG configs as they are still yaml's - see corresponding directory.
    cs.store(name="config", node=Config)


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action

@hydra.main(config_name="config", config_path="D:\Code\Python\\all\\resources\config")
def launch_rlg_hydra(cfg: DictConfig):
    log = logging.getLogger(__name__)
    OmegaConf.update(cfg,"args[play]",True)
    OmegaConf.update(cfg,"gym[task_difficulty]",1)
    OmegaConf.update(cfg,"args[headless]",True)
    OmegaConf.update(cfg,"args[num_envs]",1)
    OmegaConf.update(cfg,"args[checkpoint]",R'D:\Code\Python\all\rl_models\trifinger_ep_68_rew_1077190.2.pth')
    OmegaConf.update(cfg,"args[wandb_log]",False)


    update_cfg(cfg)
    #print(OmegaConf.to_yaml(cfg))
    print("############################")
    agent_cfg_train = OmegaConf.to_container(cfg.rlg)
    print('Started to play')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent_cfg = deepcopy(agent_cfg_train)
    params = deepcopy(agent_cfg['params'])
    config=params['config']
    builder = model_builder.ModelBuilder()
    config['network'] = builder.load(params)
    network = config['network']

    #action_space = env_info['action_space']
    action_space_shape=(9,)
    action_space_low=-1
    action_space_high=1
    observation_space_shape=(41,)
    actions_num = action_space_shape[0]
    #observation_space=env_info['observation_space']
    normalize_input = config['normalize_input']
    normalize_value = config.get('normalize_value', False)
    num_agents =  1
    model_config = {
    'actions_num' : actions_num, #9
    'input_shape' : observation_space_shape,
    'num_seqs' : num_agents, #1
    'value_size': 1,
    'normalize_value': normalize_value,#False,
    'normalize_input': normalize_input,#False,
    } 
    model = network.build(model_config)
    model.to(device)
    model.eval()
    #checkpoint = torch_ext.load_checkpoint(R'D:\Code\Python\all\rl_models\trifinger_ep_68_rew_1077190.2.pth')
    filename=R'D:\Code\Python\all\rl_models\trifinger_ep_448_rew_1500442.4.pth'
    print("=> saving checkpoint '{}'".format(filename + '.pth'))
    checkpoint =torch.load(filename,map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    if normalize_input and 'running_mean_std' in checkpoint:
        model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])


    is_rnn = model.is_rnn()
    need_init_rnn=is_rnn
    is_deterministic=True
    
    """
    obs_spec = {
            "robot_q": self._dims.GeneralizedCoordinatesDim.value,
            "robot_u": self._dims.GeneralizedVelocityDim.value,
            "object_q": self._dims.ObjectPoseDim.value,
            "object_q_des": self._dims.ObjectPoseDim.value,
            "command": action_dim
        }
        obses=torch.tensor([[-0.2594,  0.3331, -0.2931, -0.2593,  0.3331, -0.2931, -0.3068,  0.2961,
         -0.3175,  1.1919,  1.0003, -0.2394,  1.1920,  1.0003, -0.2394,  1.0905,
          1.0002, -0.3151,  0.1915, -0.4365, -0.7240, -0.0114,  0.0311,  0.9474,
          0.3184, -0.0474, -0.0216, -0.7833,  0.0000,  0.0000,  0.0000,  1.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000]], device='cpu')
    """
    start_time=time.time()
    while(1):
        
        if (trifinger_state.euler_offset_init==True)&(np.array(trifinger_state.dof_pos).all!=0)&(trifinger_state.target_set==True)&(trifinger_state.safe_flag==True):
            if distance.euclidean(trifinger_state.cube_state[0,0:3],trifinger_state.target_cube_state[0,0:3])<0.03:
                print("success!")
                break
            
            else:
                
                dof_pos_normalize=trifinger_state.input_normalize(trifinger_state.dof_pos[0],trifinger_state.dof_pos_high,trifinger_state.dof_pos_low)
                dof_vel_normalize=trifinger_state.input_normalize(trifinger_state.dof_vel[0],trifinger_state.dof_vel_high,trifinger_state.dof_vel_low)
                cube_state_normalize=trifinger_state.input_normalize(trifinger_state.cube_state[0],trifinger_state.cube_state_high,trifinger_state.cube_state_low)
                target_cube_state_normalize=trifinger_state.input_normalize(trifinger_state.target_cube_state[0],trifinger_state.cube_state_high,trifinger_state.cube_state_low)
                dof_vel_normalize=np.clip(dof_vel_normalize,trifinger_state.dof_vel_low,trifinger_state.dof_vel_high)
                
                obses=np.append(dof_pos_normalize,dof_vel_normalize)
                obses=np.append(obses,cube_state_normalize)
                obses=np.append(obses,target_cube_state_normalize)
                obses=np.append(obses,trifinger_state.last_action)  
                obses=torch.tensor(obses,dtype=torch.float32).unsqueeze(0).to(device)

                print("----------------------------")
                print("trifinger_state.dof_pos (no norm)", trifinger_state.dof_pos)
                input_dict = {
                'is_train': False,
                'prev_actions': None, 
                'obs' : obses,
                'rnn_states' : None
                }
                with torch.no_grad():
                    res_dict = model(input_dict)
                mu = res_dict['mus']
                sigmas=res_dict['sigmas']
                action = res_dict['actions']
                if is_deterministic:
                    current_action = mu
                else:
                    current_action = action
                #print(mu,sigmas,action)
                action= rescale_actions(action_space_low, action_space_high, torch.clamp(current_action, -1.0, 1.0))
                
                action=action.squeeze().to('cpu').numpy()
                print("dof_pos_normalize", dof_pos_normalize)
                print("action (before)", action)
                for idx in range(len(dof_pos_normalize)):
                    if np.abs(dof_pos_normalize[idx])>0.95:
                        if dof_pos_normalize[idx]*action[idx]>0:
                            action[idx]=0
                #action[np.abs(dof_pos_normalize) > 0.95] = 0.0
                print("action (after)", action)
                trifinger_state.last_action[0]=action
                deta_time=time.time()-start_time
                if deta_time>0.05:
                    print(deta_time)
                else:
                    time.sleep(0.05-deta_time)
                

                trifinger_state.motor_control(action)
                start_time=time.time()
                if keyboard.is_pressed('q'):
                    print("stop!")
                    action=np.zeros(9)
                    for _ in range(5):
                        
                        trifinger_state.motor_control(action)
                    break



                    

if __name__ == "__main__":
    cs = get_config_store()
    Gcan.caninit()
    tcp_server.tcp_init()
    # 新开一个线程，用于接收新连接
    thread = threading.Thread(target=tcp_server.accept_client)
    thread.setDaemon(True)
    thread.start()
    tcp_server.trifinger_state.target_set(0.0,0.15,0.032,0.0,0.0,0.0)
    launch_rlg_hydra()



# %%
