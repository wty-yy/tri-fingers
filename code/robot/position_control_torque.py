import numpy as np
import tcp_server
import Gcan
import keyboard
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from torch.distributions.normal import Normal
import tyro

import tcp_server
from tcp_server import trifinger_state
import Gcan
import threading
from scipy.spatial import distance
import keyboard
import time
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Envs:
    observation_space=(41,)
    action_space=(9,)
class Agent(nn.Module):
    def __init__(self, envs=Envs()):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.action_space)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
                                                                                                                
class PIDController:
    def __init__(self, Kp, Ki, Kd, target_angle):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target_angle = target_angle
        self.prev_error = np.zeros(len(target_angle))
        self.integral = np.zeros(len(target_angle))
    def target_set(self,target):
        self.target_angle=target
    def compute(self, feedback):
        error = self.target_angle - feedback
        self.integral += error
        derivative = error - self.prev_error

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        self.prev_error = error

        return output

def rescale_actions(low, high, action):
    m=np.multiply(np.add(high,low),0.5)
    d=np.multiply(np.subtract(high,low),0.5)
        #print(m,d)
    scaled_action=np.add(m,np.multiply(action,d))
    return scaled_action
    # d = (high - low) / 2.0
    # m = (high + low) / 2.0
    # scaled_action =  action * d + m
    # return scaled_action
def unscale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """
    Denormalizes a given input tensor from range of [-1, 1] to (lower, upper).

    @note It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape (dims,)
        upper: The maximum value of the tensor. Shape (dims,)

    Returns:
        Denormalized transform of the tensor. Shape (N, dims)
    """
    # default value of center
    offset = (lower + upper) * 0.5
    # return normalized tensor
    return x * (upper - lower) * 0.5 + offset
obs_low=torch.tensor([-0.50, 0.3, -2.7,-0.50, 0.3, -2.7,-0.50, 0.3, -2.7,
                      -3.14159268/4,-3.14159268/4,-3.14159268/4,-3.14159268/4,-3.14159268/4,-3.14159268/4,-3.14159268/4,-3.14159268/4,-3.14159268/4,
                      -0.3, -0.3, 0,-1,-1,-1,-1,
                      -0.3, -0.3, 0,-1,-1,-1,-1,
                      -0.50, 0.3, -2.7,-0.50, 0.3, -2.7,-0.50, 0.3, -2.7])
obs_high=torch.tensor([0.50, 1.57, -0.0,0.50, 1.57, -0.0,0.50, 1.57, -0.0,
                      3.14159268/4,3.14159268/4,3.14159268/4,3.14159268/4,3.14159268/4,3.14159268/4,3.14159268/4,3.14159268/4,3.14159268/4,
                      0.3, 0.3, 0.3,1,1,1,1,
                      0.3, 0.3, 0.3,1,1,1,1,
                      0.50, 1.57, -0.0,0.50, 1.57, -0.0,0.50, 1.57, -0.0])
def get_action():
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
    obses[:,9:18]=0  #将速度mask掉 同仿真对齐
    action=agent.actor_mean(obses).squeeze().to('cpu').detach().numpy()
    #obs_tran=unscale_transform(obses,obs_low,obs_high).squeeze().to('cpu').detach().numpy()
    #print("obs:",obs_tran)
    action= rescale_actions(trifinger_state.dof_pos_low, trifinger_state.dof_pos_high, action)
    #print("action",action)
    return action
@dataclass
class Args:
    checkpoint: str=R'D:\Code\Python\all\rl_models\4_12_2.pth'
    device: str='cpu'




if __name__=='__main__':
    args = tyro.cli(Args)
    device=args.device
    agent = Agent().to(device)
    if args.checkpoint is not None and args.checkpoint !='':
        filename=args.checkpoint
        print(f"载入模型：{filename}")
        checkpoint=torch.load(filename,map_location=torch.device(device))
        agent.load_state_dict(checkpoint)
    else:
        print("no_model_input!")
        exit()
    Kp = 1.5
    Ki = 0.04
    Kd = 0.2
    target = np.array([0.0,0.9,-1.7]*3)
    # 创建PID控制器实例
    pid_controller = PIDController(Kp, Ki, Kd, target)
    Gcan.caninit()
    tcp_server.tcp_init()
    # 新开一个线程，用于接收新连接
    thread = threading.Thread(target=tcp_server.accept_client)
    thread.setDaemon(True)
    thread.start()
    trifinger_state.target_set(0.10,0.10,0.0325,0.0,0.0,0.0)
    start_time=time.time()
    delay_time=0
    while(1):
        
        if (trifinger_state.euler_offset_init==True)&(np.array(trifinger_state.dof_pos).all!=0)&(trifinger_state.target_set==True)&(trifinger_state.safe_flag==True):
            if distance.euclidean(trifinger_state.cube_state[0,0:3],trifinger_state.target_cube_state[0,0:3])<0.03:
                print("success!")
                break
            
            else:
                 # 计算PID输出，即力矩
                feedback_angle=tcp_server.trifinger_state.dof_pos[0]
                torque = pid_controller.compute(feedback_angle)
                torque = np.clip(torque,-1.0,1.0)
                tcp_server.trifinger_state.motor_control(torque)
                deta_time=time.time()-start_time
                if deta_time>0.05:
                    pass
                    #print("delay_time",deta_time)
                else:
                    time.sleep(0.05-deta_time)
                start_time=time.time()
                
                if np.all(abs(trifinger_state.dof_pos[0]-pid_controller.target_angle)<0.05):
                    action=get_action()
                    pid_controller.target_set(action)
                    trifinger_state.last_action[0]=action
                    print(delay_time)

                    delay_time=0
                elif delay_time>5:
                    print(f"bad_pos:{pid_controller.target_angle}\npos{trifinger_state.dof_pos}")
                    action=get_action()
                    pid_controller.target_set(action)
                    trifinger_state.last_action[0]=action
                    
                    delay_time=0
                else:
                    delay_time+=1
                     
                
                if keyboard.is_pressed('q'):
                    print("stop!")
                    action=np.zeros(9)
                    for _ in range(5):
                        
                        trifinger_state.motor_control(action)
                    break
