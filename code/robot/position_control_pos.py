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
import torch.nn.functional as F
import tcp_server
from tcp_server import trifinger_state
import Gcan
import threading
from scipy.spatial import distance
import keyboard
import time
from pathlib import Path
import csv

parent_path=Path(__file__).parent
data_path= parent_path/"data"

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Envs:
    observation_space=(41,)
    action_space=(9,)
    action_space_low=-1.0*np.ones(9)
    action_space_high=np.ones(9)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SAC_Actor(nn.Module):
    def __init__(self, env=Envs()):
        super().__init__()
        print("start_dim:",np.array(env.observation_space).prod())
        self.fc1 = nn.Linear(np.array(env.observation_space).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.action_space))
        self.fc_logstd = nn.Linear(256, np.prod(env.action_space))
        # action rescaling
        print("env.action_space.high",env.action_space_high)
        print("env.action_space.low",env.action_space_low)

        self.register_buffer(
            "action_scale", torch.tensor((env.action_space_high - env.action_space_low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space_high + env.action_space_low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
class PPOAgent(nn.Module):
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
# obs_low=torch.tensor([-0.50, 0.3, -2.7,-0.50, 0.3, -2.7,-0.50, 0.3, -2.7,
#                       -3.14159268/4,-3.14159268/4,-3.14159268/4,-3.14159268/4,-3.14159268/4,-3.14159268/4,-3.14159268/4,-3.14159268/4,-3.14159268/4,
#                       -0.3, -0.3, 0,-1,-1,-1,-1,
#                       -0.3, -0.3, 0,-1,-1,-1,-1,
#                       -0.50, 0.3, -2.7,-0.50, 0.3, -2.7,-0.50, 0.3, -2.7])
# obs_high=torch.tensor([0.50, 1.57, -0.0,0.50, 1.57, -0.0,0.50, 1.57, -0.0,
#                       3.14159268/4,3.14159268/4,3.14159268/4,3.14159268/4,3.14159268/4,3.14159268/4,3.14159268/4,3.14159268/4,3.14159268/4,
#                       0.3, 0.3, 0.3,1,1,1,1,
#                       0.3, 0.3, 0.3,1,1,1,1,
#                       0.50, 1.57, -0.0,0.50, 1.57, -0.0,0.50, 1.57, -0.0])
def get_action():
    dof_pos=trifinger_state.dof_pos[0][[3,4,5,6,7,8,0,1,2]]   #在归一化前先将关节顺序与仿真对齐
    dof_pos_normalize=trifinger_state.input_normalize(dof_pos,trifinger_state.dof_pos_high,trifinger_state.dof_pos_low)
    # dof_pos_normalize=dof_pos_normalize[[3,4,5,6,7,8,0,1,2]]
    dof_vel_normalize=trifinger_state.input_normalize(trifinger_state.dof_vel[0],trifinger_state.dof_vel_high,trifinger_state.dof_vel_low)
    cube_state_normalize=trifinger_state.input_normalize(trifinger_state.cube_state[0],trifinger_state.cube_state_high,trifinger_state.cube_state_low)
    target_cube_state_normalize=trifinger_state.input_normalize(trifinger_state.target_cube_state[0],trifinger_state.cube_state_high,trifinger_state.cube_state_low)
    dof_vel_normalize=np.clip(dof_vel_normalize,trifinger_state.dof_vel_low,trifinger_state.dof_vel_high)
    
    obses=np.append(dof_pos_normalize,dof_vel_normalize)
    obses=np.append(obses,cube_state_normalize)
    obses=np.append(obses,target_cube_state_normalize)
    obses=np.append(obses,trifinger_state.last_action[0])  
    obses=torch.tensor(obses,dtype=torch.float32).unsqueeze(0).to(device)
    obses[:,9:18]=0  #将速度mask掉 同仿真对齐
    if obses.shape!=(1,41):
        print(f"the shape of obs is wrong:{obses.shape}")
    if args.script=="SAC":
        action, _, mean = actor.get_action(obses)
        action=mean.squeeze().to('cpu').detach().numpy() ##action 真机没性能
    elif args.script=="PPO":
        action=agent.actor_mean(obses).squeeze().to('cpu').detach().numpy()
    
    obses=obses.cpu().squeeze().detach().numpy()
    #obs_tran=unscale_transform(obses,obs_low,obs_high).squeeze().to('cpu').detach().numpy()
    #print("obs:",obs_tran)
    action=np.clip(action,-1,1)
    trifinger_state.last_action[0]=action.copy()   #直接获取归一化的action
    action= rescale_actions(trifinger_state.dof_pos_low, trifinger_state.dof_pos_high, action)
    #print("action",action)
    action=action[[6,7,8,0,1,2,3,4,5]]  #反归一化后将关节顺序与真实机器人对齐
    return action,obses
@dataclass
class Args:
    checkpoint: str=R'D:\Code\Python\all\rl_models\sac_cube_noise_step310000_288.pt'
    device: str='cpu'
    data_save: bool=True
    script: str="SAC"

#仿真机器人关节  1 2 3 4 5 6 7 8 9
#真实机器人关节  4 5 6 7 8 9 1 2 3


if __name__=='__main__':
    args = tyro.cli(Args)
    device=args.device
    if args.script=="SAC":
        actor = SAC_Actor().to(device)
    elif args.script=="PPO":
        agent = PPOAgent().to(device)
    else:
        print("script wrong")
        exit()
    if args.checkpoint is not None and args.checkpoint !='':
        filename=args.checkpoint
        print(f"载入模型：{filename}")
        checkpoint=torch.load(filename,map_location=torch.device(device))
        if args.script=='SAC':
            actor.load_state_dict(checkpoint["actor"])
        elif args.script=="PPO":
            agent.load_state_dict(checkpoint)
    else:
        print("no_model_input!")
        exit()
    

    target_pos = np.array([0.0,0.9,-1.7]*3)
    start_flag=0 #判断是否到达默认开始位置
    Gcan.caninit()
    tcp_server.tcp_init()
    # 新开一个线程，用于接收新连接
    thread = threading.Thread(target=tcp_server.accept_client)
    thread.setDaemon(True)
    thread.start()
    trifinger_state.target_set(0.10,0.0,0.0325,0.0,0.0,0.0)
    start_time=time.time()
    delay_time=0
    run_step=0
    data_obs=np.zeros([10000,41])
    data_action=np.zeros([1000,9])
    data_done=np.zeros(10000)
    target=np.zeros(3)
    object=np.zeros(3)
    success=0
    epoch=0
    if args.data_save:
        import os
        data_save_path=os.path.join(data_path,"experient_ppo_action_noise.csv")
        

        with open(data_save_path,"w",encoding='utf8', newline='')as csv_file:
            writer=csv.writer(csv_file)
            writer.writerow([str(x) for x in range(16)])
    while(1):
        
        if (trifinger_state.euler_offset_init==True)&(np.array(trifinger_state.dof_pos).all!=0)&(trifinger_state.target_init==True)&(trifinger_state.safe_flag==True):
            if distance.euclidean(trifinger_state.cube_state[0,0:3],trifinger_state.target_cube_state[0,0:3])<0.03:
                print(f"success!   step:{run_step}")
                # data_done[run_step]=1
                success=1
                
                if args.data_save:
                    if run_step>0:
                        with open(data_save_path,"a",encoding='utf8', newline='')as csv_file:
                            writer=csv.writer(csv_file)
                            data=np.append(target,object)
                            data=np.append(data,success)
                            data=np.append(data,run_step)
                            # data=data.append(data_obs[idx]).append(data_action[idx]).append(data_done[idx])
                            writer.writerow(data)
                run_step=0
                x,y =trifinger_state.random_xy()
                print("new_target_pos",f"x:{x},y:{y}")
                trifinger_state.target_set(x,y,0.0325,0.0,0.0,0.0)
                target=trifinger_state.target_cube_state[0][0:3]
                object=trifinger_state.cube_state[0][0:3]
                epoch+=1
                if epoch>500:
                    action=np.array([0.0,0.9,-1.7]*3) 
                    trifinger_state.motor_control(action)
                    break

                
                
            elif run_step>50:
                print("timeout",run_step)
                success=0
                if args.data_save:
                    with open(data_save_path,"a",encoding='utf8', newline='')as csv_file:
                        writer=csv.writer(csv_file)
                        data=np.append(target,object)
                        data=np.append(data,success)
                        data=np.append(data,run_step)
                        # data=data.append(data_obs[idx]).append(data_action[idx]).append(data_done[idx])
                        writer.writerow(data)
                run_step=0
                x,y =trifinger_state.random_xy()
                print("new_target_pos",f"x:{x},y:{y}")
                trifinger_state.target_set(x,y,0.0325,0.0,0.0,0.0)
                target=trifinger_state.target_cube_state[0][0:3]
                object=trifinger_state.cube_state[0][0:3]
                epoch+=1
                if epoch>300:
                    action=np.array([0.0,0.9,-1.7]*3) 
                    trifinger_state.motor_control(action)
                    break
            else:
                if start_flag==0:
                    tcp_server.trifinger_state.motor_control(target_pos)
                    start_flag=1
                    target=trifinger_state.target_cube_state[0][0:3]
                    object=trifinger_state.cube_state[0][0:3]

                else:
                    deta_time=time.time()-start_time
                    if deta_time>0.05:
                        pass
                        # print("delay_time",deta_time)
                    else:
                        time.sleep(0.05-deta_time)
                    start_time=time.time()
                    
                    if np.all(abs(trifinger_state.dof_pos[0]-target_pos)<0.05):
                        print("reached_pos",target_pos)
                        action,obs=get_action()
                        tcp_server.trifinger_state.motor_control(action)
                        
                        target_pos=np.clip(action.copy(),trifinger_state.dof_pos_low,trifinger_state.dof_pos_high)
                        # print(action)
                        
                        # data_obs[run_step]=obs
                        # data_action[run_step]=trifinger_state.last_action[0]
                        run_step+=1

                        delay_time=0
                    elif delay_time>50:
                        print("can't reach",target_pos)
                        action,obs=get_action()
                        tcp_server.trifinger_state.motor_control(action)
                        
                        target_pos=np.clip(action.copy(),trifinger_state.dof_pos_low,trifinger_state.dof_pos_high)
                        # print(action)
                        # data_obs[run_step]=obs
                        # data_action[run_step]=trifinger_state.last_action[0]
                        run_step+=1

                        delay_time=0
                    else:
                        delay_time+=1
                  
                    if keyboard.is_pressed('q'):
                        print("stop!")
                        action=np.array([0.0,0.9,-1.7]*3) 
                        trifinger_state.motor_control(action)
                        break
                    
