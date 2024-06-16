import os
import random
import time
from dataclasses import dataclass

# import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from leibnizgym.utils import rlg_train
import gym
from tqdm import trange
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 7
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_group:str="SAC"
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    checkpoint_path: str="/data/user/wanqiang/document/leibnizgym/models"

    # Algorithm specific arguments
    env_id: str = "Trifinger"
    """the environment id of the task"""
    env_nums:int=1
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    action_noise: float=0.1
    cube_pos_noise:float=0.1
    noise_clip:float=0.2
    action_domain_randomization: bool=True
    cube_domain_randomization: bool=True
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    
    """automatic tuning of the entropy coefficient"""


# def make_env(env_id, seed, idx, capture_video, run_name):
#     def thunk():
#         if capture_video and idx == 0:
#             env = gym.make(env_id, render_mode="rgb_array")
#             env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         else:
#             env = gym.make(env_id)
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         env.action_space.seed(seed)
#         return env

#     return thunk

class ReplayBuffer:
    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    timeouts: torch.Tensor
    def __init__(self,
                buffer_size:int,
                obs_dim:int,
                action_dim:int,
                n_envs:int,
                device:str
                 ) :
        self.device=device
        self.obs_dim=obs_dim
        self.action_dim=action_dim
        self.n_envs=n_envs
        self.buffer_size = max(buffer_size // n_envs, 1)
        self.observations = torch.zeros((self.buffer_size, self.n_envs, self.obs_dim), dtype=torch.float32,device=self.device)
        self.next_observations = torch.zeros((self.buffer_size, self.n_envs, self.obs_dim), dtype=torch.float32,device=self.device)
        self.actions=torch.zeros((self.buffer_size, self.n_envs,self.action_dim ), dtype=torch.float32,device=self.device)
        self.rewards=torch.zeros((self.buffer_size, self.n_envs ), dtype=torch.float32,device=self.device)
        self.dones=torch.zeros((self.buffer_size, self.n_envs ), dtype=torch.float32,device=self.device)
        self.pos=0
        self.full=False

    def add(self,
            obs:torch.Tensor,          
            next_obs:torch.Tensor,
            action:torch.Tensor,
            reward:torch.Tensor,
            done:torch.Tensor,
            ):
        self.observations[self.pos]=obs
        self.next_observations[self.pos]=next_obs
        self.actions[self.pos]=action
        self.rewards[self.pos]=reward
        self.dones[self.pos]=done
        self.pos+=1
        if self.pos==self.buffer_size:
            self.full=True
            self.pos=0

    def sample(self, batch_size: int):
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        data = {
            "observations":self.observations[batch_inds, :, :],
            "next_observations":self.next_observations[batch_inds, :, :],
            "actions":self.actions[batch_inds,:,:],
            "dones":self.dones[batch_inds,:],
            "rewards":self.rewards[batch_inds,:]

        }
        return data
class Data:
    def __init__(self,data) -> None:
        self.observations=data['observations']
        self.next_observations=data['next_observations']
        self.actions=data['actions']
        self.dones=data['dones']
        self.rewards=data['rewards']
        


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        print("start_dim:",np.array(env.observation_space.shape).prod())
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        print("env.action_space.high",env.action_space.high)
        print("env.action_space.low",env.action_space.low)

        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
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

from leibnizgym.utils.torch_utils import saturate, unscale_transform, scale_transform, quat_diff_rad

def test(hydra_cfg):
    
    from omegaconf import OmegaConf
    gym_cfg = OmegaConf.to_container(hydra_cfg.gym)
    rlg_cfg = OmegaConf.to_container(hydra_cfg.rlg)
    cli_args= hydra_cfg.args
    
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    envs=rlg_train.create_rlgpu_env2(task_cfg=gym_cfg,cli_args=cli_args)
    record=False
    if record:
        import csv
        from pathlib import Path
        path=Path(__file__).parents[1]
        path=path/'experient'
        path=os.path.join(path,'sac_cube_noise_check.csv')
        
        with open(path,"w")as csvfile:
            writer=csv.writer(csvfile)
            b=['target','object','success','steps']
            writer.writerow(b)
    actor = Actor(envs).to(device)
    if 'checkpoint' in cli_args and cli_args['checkpoint'] is not None and cli_args['checkpoint'] !='':
        filename=cli_args['checkpoint']
        print(f"载入模型：{filename}")
        checkpoint=torch.load(filename)
        print(checkpoint["actor"])
        actor.load_state_dict(checkpoint["actor"])
    print("start_sac_test")
    obs=torch.ones(41,device=device)*0.1
    print(actor.get_action(obs))
    exit()
    for _ in trange(5000):
        obs=envs.reset()
        obs[:,9:18]=0
        # print(obs)
        get_pos=obs.clone()
        obs_trans=unscale_transform(
                        get_pos[0],
                        lower=envs.obs_scale.low,
                        upper=envs.obs_scale.high
                    )
        step=0
        object=obs_trans[18:21].detach().squeeze().cpu().numpy()
        target=obs_trans[25:28].detach().squeeze().cpu().numpy()
        # print(target,object)

        success=0
        while (success==0) &(step<100):
            # print(obs)

            action, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            action=torch.clip(action,-1,1)
            nex_obs, rewards, next_done, info = envs.step(action)
            nex_obs[:,9:18]=0
            success=next_done.detach().squeeze().cpu().numpy()
            obs=nex_obs.clone()
            step+=1
        if record:
            
            data=np.append(target,object)
            data=np.append(data,success)
            data=np.append(data,step)
            # print(data)
            with open(path,"a")as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(data)

def train(hydra_cfg):
    from omegaconf import OmegaConf
    gym_cfg = OmegaConf.to_container(hydra_cfg.gym)
    cli_args= hydra_cfg.args

    args = tyro.cli(Args)
    args.env_nums=cli_args.num_envs
    args.track= not cli_args.wandb_log

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            group=args.wandb_group,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    

    # env setup
    # envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    # assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    envs=rlg_train.create_rlgpu_env2(task_cfg=gym_cfg,cli_args=cli_args)
    single_action_space = envs.action_space
    single_observation_space = envs.observation_space
    print(single_action_space.shape,single_observation_space)
    assert isinstance(envs.action_space, gym.spaces.Box), "only continuous action space is supported"
    max_action = float(1)

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    rb = ReplayBuffer(
        args.buffer_size,
        envs.observation_space.shape[0],
        envs.action_space.shape[0],
        args.env_nums,
        device,
    )
    start_time = time.time()

    if args.action_domain_randomization:
        rand_fre=200    
        dof_offset=(torch.randn(9,dtype=torch.float32,device=device)*args.action_noise).clamp(-args.noise_clip,args.noise_clip)
        # cube_pos_offset=(torch.randn(7,dtype=torch.float32,device=device))*0.1/2
    if args.cube_domain_randomization:
        rand_fre=200    
        cube_pos_offset=(torch.randn(7,dtype=torch.float32,device=device)*args.cube_pos_noise).clamp(-args.noise_clip,args.noise_clip)

    print("-"*20)
    if args.checkpoint_path!="":
        str_time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        args.checkpoint_path=os.path.join(args.checkpoint_path,f"sac_{str_time}")
        os.makedirs(args.checkpoint_path, exist_ok=True)
        print("checkpoint_path:",args.checkpoint_path)
    print("start sac training")
    print(f"num_env:{args.env_nums}")
    print(f'num_iterations:{args.total_timesteps}\t ')
    print(f"track:{args.track}   wandb:{cli_args.wandb_log}")
    print(f"device:{device}")
    print('action_randomazition:',args.action_domain_randomization)
    print('cube_randomazition:',args.cube_domain_randomization)

    print("-"*20)

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    obs[:,9:18]=0
    for global_step in trange(args.total_timesteps,desc="step:"):
        
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = torch.tensor(np.array([envs.action_space.sample() for _ in range(envs.num_envs)]),device=device)
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach()
        if args.action_domain_randomization:
                if global_step%rand_fre==0:
                    dof_offset=(torch.randn(9,dtype=torch.float32,device=device)*args.action_noise).clamp(-args.noise_clip,args.noise_clip)
                    # cube_pos_offset=(torch.randn(7,dtype=torch.float32,device=device))*0.1/2
                actions[:,0:9]+=dof_offset
            
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations,  infos = envs.step(actions)
        next_obs[:,9:18]=0
        if args.cube_domain_randomization:
                if global_step%rand_fre==0:
                    cube_pos_offset=(torch.randn(7,dtype=torch.float32,device=device)*args.cube_pos_noise).clamp(-args.noise_clip,args.noise_clip)
                next_obs[:,18:25]+=cube_pos_offset
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.clone()

        rb.add(obs, real_next_obs, actions, rewards, terminations)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            data=Data(data)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                # print(data.next_observations.shape,next_state_actions.shape)    #(256,1024,41)  (256,1024,9)    

                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                # print(qf2_next_target.shape,next_state_log_pi.shape)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                # print(min_qf_next_target.shape)
                # print(data.rewards.shape,data.dones.shape)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("reward",torch.mean(data.rewards).item(),global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
            if global_step %1000==0:
                if args.checkpoint_path!="":

                    if torch.mean(data.rewards).item()>50:
                        path=os.path.join(args.checkpoint_path,f"sac_continus_step{global_step}_{int(torch.mean(data.rewards).item())}")
                        state=dict()
                        state["actor"]=actor.state_dict()
                        state["qf1"]=qf1.state_dict()
                        state["qf2"]=qf2.state_dict()
                        state["q_optimizer"]=q_optimizer.state_dict()
                        state["actor_optimizer"]=actor_optimizer.state_dict()
                        torch.save(state,path)    
    
    writer.close()