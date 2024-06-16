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
# from stable_baselines3.common.buffers import ReplayBuffer
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
    wandb_group:str="DDPG"
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    checkpoint_path: str="/data/user/wanqiang/document/leibnizgym/models"

    # Algorithm specific arguments
    env_id: str = "trifinger"
    """the environment id of the Atari game"""
    env_nums:int=1
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0 #0.1
    """the scale of exploration noise"""
    learning_starts: int = 5e3 #25e3   #1024 envs
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0  #0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""


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
class QNetwork(nn.Module):
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


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


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
    envs=rlg_train.create_rlgpu_env2(gym_cfg=gym_cfg,cli_args=cli_args)
    single_action_space = envs.action_space
    single_observation_space = envs.observation_space
    print(single_action_space.shape,single_observation_space)
    assert isinstance(envs.action_space, gym.spaces.Box), "only continuous action space is supported"
    
    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    # envs.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.observation_space.shape[0],
        envs.action_space.shape[0],
        args.env_nums,
        device,
    )
    start_time = time.time()
    print("-"*20)
    print("start_ddpg training ")
    if args.checkpoint_path!="":
        str_time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        args.checkpoint_path=os.path.join(args.checkpoint_path,f"ddpg_{str_time}")
        os.makedirs(args.checkpoint_path, exist_ok=True)
        print("checkpoint_path:",args.checkpoint_path)
    
    print(f"num_env:{args.env_nums}")
    print(f'num_iterations:{args.total_timesteps}\t ')
    print(f"track:{args.track}   wandb:{cli_args.wandb_log}")
    print(f"device:{device}")

    print("-"*20)
    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    obs[:,9:18]=0
    for global_step in trange(args.total_timesteps,desc="step:"):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = torch.tensor(np.array([envs.action_space.sample() for _ in range(envs.num_envs)]),device=device)
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                # actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions=torch.clip(actions,torch.tensor(envs.action_space.low,dtype=torch.float32,device=device), torch.tensor(envs.action_space.high,dtype=torch.float32,device=device))
                actions = actions.detach()
                # actions = actions.cpu.clip(envs.action_space.low, envs.action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations,  infos = envs.step(actions)
        next_obs[:,9:18]=0
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
                next_state_actions = target_actor(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("reward",torch.mean(data.rewards).item(),global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            if global_step %1000==0:
                if args.checkpoint_path!="":

                    if torch.mean(data.rewards).item()>50:
                        path=os.path.join(args.checkpoint_path,f"ddpg_continus_step{global_step}_{int(torch.mean(data.rewards).item())}")
                        state=dict()
                        state["actor"]=actor.state_dict()
                        state["qf1"]=qf1.state_dict()
                        state["q_optimizer"]=q_optimizer.state_dict()
                        state["actor_optimizer"]=actor_optimizer.state_dict()
                        torch.save(state,path)    




    

    writer.close()