继承关机：IsaacEnvBase -> TrifingerEnv  传入-> VecTaskPython

env_base.py 中 IsaacEnvBase.step 函数

- 先把动作存入action_buf
- 执行pre_step（设置动作执行方法）
- simulate（开始执行物理仿真，执行次数为control_decimation）
- post_step（获取机器人和方块的state，计算reward）

环境和奖励文件：leibnizgym/envs/trifinger/trifinger_env.py, rewards.py

环境创建文件：leibnizgym/utils/rlg_train.py -> RlGameGpuEnvAdapter和create_rl_gpu_env将TrifingerEnv传入到VeecTaskPython

训练文件：scripts/rlg_hydra.py

算法代码：scripts/sac.py, ppo_tt.py...

模型保存：output/{日期} 或者 models/

isaac gym

- urdf文件：resources/assets/trifinger/robot_properties_fingers/urdf/edu/trifingeredu.urdf,
- 网格文件：resources/assets/trifinger/robot_properties_fingers/meshes/...