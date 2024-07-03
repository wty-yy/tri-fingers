# 三指机器人参考文档

> 当前还在学习万强所完成的工作，项目完善中。。。

## 机器人硬件安装

> 教程 [GitHub - open_robot_actuator_hardware](https://github.com/open-dynamic-robot-initiative/open_robot_actuator_hardware)，论文 [TriFinger: An Open-Source Robot for Learning Dexterity](https://arxiv.org/abs/2008.03596)

![机器人](./assets/figures/机器人.png)

每个1个开发板 `F28069M` 可以最多控制2个驱动板 `DRV8305EVM`，每个驱动板控制一个电机 `MN4004 300KV`，每个电机还需要一个编码器 `AEDT-981x` 通过检测一个连接电机的转片来获取当前手臂位置和速度信息，每个手臂三个自由度，每个自由度需要一个电机控制，则需要9个电机。

按照手臂将整个机器分为3大块，每一块需要3个电机，因此需要2个开发板和1个驱动板，所以总共需要 `2*3` 个开发板，9个驱动板，9个电机。

### 电机处理

[电机改装教程](https://github.com/open-dynamic-robot-initiative/open_robot_actuator_hardware/blob/master/mechanics/actuator_module_v1/details/details_motor_preparation.md#details-motor-preparation)：先将转子（Rotor）和定子（Stator）拆分出来，分别进行处理

- 转子：将转子中间的孔径增大，把[加长的定制电机轴](https://github.com/open-dynamic-robot-initiative/open_robot_actuator_hardware/blob/master/mechanics/actuator_module_v1/drawings/motor_shaft.PDF)插进去。s
- 定子：将定子原本的三相线拆下来，焊接到更长的三相线上。

改装了电机还需要制作对应的执行器模块，需要使用两个半径大小1:3的减速齿轮，总共达到1:9的减速效果，从而能加强力矩输出效果，此部分逐步安装教程请见 [Step-by-Step Instructions](https://github.com/open-dynamic-robot-initiative/open_robot_actuator_hardware/?tab=readme-ov-file#step-by-step-instructions)。

### 接线

- 每个电机需要三相电通电（3根）连接到驱动板上
- 每个驱动板需要电源线通电（2根）连接外置电源（电压至少为23V，充电能到25V）
- 每个编码器需要 5v 线和接地线+3根通道线（5根）连接到开发板上
- 每个开发板需要连接CAN通讯线（2根）连接到GCAN设备（交换机）上

综上，一个手臂的数字控制组件包含 `3*3+2*3+5*3+2*2=34` 根线，总共需要 `34*3=102` 根线。

## 软件

### 开发板程序烧录

## 相机

我们需要两个（可以更多）摄像头 `D435`（`D435i` 也一样） 对方块位置进行识别，需要对相机的內外参进行标定，这些参数标定需要在 Linux 系统下进行。

我们在 Docker 下使用 ROS1 + RealSense + Kalibr 获取内参、畸变矩阵和外参，参考 [MyBlog - 小孔成像相机模型原理以及标定实现](https://wty-yy.xyz/posts/44869/)
