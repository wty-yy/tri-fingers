import numpy as np
import tcp_server
import Gcan
import keyboard
import time


# 示例用法
if __name__ == "__main__":
    

    target = np.array([0.0,0.9,-1.7]*3)  # 目标角度为90度
    #dof_offset=np.array([0.4,0.3,0.2]*3)  #关节重力影响

    # 创建PID控制器实例
    Gcan.caninit()
    start_flag=0
    # 模拟电机反馈
    
    start_time=time.time()
    while(True):
        if keyboard.is_pressed('g'):
            start_flag=1
        if start_flag==1:
            # 计算PID输出，即力矩
            start_time=time.time()
            tcp_server.trifinger_state.motor_control(target)
            deta_time=time.time()-start_time
            print("delay_time",deta_time)
            if deta_time>0.05:
                print("delay_time",deta_time)
            else:
                time.sleep(0.05-deta_time)
            start_time=time.time()
            # 模拟电机运动，这里简单地假设电机在一定时间内运动到目标位置
            #feedback_angle += torque*0.785
            print('-'*20)
            print("pos:",tcp_server.trifinger_state.dof_pos[0])
            #if np.abs(pid_controller.target_angle-feedback_angle)
            start_flag=0
                
            if keyboard.is_pressed('q'):
                print("stop!")
                action=np.array([0.0,0.9,-1.7]*3)    
                tcp_server.trifinger_state.motor_control(action)
                


