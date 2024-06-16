import numpy as np
import tcp_server
import Gcan
import keyboard
import time
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

# 示例用法
if __name__ == "__main__":
    
    # 设置PID参数和目标值
    Kp = np.array([1.5,1.5,1.5]*3)
    Ki = np.array([0.1,0.1,0.1]*3)
    Kd = np.array([0.2,0.2,0.2]*3)
    target = np.array([0.0,0.9,-1.7]*3)  # 目标角度为90度
    #dof_offset=np.array([0.4,0.3,0.2]*3)  #关节重力影响

    # 创建PID控制器实例
    pid_controller = PIDController(Kp, Ki, Kd, target)
    Gcan.caninit()
    start_flag=0
    # 模拟电机反馈
    feedback_angle = np.zeros(9)  # 初始反馈角度
    start_time=time.time()
    while(True):
        if keyboard.is_pressed('g'):
            start_flag=1
        if start_flag==1:
            # 计算PID输出，即力矩
            feedback_angle=tcp_server.trifinger_state.dof_pos[0]
            deta1_time=time.time()-start_time
            torque = pid_controller.compute(feedback_angle)
            deta2_time=time.time()-start_time
            torque = np.clip(torque,-1.0,1.0)
            deta3_time=time.time()-start_time
            tcp_server.trifinger_state.motor_control(torque)
            deta_time=time.time()-start_time
            if deta_time>0.05:
                print("delay_time",deta_time,"delay1_time",deta1_time,"delay2_time",deta2_time,"delay3_time",deta3_time)
            else:
                time.sleep(0.05-deta_time)
            start_time=time.time()
            # 模拟电机运动，这里简单地假设电机在一定时间内运动到目标位置
            #feedback_angle += torque*0.785
            print('-'*20)
            print("torque:",torque)
            print("Feedback Angle:", feedback_angle)
            #if np.abs(pid_controller.target_angle-feedback_angle)
            if keyboard.is_pressed('c'):
                start_angle=tcp_server.trifinger_state.dof_pos[0].copy()
                torque[8]=-1.0
                torque = np.clip(torque,-1.0,1.0)
                tcp_server.trifinger_state.motor_control(torque)
                time.sleep(0.5)
                deta=tcp_server.trifinger_state.dof_pos[0]-start_angle
                print(deta)
                print("stop!")
                action=np.zeros(9)
                for _ in range(5):
                    
                    tcp_server.trifinger_state.motor_control(action)
                break
                
            if keyboard.is_pressed('q'):
                print("stop!")
                action=np.zeros(9)
                for _ in range(5):
                    
                    tcp_server.trifinger_state.motor_control(action)
                break
