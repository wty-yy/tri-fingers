import socket  
import json
import numpy as np

ADDRESS = ('192.168.1.143', 123)
# ADDRESS = ('192.168.43.84', 123)

# 如果开多个客户端，这个client_type设置不同的值，比如客户端1为linxinfa，客户端2为linxinfa2
client_type ='pose'
target_pos=np.zeros(3)
def tcp_init():
    client=socket.socket()
    client.connect(ADDRESS)
    print(client.recv(1024).decode(encoding='utf8'))
    send_data(client, 'CONNECT','0')
    return client
def send_data(client, cmd, kv):
    """
    发包JSON格式为
    {
        'COMMAND': 命令 (str)
        'client_type': 客户机的类型 (str)
        'data': 数据 (str)
    }
    """
    global client_type
    jd = {}
    jd['COMMAND'] = cmd
    jd['client_type'] = client_type
    jd['data'] = kv
    
    jsonstr = json.dumps(jd)
    print('send: ' + jsonstr)
    client.sendall(jsonstr.encode('utf8'))

def input_client_type():
    return input("注册客户端，请输入名字 :")
def get_target(client):
    global target_pos
    while True:
        msg=client.recv(1024).decode(encoding='utf8')
        if len(msg)>120:
            continue
        jd=json.loads(msg)
        if jd['client_type']=='trifinger':
            data=jd['data']
            parts=data.replace('[','',1).replace(']','',1).replace('\n','').split(' ')
            data=[float(part)   for part in parts if part !='']
            target_pos=data

    
if '__main__' == __name__:
    client_type = input_client_type()
    client = socket.socket() 
    client.connect(ADDRESS)
    print(client.recv(1024).decode(encoding='utf8'))
    send_data(client, 'CONNECT',0)
    import time
    while 1:
        send_data(client,'SEND_DATA',0)
        print(client.recv(1024).decode(encoding='utf8'))
        time.sleep(1)


