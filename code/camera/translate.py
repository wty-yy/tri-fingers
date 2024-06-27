import numpy as np
def calPoseFrom3Points(Oab, Pxb, Pyb):
    '''
    作者：xyq；
    日期：2022.1.18；
    功能：已知坐标系a的原点、x轴正半轴上任一点和y轴正半轴上任一点在坐标系b下的坐标，
        求解坐标系a到坐标系b的旋转矩阵R和平移矩阵T；
    输入：坐标系a的原点在坐标系b下的坐标:Oab(x1,y1,z1);
        坐标系a的x轴正半轴上任一点在坐标系b下的坐标:Pxb(x2,y2,z2);
        坐标系a的y轴正半轴上任一点在坐标系b下的坐标:Pyb(x3,y3,z3);
    输出：坐标系n到坐标系s的旋转矩阵Rns，输出格式为矩阵;
    DEMO：
            import geomeas as gm
            import numpy as np

            Oab = np.array([-37.84381632, 152.36389864, 41.68600167])
            Pxb = np.array([-19.59820338, 139.58818292, 45.55380309])
            Pyb = np.array([-38.23270656, 157.3130709, 59.86810327])

            print(gm.Pose().calPoseFrom3Points(Oab, Pxb, Pyb))
    '''
    print(Oab,Pxb,Pyb)
    x = (Pxb - Oab) / np.linalg.norm(Pxb - Oab)
    print(np.linalg.norm(Pxb - Oab))
    y = (Pyb - Oab) / np.linalg.norm(Pyb - Oab)
    print(np.linalg.norm(Pyb - Oab))
    z = np.cross(x, y)
    length = np.linalg.norm(z)
    print(length)
    z = z / length
    Rab = np.matrix([x, y, z]).transpose()
    Tab = np.matrix(Oab).transpose()
    return Rab, Tab


#[ 0.02923178 -0.01699791  0.501     ] [0.0095496  0.01246656 0.476     ] [ 6.47407402e-02 -9.01732536e-06  4.87000000e-01]
Oab=np.array([ 0.02923178 ,-0.01699791 , 0.501] )
Pxb=np.array([0.0095496 , 0.01246656 ,0.476    ])
Pyb=np.array([ 6.47407402e-02, -9.01732536e-06 , 4.87000000e-01    ])
print(calPoseFrom3Points(Oab, Pxb, Pyb))