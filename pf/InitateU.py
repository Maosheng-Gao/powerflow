# coding=UTF-8
# ---------------极坐标形式----------------#
from numpy import cos, sin, angle
import OutTxt


def PolarU(U_Amplitude, U_Angle, **Option):
    U_Complex = U_Amplitude * cos(U_Angle) + U_Amplitude * sin(U_Angle) * 1j
    U = abs(U_Complex)
    Angle = angle(U_Complex)
    if 'string' not in Option:
        Option['string'] = '\n电压初始化为：\n'
        OutTxt.Complex(U_Complex, **Option)
    return (U, Angle)


# ---------------直角坐标形式--------------#
def RectanU(U_Amplitude, U_Angle, **Option):
    U_e = U_Amplitude * cos(U_Angle)
    U_f = U_Amplitude * cos(U_Angle)
    if 'string' not in Option:
        Option['string'] = '\n 电压实部初始化为：\n'
        OutTxt.Real(U_e, **Option)
        Option['string'] = '\n 电压虚部初始化为：\n'
        OutTxt.Real(U_f, **Option)
