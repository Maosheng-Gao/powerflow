# coding=UTF-8
from tools.utils import GetY, GetNetData, PolarNR
import matlab.engine
import numpy as np
import scipy.io as scio
eng = matlab.engine.start_matlab()
eng.cd("./matlab", nargout=0)
netData = eng.case9()
bus = np.array(netData['bus'])
gen = np.array(netData['gen'])
branch = np.array(netData['branch'])
Y = GetY(bus, branch)  # 节点导纳矩阵
U = bus[:, 7]  # 获取电压幅值核相角初始化值
Angle = bus[:, 8]/180*np.pi
# 开始计算
G = np.real(Y)
PQNode, PVNode, SlackNode, P_Real, Q_Real = GetNetData(bus, gen)


# # 使用书上的例题进行的测试
# Y = np.array([[1.1474-13.9580*1j, -0.2494+4.9875*1j, -0.9430+9.430*1j],
#               [-0.2494+4.9875*1j, 0.74445-9.9080*1j, -0.49505+4.9505*1j],
#               [-0.9430 + 9.430*1j, -0.4951+4.951*1j, 1.48515-14.8315*1j]])
# U = np.array([1.0, 1.01, 1.0])
# Angle = np.array([0.0, 0.0, 0.0])
# PQNode = np.array([0])
# PVNode = [1, 2]
# SlackNode = 2
# P_Real = np.array([-2.0, 0.5, 0])
# Q_Real = np.array([-1.0, 0.0, 0])

Iter = 0
MaxIter = 1000
Tol = 1e-4


# while True:
#     Iter = Iter+1
#     U, Angle, MaxError = PolarNR(U, Angle, Y, PQNode, PVNode, SlackNode, P_Real, Q_Real, Tol)
#     if Iter > MaxIter or MaxError < Tol:
#         break
# print(Iter, 'Solved')
