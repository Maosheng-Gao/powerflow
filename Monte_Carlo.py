import numpy as np
import random
import copy
from pypower.idx_bus import PD, QD, VM, VA, GS, BUS_TYPE, PV, PQ, REF


def Nweibull(a, scale, size):
    return scale*np.random.weibull(a, size)


def Monte_Carlo(ppc, varance):
    bus = ppc['bus']
    windF = ppc['windF']
    PV = ppc['PV']
    # 对负荷抽样，假定负荷均服从正态分布
    load_e_p = np.random.normal(loc=bus[:, PD],  scale=abs(varance*bus[:, PD]))
    load_e_q = np.random.normal(loc=bus[:, QD], scale=abs(varance * bus[:, QD]))

    # 风速威布尔分布  %假设风电只有 有功
    Pwind = np.zeros((windF.shape[0], 1))
    Vwind = np.zeros((windF.shape[0], 1))

    W_pro = 0   # 风机故障率
    W_break = np.ones((windF.shape[0], 1))
    random_num = np.random.rand(windF.shape[0], 1)   # 生成连续均匀分布的随机数
    W_break[np.where(random_num-W_pro < 0)[0], 0] = 0
    # 判断风机是否被抽掉  风机出力
    for tt in range(windF.shape[0]):
        if windF[tt, 1] == 0:
            Pwind[tt, 0] = 0
        else:
            if W_break[tt, 0] == 0:
                windF[tt, 0] = 0
                Pwind[tt, 0] = 0
            else:
                Vwind[tt, 0] = Nweibull(windF[tt, 2], windF[tt, 3], (1, 1))  # 风速服从威布尔分布
                if Vwind[tt, 0] > windF[tt, 4] and Vwind[tt] <= windF[tt, 5]:
                    Pwind[tt] = windF[tt, 1]*(Vwind[tt]-windF[tt, 4])/(windF[tt, 5]-windF[tt, 4])  # 根据风速算功率
                else:
                    if Vwind[tt]> windF[tt, 5] and Vwind[tt] <= windF[tt, 6]:
                        Pwind[tt] = windF[tt, 1]
                    else:
                        Pwind[tt] = 0
    Ppv = np.zeros((PV.shape[0], 1))
    PV_pro = 0    # 光伏电站故障率
    PV_break = np.ones((PV.shape[0], 1))
    random_num = np.random.rand(PV.shape[0], 1)  # 生成连续均匀分布的随机数
    PV_break[np.where(random_num - PV_pro < 0)[0], 0] = 0

    # 光伏发电 出力
    for ii in range(PV.shape[0]):
        if PV[ii, 1] == 0:
            Ppv[ii] = 0
        else:
            if PV_break[ii] == 0:
                PV[ii, 1] = 0
                Ppv[ii] = 0
            else:
                Ppv[ii] = PV[ii,1] * np.random.beta(PV[ii, 2], PV[ii, 3], (1, 1))   # 光伏功率计算

    PQ_after = np.zeros((bus.shape[0], 1))  # 新能源的功率

    for ii in range(windF.shape[0]):
        PQ_after[int(windF[ii, 0]), 0] = Pwind[ii, 0] + PQ_after[int(windF[ii, 0]), 0]

    for ii in range(PV.shape[0]):
        PQ_after[int(PV[ii, 0]), 0] = Ppv[ii, 0] + PQ_after[int(PV[ii, 0]), 0]

    bus[:, PD] = load_e_p
    bus[:, QD] = load_e_q
    bus[:, PD] = bus[:, PD] - PQ_after[:, 0]

    # 返回更新之后的ppc
    ppc['bus'] = bus
    return ppc


if __name__ == '__main__':
    from pypower.loadcase import loadcase
    ppc = loadcase('pypower/case14')
    # add wind power in ppc
    # WindP_bus(0), WindP_powermax(1), WindP_K(2), WindP_D(3),
    # WindP_inspeed(4), WindP_ratespeed(5), WindP_outspeed(6)
    ppc['windF'] = np.array([[11, 70, 2.016, 5.089, 3.5, 15, 25]])

    # PV_bus, PV_powermax(1), PV_Alpha(2), PV_Beta(3)
    ppc['PV'] = np.array([[4, 45, 2.06, 2.5]])
    ppc0 = copy.deepcopy(ppc)
    ppc2 = Monte_Carlo(ppc.copy(), 0.1)
    print(2)
