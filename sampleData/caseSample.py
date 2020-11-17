import numpy as np
from pypower.loadcase import loadcase
from sampleData.Monte_Carlo import Monte_Carlo


def case14_non_topo_change(ppc):
    # 蒙特卡洛抽样
    # WindP_bus(0), WindP_powermax(1), WindP_K(2), WindP_D(3),
    # WindP_inspeed(4), WindP_ratespeed(5), WindP_outspeed(6)
    ppc['windF'] = np.array([[11, 70, 2.016, 5.089, 3.5, 15, 25]])
    # PV_bus, PV_powermax(1), PV_Alpha(2), PV_Beta(3)
    ppc['PV'] = np.array([[4, 45, 2.06, 2.5]])
    ppc = Monte_Carlo(ppc, 0.1)
    return ppc


if __name__ == "__main__":
    # res, sucess, Ybus = runpowerflow('pypower/case9')
    # coding=UTF-8
    ppc = loadcase('pypower/case30')

    # 蒙特卡洛抽样
    # WindP_bus(0), WindP_powermax(1), WindP_K(2), WindP_D(3),
    # WindP_inspeed(4), WindP_ratespeed(5), WindP_outspeed(6)
    ppc['windF'] = np.array([[11, 70, 2.016, 5.089, 3.5, 15, 25]])
    # PV_bus, PV_powermax(1), PV_Alpha(2), PV_Beta(3)
    ppc['PV'] = np.array([[4, 45, 2.06, 2.5]])

