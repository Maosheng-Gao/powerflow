from pypower.runpf import runpf as runpowerflow
from pypower.loadcase import loadcase
from pf.utils import GetY, GetNetData, PolarNR
from pypower.makeYbus import makeYbus
import numpy as np
from pf.makeMatrix import makeY
from scipy.sparse import csr_matrix
from pypower.ppoption import ppoption
from Monte_Carlo import Monte_Carlo


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

    # ppc = res
    bus = ppc['bus']
    gen = ppc['gen']
    branch = ppc['branch']
    Ybus = csr_matrix(makeY(bus, branch))

    PQNode, PVNode, SlackNode, P_Real, Q_Real = GetNetData(bus, gen)
    Sbus = P_Real/100+1j*Q_Real/100

    U = bus[:, 7]
    Angle = bus[:, 8]
    V0 = bus[:, 7] * np.exp(1j * np.pi/180 * bus[:, 8])

    from pypower.newtonpf import newtonpf
    ppopt = ppoption()
    ppopt['PF_MAX_IT'] = 100
    newtonpf(Ybus, Sbus, V0, SlackNode, PVNode, PQNode, ppopt=ppopt)
    # newtonpf(Ybus, Sbus, V0, ref, pv, pq, ppopt=None)
