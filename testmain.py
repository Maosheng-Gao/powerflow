from pypower.runpf import runpf as runpowerflow
from pypower.loadcase import loadcase
from pf.utils import GetY, GetNetData, PolarNR
from pypower.makeYbus import makeYbus
import numpy as np
from pf.makeMatrix import makeY
from scipy.sparse import csr_matrix
from pypower.ppoption import ppoption


if __name__ == "__main__":
    # res, sucess, Ybus = runpowerflow('pypower/case9')
    # coding=UTF-8
    ppc = loadcase('pypower/case30')
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
