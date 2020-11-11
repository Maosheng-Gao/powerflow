from pf.utils import GetY, GetNetData, PolarNR
import numpy as np
from pf.makeMatrix import makeY
from scipy.sparse import csr_matrix
from pypower.ppoption import ppoption
from pypower.newtonpf import newtonpf
from caseSample import case14_non_topo_change


def runpf(ppc):
    bus = ppc['bus']
    gen = ppc['gen']
    branch = ppc['branch']
    Ybus = csr_matrix(makeY(bus, branch))

    PQNode, PVNode, SlackNode, P_Real, Q_Real = GetNetData(bus, gen)
    Sbus = P_Real / 100 + 1j * Q_Real / 100

    V0 = bus[:, 7] * np.exp(1j * np.pi / 180 * bus[:, 8])

    ppopt = ppoption()
    ppopt['PF_MAX_IT'] = 100
    res = newtonpf(Ybus, Sbus, V0, SlackNode, PVNode, PQNode, ppopt=ppopt)
    if res[1] == 1:
        return np.abs(res[0]), np.angle(res[0])
    else:
        print('Do not converged in {} iterations.'.format(res[2]))
        return None


if __name__ == "__main__":
    # res, sucess, Ybus = runpowerflow('pypower/case9')
    # coding=UTF-8
    np.random.seed(0)
    for x in range(10):
        ppc = case14_non_topo_change()
        print(runpf(ppc))

