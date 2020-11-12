from pf.utils import GetY, GetNetData, PolarNR
import numpy as np
from pf.makeMatrix import makeY
from scipy.sparse import csr_matrix
from pypower.ppoption import ppoption
from pypower.newtonpf import newtonpf
from caseSample import case14_non_topo_change
from pypower.loadcase import loadcase
import copy


def runpf(ppc):
    bus = ppc['bus']
    gen = ppc['gen']
    branch = ppc['branch']
    # from pypower.makeYbus import makeYbus
    # Ybus = makeYbus(ppc['baseMVA'], bus, branch)
    Ybus = makeY(bus, branch)

    PQNode, PVNode, SlackNode, P_Real, Q_Real = GetNetData(bus, gen)
    # Sbus = P_Real / 100 + 1j * Q_Real / 100
    from pypower.makeSbus import makeSbus
    Sbus = makeSbus(ppc['baseMVA'], bus, gen)

    V0 = bus[:, 7] * np.exp(1j * np.pi / 180 * bus[:, 8])

    ppopt = ppoption()
    ppopt['PF_MAX_IT'] = 100
    res = newtonpf(Ybus, Sbus, V0, SlackNode, PVNode, PQNode, ppopt=ppopt)
    if res[1] == 1:
        return np.abs(res[0]), np.angle(res[0])
    else:
        print('Do not converged in {} iterations.'.format(res[2]))
        return None, None


def prepare_train_data(case, save_dir, data_size=10000):
    if case == 'case14':
        ppc0 = loadcase('pypower/case14')
        Ybus = makeY(ppc0['bus'], ppc0['branch'])
        G = np.real(Ybus)
        B = np.imag(Ybus)
        input_P = []
        input_Q = []
        input_V = []
        input_Seta = []
        output_V = []
        output_Seta = []

        np.random.seed(0)
        while data_size >= 0:
            ppc = case14_non_topo_change(copy.deepcopy(ppc0))
            PQNode, PVNode, SlackNode, P_Real, Q_Real = GetNetData(ppc['bus'], ppc['gen'])
            v, angle = runpf(ppc)
            if v is not None:
                input_P.append(P_Real)
                input_Q.append(Q_Real)
                output_V.append(v)
                output_Seta.append(angle)
                input_V.append(ppc['bus'][:, 7])
                input_Seta.append(ppc['bus'][:, 8] * np.pi / 180)
                data_size -= 1
        print('Sample successfully.')
        np.savez(save_dir, G=G, B=B, input_Seta=input_Seta, input_V=input_V,
                 input_Q=input_Q, input_P=input_P, output_V=output_V, output_Seta=output_Seta)


if __name__ == "__main__":
    from pypower.runpf import runpf as runpowerflow
    res, sucess, Ybus = runpowerflow('pypower/case14')
    # # coding=UTF-8
    # prepare_train_data('case14', './sampleData/data1.npz', data_size=100)
    # data = np.load('./sampleData/data1.npz', allow_pickle=True)
    # # print(data.keys())
    # G = data['G']
    # output_V = data['output_V']
    # print(np.array(data['G']))

    ppc = loadcase('pypower/case14')
    bus = ppc['bus']
    gen = ppc['gen']
    branch = ppc['branch']
    Ybus = makeY(bus, branch)

    PQNode, PVNode, SlackNode, P_Real, Q_Real = GetNetData(bus, gen)
    # Sbus = P_Real / 100 + 1j * Q_Real / 100
    from pypower.makeSbus import makeSbus

    Sbus = makeSbus(ppc['baseMVA'], bus, gen)

    V0 = bus[:, 7] * np.exp(1j * np.pi / 180 * bus[:, 8])

    ppopt = ppoption()
    ppopt['PF_MAX_IT'] = 100
    res = newtonpf(Ybus, Sbus, V0, SlackNode, PVNode, PQNode, ppopt=ppopt)

    print('Do not converged in {} iterations.'.format(res[2]))

