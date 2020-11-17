from pf.utils import GetY, GetNetData, PolarNR
import numpy as np
from pf.makeMatrix import makeY
from sampleData.caseSample import case14_non_topo_change

import copy
from sys import stdout, stderr
from time import time
from numpy import r_, c_, ix_, zeros, pi, ones, exp, argmax, union1d
from numpy import flatnonzero as find

from pypower.bustypes import bustypes
from pypower.ext2int import ext2int
from pypower.loadcase import loadcase
from pypower.makeYbus import makeYbus
from pypower.makeSbus import makeSbus

from pypower.ppver import ppver
from pypower.ppoption import ppoption

from pypower.newtonpf import newtonpf


from pypower.idx_bus import PD, QD, VM, VA, GS, BUS_TYPE, PV, PQ, REF
from pypower.idx_brch import PF, PT, QF, QT
from pypower.idx_gen import PG, QG, VG, QMAX, QMIN, GEN_BUS, GEN_STATUS


def runpf(ppc, ppopt):
    # add zero columns to branch for flows if needed
    if ppc["branch"].shape[1] < QT:
        ppc["branch"] = c_[ppc["branch"],
                           zeros((ppc["branch"].shape[0],
                                  QT - ppc["branch"].shape[1] + 1))]

    # convert to internal indexing
    ppc = ext2int(ppc)
    baseMVA, bus, gen, branch = \
        ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"]

    # get bus index lists of each type of bus
    ref, pv, pq = bustypes(bus, gen)

    # generator info
    on = find(gen[:, GEN_STATUS] > 0)  # which generators are on?
    gbus = gen[on, GEN_BUS].astype(int)  # what buses are they at?

    # -----  run the power flow  -----
    v = ppver('all')
    stdout.write('PYPOWER Version %s, %s' % (v["Version"], v["Date"]))
    # initial state
    V0 = bus[:, VM] * exp(1j * pi / 180 * bus[:, VA])
    vcb = ones(V0.shape)  # create mask of voltage-controlled buses
    vcb[pq] = 0  # exclude PQ buses
    k = find(vcb[gbus])  # in-service gens at v-c buses
    V0[gbus[k]] = gen[on[k], VG] / abs(V0[gbus[k]]) * V0[gbus[k]]

    # build admittance matrices
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)

    # compute complex bus power injections [generation - load]
    Sbus = makeSbus(baseMVA, bus, gen)

    V, success, i = newtonpf(Ybus, Sbus, V0, ref, pv, pq, ppopt)

    if success:
        return V, Sbus, Ybus
    else:
        print('Do not converged in {} iterations.'.format(i))
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
            ppopt = ppoption(None)
            v, Sbus, Ybus = runpf(ppc, ppopt)
            if v is not None:
                G = np.real(Ybus.todense())
                B = np.imag(Ybus.todense())
                input_P.append(np.real(Sbus))
                input_Q.append(np.imag(Sbus))
                output_V.append(np.abs(v))
                output_Seta.append(np.angle(v))
                input_V.append(ppc['bus'][:, 7])
                input_Seta.append(ppc['bus'][:, 8] * np.pi / 180)
                data_size -= 1
        print('Sample successfully.')
        np.savez(save_dir, G=G, B=B, input_Seta=input_Seta, input_V=input_V,
                 input_Q=input_Q, input_P=input_P, output_V=output_V, output_Seta=output_Seta)


def makePV_V(ppc):
    bus = ppc['bus']
    bus_type = bus[:, 1]
    ref = np.where(bus_type == 3)[0][0]
    gen = ppc['gen']
    Vm = bus[:, 7]
    index = gen[:, 0].astype('int32') - 1
    facter = np.zeros(Vm.shape)
    facter[index] = 1
    PV_V = Vm * facter

    return PV_V[:, np.newaxis].astype('float32'), ref


if __name__ == "__main__":
    # from pypower.runpf import runpf as runpowerflow
    # res, sucess = runpowerflow('pypower/case14')
    # # # coding=UTF-8
    # prepare_train_data('case14', './sampleData/test1.npz', data_size=1000)
    # data = np.load('./sampleData/data1.npz', allow_pickle=True)
    # # print(data.keys())
    # G = data['G']
    # output_V = data['output_V']
    # print(np.array(data['G']))

    ppc = loadcase('../pypower/case14')
    makePV_V(ppc)



