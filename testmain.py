from pypower.runpf import runpf as runpowerflow
from pypower.loadcase import loadcase
from pf.utils import GetY, GetNetData, PolarNR
from pypower.makeYbus import makeYbus

if __name__ == "__main__":
    res = runpowerflow('pypower/case14')
    print(123)
    # coding=UTF-8

    ppc = loadcase('pypower/case14')
    bus = ppc['bus']
    gen = ppc['gen']
    branch = ppc['branch']
    Ybus, Yf, Yt = makeYbus(ppc['baseMVA'], bus, branch)

    PQNode, PVNode, SlackNode, P_Real, Q_Real = GetNetData(bus, gen)


    U = bus[:, 7]
    Angle = bus[:, 8]

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

    while True:
        Iter = Iter+1
        U, Angle, MaxError = PolarNR(U, Angle, Ybus, PQNode, PVNode, SlackNode, P_Real, Q_Real, Tol)
        if Iter > MaxIter or MaxError < Tol:
            break
    print(Iter, 'Solved')
