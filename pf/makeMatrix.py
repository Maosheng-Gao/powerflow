import numpy as np
from scipy.sparse import csr_matrix as sparse


def makeY(nodeData, lineData):
    """
    make the admittance matrix
    :param nodeData: case data from matpower case
    :param lineData: case data from matpower case
    :return: admittance matrix
    """
    NodeNumber = nodeData.shape[0]
    LineNumber = lineData.shape[0]
    Y = np.zeros([NodeNumber, NodeNumber])+1j*np.zeros([NodeNumber, NodeNumber])
    G0 = nodeData[:, 4] # 节点对地电导
    B0 = nodeData[:, 5] # 节点对地电纳

    # 计算RX构成的导纳
    for l in range(LineNumber):
        fromBus = int(lineData[l, 0]-1)
        toBus = int(lineData[l, 1]-1)

        R = lineData[l, 2]
        X = lineData[l, 3]
        B_charging = lineData[l, 4]
        Y[fromBus, fromBus] = Y[fromBus, fromBus]+B_charging + 1/(R+1j*X)
        Y[toBus, toBus] = Y[toBus, toBus] + B_charging + 1/(R+1j*X)
        Y[fromBus, toBus] = Y[fromBus, toBus] - 1/(R+1j*X)
        Y[toBus, fromBus] = Y[toBus, fromBus] - 1/(R+1j*X)

    # 添加节点对地电纳
    for n in range(NodeNumber):
        Node = int(nodeData[n, 0]-1)
        Y[Node, Node] = Y[Node, Node] + G0[n] + 1j*B0[n]

    return Y


def makeSbus(baseMVA, bus, gen):
    """Builds the vector of complex bus power injections.

    Returns the vector of complex bus power injections, that is, generation
    minus load. Power is expressed in per unit.

    @see: L{makeYbus}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    ## generator info
    on = np.flatnonzero(gen[:, 7] > 0)      ## which generators are on?
    gbus = gen[on, 0]                   ## what buses are they at?

    ## form net complex bus power injection vector
    nb = bus.shape[0]
    ngon = on.shape[0]
    
    ## connection matrix, element i, j is 1 if gen on(j) at bus i is ON
    Cg = sparse((np.ones(ngon), (gbus, range(ngon))), (nb, ngon))

    ## power injected by gens plus power injected by loads converted to p.u.
    Sbus = ( Cg * (gen[on, 1] + 1j * gen[on, 2]) -
             (bus[:, 3] + 1j * bus[:, 4]) ) / baseMVA

    return Sbus