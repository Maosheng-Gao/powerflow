import numpy as np

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
        fromBus = int(lineData[l, 0])
        toBus = int(lineData[l, 0])

        R = lineData[l, 2]
        X = lineData[l, 3]
        B_charging = lineData[l, 4]
        Y[fromBus, toBus] = Y[fromBus, toBus]+B_charging + 1/(R+1j*X)
        Y[toBus, toBus] = Y[toBus, toBus] + B_charging + 1/(R+1j*X)
        Y[fromBus, toBus] = Y[fromBus, toBus] - 1/(R+1j*X)
        Y[toBus, fromBus] = Y[toBus, fromBus] - 1/(R+1j*X)

    # 添加节点对地电纳
    for line in range(LineNumber):
        Node = int(nodeData[line, 0]-1)
        Y[Node, Node] = Y[Node, Node] + G0[line] + 1j*B0[line]

    return Y