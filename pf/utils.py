# coding=UTF-8
# 形成节点导纳矩阵
# NodeData: bus_i type Pd Qd Pg Qg Gs Bs Vm Va
# LineData: fbus  tbus r  x  b_2  ratio
import numpy as np
# import matlab
import matlab.engine
import numpy as np
import scipy.io as scio
# eng = matlab.engine.start_matlab()
#
# eng.cd("../matlab", nargout=0)


def GetY(bus, branch):
    NumNode = bus.shape[0]
    NumLine = branch.shape[0]
    Y = np.zeros([NumNode, NumNode])+np.zeros([NumNode,NumNode])*1j
    for i in range(NumLine):
        Node1 = int(branch[i, 0]-1)
        Node2 = int(branch[i, 1]-1)
        # print(Node1,Node2)
        R = branch[i, 2]
        X = branch[i, 3]
        if branch[i, 8] == 0:   # 普通线路，无变压器
            B_2 = branch[i, 4]
            Y[Node1, Node1] = Y[Node1, Node1]+B_2*1j+1/(R+1j*X)
            Y[Node2, Node2] = Y[Node2, Node2]+B_2*1j+1/(R+1j*X)
            Y[Node1, Node2] = Y[Node1, Node2]-1/(R+1j*X)
            Y[Node2, Node1] = Y[Node2, Node1]-1/(R+1j*X)
        else:  # 有变压器支路
            K = branch[i, 8]
            YT = 1/(R+1j*X)
            Y[Node1, Node1] = Y[Node1, Node1]+(K-1)/K*YT+YT/K
            Y[Node2, Node2] = Y[Node2, Node2]+(1-K)/K**2*YT+YT/K
            Y[Node1, Node2] = Y[Node1, Node2]-1/K*YT
            Y[Node2, Node1] = Y[Node2, Node1]-1/K*YT
    # # 节点导纳矩阵的自导纳
    # G0 = bus[:, 4]  # 节点对地电导
    # B0 = bus[:, 5]  # 节点对地电纳
    # for i in range(NumNode):
    #     Node = int(bus[i, 0]-1)   # 第一列为节点编号
    #     Y[Node, Node] = Y[Node, Node]+G0[i]+1j*B0[i]

    return Y


def GetNetData(bus, gen):
    """
    输出网络的各个节点的信息
    :param bus: matpower中bus数据
    :param gen: matpower中gen数据
    :return: PQNode, PVNode, SlackNode, P_Real, Q_Real
    """
    numBus = bus[:, 2].shape
    numGen = gen.shape[0]
    PQNode = bus[np.where(bus[:, 1] == 1), 0][0]-1  # PQ节点
    PVNode = bus[np.where(bus[:, 1] == 2), 0][0]-1  # PV节点
    SlackNode = bus[np.where(bus[:, 1] == 3), 0][0, 0]-1  # 平衡节点
    P_gen = np.zeros(numBus)
    Q_gen = np.zeros(numBus)
    for x in range(numGen):
        P_gen[int(gen[x, 0])-1] = gen[x, 1]
        Q_gen[int(gen[x, 0])-1] = gen[x, 2]
    P_Real = -bus[:, 2] + P_gen  # 节点输入有功功率
    Q_Real = -bus[:, 3] + Q_gen  # 节点输入无功功率

    return np.array(PQNode, dtype='int32'), np.array(PVNode, dtype='int32'), np.array([SlackNode], dtype='int32'), \
           P_Real, Q_Real


def PolarNR(U, Angle, Y, PQNode, PVNode, SlackNode, P_Real, Q_Real, Tol):
    P_iter = 0  # 为形成雅可比矩阵
    Q_iter = 0  # 为形成雅可比矩阵
    NumBus = Y.shape[0]  # 节点数目
    NumPQ = max(PQNode.shape)  # PQ节点数目
    G = Y.real
    B = Y.imag
    P = np.zeros([NumBus, 1])
    Q = np.zeros([NumBus, 1])
    DeltaP = np.zeros([NumBus-1, 1])
    DeltaQ = np.zeros([NumPQ, 1])
    # 求解功率不平衡量
    for i in range(NumBus):
        P[i] = U[i]*np.sum(U*(G[i, :]*np.cos(Angle[i]-Angle) + B[i, :]*np.sin(Angle[i]-Angle)))
        Q[i] = U[i]*np.sum(U*(G[i, :]* np.sin(Angle[i]-Angle) - B[i, :]*np.cos(Angle[i]-Angle)))
        if i != SlackNode:    # 不是平衡节点
            DeltaP[P_iter] = P_Real[i]-P[i]  # NumPQ+NumPV
            if i in PQNode:    # PQ节点
                DeltaQ[Q_iter] = Q_Real[i]-Q[i] # NumPQ
                Q_iter = Q_iter+1
            P_iter = P_iter+1
    DeltaPQ = np.vstack([DeltaP, DeltaQ])  # 功率不平衡量
    MaxError = np.max(np.abs(DeltaPQ))
    print(MaxError)
    if MaxError<Tol:
        return(U, Angle, MaxError)
    HN_iter = -1   # 初始化雅可比矩阵
    H = np.zeros([NumBus-1, NumBus-1])
    N = np.zeros([NumBus-1, NumPQ])
    # H and N
    for i in range(NumBus):
        if i != SlackNode:  # PQ或PV节点
            H_iter_y = -1
            N_iter_y = -1
            HN_iter = HN_iter+1  # 记录H和N的行数
            for j in range(NumBus):
                if j != SlackNode:
                    H_iter_y = H_iter_y+1  # 记录H列数
                    if i != j:   # 非平衡节点计算H矩阵
                        Angleij = Angle[i]-Angle[j]
                        H[HN_iter, H_iter_y] = -U[i]*U[j]*(G[i, j]*np.sin(Angleij) - B[i, j]*np.cos(Angleij))
                    else:
                        H[HN_iter, H_iter_y] = Q[i]+U[i]**2*B[i, i]
                    if j in PQNode:
                        N_iter_y = N_iter_y+1  # 记录N的列数
                        if i != j:
                            Angleij = Angle[i]-Angle[j]
                            N[HN_iter, N_iter_y] = -U[i]*U[j]*(G[i, j]*np.cos(Angleij) + B[i, j]*np.sin(Angleij))
                        else:
                            N[HN_iter, N_iter_y] = -P[i]-G[i, i]*U[i]**2
    # J and L
    JL_iter = -1   # 初始化雅可比矩阵
    J = np.zeros([NumPQ, NumBus-1])
    L = np.zeros([NumPQ, NumPQ])
    for i in range(NumBus):
        if i in PQNode:    # PQ节点
            JL_iter = JL_iter+1 # J和L的行数
            J_iter_y = -1
            L_iter_y = -1
            for j in range(NumBus):
                if j != SlackNode:  # 非平衡节点
                    J_iter_y = J_iter_y+1
                    if i != j:
                        Angleij = Angle[i]-Angle[j]
                        J[JL_iter, J_iter_y] = U[i]*U[j]*(G[i, j]*np.cos(Angleij)+B[i, j]*np.sin(Angleij))
                    else:
                        J[JL_iter, J_iter_y] = -P[i]+G[i, i]*U[i]**2
                    if j in PQNode:  # PQ节点
                        L_iter_y = L_iter_y+1
                        if i!=j:
                            Angleij = Angle[i]-Angle[j]
                            L[JL_iter, L_iter_y] = -U[i]*U[j]*(G[i, j]*np.sin(Angleij)-B[i, j]*np.cos(Angleij))
                        else:
                            L[JL_iter, L_iter_y] = -Q[i]+B[i, i]*U[i]**2
    # 修正
    Jaccobi = np.vstack([np.hstack([H, N]), np.hstack([J, L])])
    Delta = np.linalg.solve(Jaccobi, DeltaPQ)
    DeltaAngle = Delta[0:NumBus-1]
    DeltaU_U = Delta[NumBus-1:]
    DA_iter = -1
    U_U_iter = -1
    for i in range(NumBus):
        if i != SlackNode:
            DA_iter = DA_iter+1
            Angle[i] = Angle[i]-DeltaAngle[DA_iter]
            if i in PQNode:
                U_U_iter = U_U_iter+1
                U[i] = U[i]-U[i]*DeltaU_U[U_U_iter]
    return U, Angle, MaxError


if __name__ == '__main__':
    A = eng.case14()
    bus = np.array(A['bus'])
    gen = np.array(A['gen'])
    branch = np.array(A['branch'])
    GetY(bus, branch)
    GetNetData(bus, gen, branch)
    print(A)