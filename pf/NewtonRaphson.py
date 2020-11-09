# coding=UTF-8
# 形成节点导纳矩阵
import numpy as np


def PolarNR(U, Angle, Y, PQNode, PVNode, SlackNode, P_Real, Q_Real, Tol, **Option):
    P_iter = 0  # 为形成雅可比矩阵
    Q_iter = 0  # 为形成雅可比矩阵
    PQNode = PQNode-1
    PVNode = PVNode-1
    SlackNode = SlackNode-1
    NumNode = Y.shape[0] # 节点数目
    NumPQ = max(PQNode.shape) # PQ节点数目
    G = Y.real
    B = Y.imag
    P = np.zeros([NumNode,1])
    Q = np.zeros([NumNode,1])
    DeltaP = np.zeros([NumNode-1,1])
    DeltaQ = np.zeros([NumPQ,1])
    for i in range(NumNode):  # 求解功率不平衡量
        P[i] = U[i]*np.sum(U*(G[i,:]*np.cos(Angle[i]-Angle) +  B[i,:]*np.sin(Angle[i]-Angle)))
        Q[i] = U[i]*np.sum(U*(G[i,:]* np.sin(Angle[i]-Angle) -  B[i,:]*np.cos(Angle[i]-Angle)))
        if i!=SlackNode:    # 不是平衡节点
            DeltaP[P_iter] = P_Real[i]-P[i]  # NumPQ+NumPV
            if i in PQNode:    # PQ节点
                DeltaQ[Q_iter] = Q_Real[i]-Q[i] # NumPQ
                Q_iter = Q_iter+1
            P_iter = P_iter+1
    DeltaPQ = np.vstack([DeltaP,DeltaQ])  # 功率不平衡量
    Option['string'] = '功率不平衡量为：\n'
    Real(DeltaPQ,**Option)
    MaxError = np.max(np.abs(DeltaPQ))
    print(MaxError)
    if MaxError<Tol:
        return(U,Angle,MaxError)
    ## H and N
    HN_iter = -1   # 初始化雅可比矩阵
    H = np.zeros([NumNode-1,NumNode-1])
    N = np.zeros([NumNode-1,NumPQ])
    # H and N
    for i in range(NumNode):
        if i!=SlackNode:  # PQ或PV节点
            H_iter_y = -1
            N_iter_y = -1
            HN_iter = HN_iter+1  # 记录H和N的行数
            for j in range(NumNode):
                if j != SlackNode:
                    H_iter_y = H_iter_y+1  # 记录H列数
                    if i != j:   # 非平衡节点计算H矩阵
                        Angleij = Angle[i]-Angle[j]
                        H[HN_iter,H_iter_y] = -U[i]*U[j]*(G[i,j]*np.sin(Angleij)-B[i,j]*np.cos(Angleij))
                    else:
                        H[HN_iter,H_iter_y] = Q[i]+U[i]**2*B[i,i]
                    if j in PQNode:
                        N_iter_y = N_iter_y+1  # 记录N的列数
                        if i != j:
                            Angleij = Angle[i]-Angle[j]
                            N[HN_iter,N_iter_y] = -U[i]*U[j]*(G[i,j]*np.cos(Angleij)+B[i,j]*np.sin(Angleij))
                        else:
                            N[HN_iter,N_iter_y] = -P[i]-G[i,i]*U[i]**2
    ## J and L
    JL_iter = -1   # 初始化雅可比矩阵
    J = np.zeros([NumPQ,NumNode-1])
    L = np.zeros([NumPQ,NumPQ])
    for i in range(NumNode):
        if i in PQNode:    # PQ节点
            JL_iter = JL_iter+1 # J和L的行数
            J_iter_y = -1
            L_iter_y = -1
            for j in range(NumNode):
                if j!=SlackNode:  # 非平衡节点
                    J_iter_y = J_iter_y+1
                    if i!=j:
                        Angleij = Angle[i]-Angle[j]
                        J[JL_iter,J_iter_y] = U[i]*U[j]*(G[i,j]*np.cos(Angleij)+B[i,j]*np.sin(Angleij))
                    else:
                        J[JL_iter,J_iter_y] = -P[i]+G[i,i]*U[i]**2
                    if j in PQNode:  # PQ节点
                        L_iter_y = L_iter_y+1
                        if i!=j:
                            Angleij = Angle[i]-Angle[j]
                            L[JL_iter,L_iter_y] = -U[i]*U[j]*(G[i,j]*np.sin(Angleij)-B[i,j]*np.cos(Angleij))
                        else:
                            L[JL_iter,L_iter_y] = -Q[i]+B[i,i]*U[i]**2
    # 修正
    Jaccobi = np.vstack([np.hstack([H,N]),np.hstack([J,L])])
    Option['string'] = 'jacobi矩阵为：\n'
    Real(Jaccobi,**Option)
    Delta = np.linalg.solve(Jaccobi,DeltaPQ)
    Option['string'] = '方程组求解结果：\n'
    Real(Delta,**Option)
    DeltaAngle = Delta[0:NumNode-1]  # 注意下标
    DeltaU_U = Delta[NumNode-1:]
    DA_iter = -1
    U_U_iter = -1
    for i in range(NumNode):
        if i!=SlackNode:
            DA_iter = DA_iter+1
            Angle[i] = Angle[i]-DeltaAngle[DA_iter]
            if i in PQNode:
                U_U_iter = U_U_iter+1
                U[i] = U[i]-U[i]*DeltaU_U[U_U_iter]
    Option['string'] = '更新之后的电压幅值为：\n'
    Real(U,**Option)
    Option['string'] = '相角为：\n'
    Real(Angle,**Option)
