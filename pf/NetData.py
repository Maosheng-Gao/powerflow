# coding=UTF-8
import numpy as np


# ----------------------------NodeData-------------------------#
# NodeData: bus_i type Pd Qd Pg Qg Gs Bs Vm Va
# bus_i为节点编号
# Pd和Qd为节点的负荷功率
# Pg和Qg为节点的发电机输出功率
# Gs和Bs为节点对地电导和电纳
# Vm为电压幅值，Va为电压相角
def GetNumNode(FilePath_Node):
    with open(FilePath_Node, 'r') as file:
        NumNode = 0
        for line in file:
            NumNode = NumNode + 1  # 得到节点数目
        print('节点数目：', NumNode)
    return (NumNode)


def GetNodeData(FilePath_Node, *NumNode, **PlotOption):
    if not NumNode:  # 空的
        NumNode = GetNumNode(FilePath_Node)
    NodeData = np.zeros([NumNode, 10])  # 初始化
    i = 0
    with open(FilePath_Node, 'r') as file:
        for line in file:
            NodeData[i, :] = np.array([float(i) for i in line.split()])
            i = i + 1
    if 'show' in PlotOption and PlotOption['show'] == 1:
        print('NodeData=', NodeData)
    return (NodeData)


# ----------------------------LineData------------------------#
# LineData:fbus  tbus   r    x    b    ratio
# fbus为起始节点
# tbus为终止节点
# r为支路电阻
# x为支路电抗
# b为节点对地导纳
# ratio为变压器变比
def GetNumLine(FilePath_Line):
    with open(FilePath_Line, 'r') as file:
        NumLine = 0
        for line in file:
            NumLine = NumLine + 1  # 得到节点数目
        print('支路数目：', NumLine)
    return (NumLine)


def GetLineData(FilePath_Line, *NumLine, **PlotOption):
    if not NumLine:  # 空的
        NumLine = GetNumLine(FilePath_Line)
    LineData = np.zeros([NumLine, 6])  # 初始化
    i = 0
    with open(FilePath_Line, 'r') as file:
        for line in file:
            LineData[i, :] = np.array([float(i) for i in line.split()])
            i = i + 1
    if 'show' in PlotOption and PlotOption['show'] == 1:
        print('LineData = ', LineData)
    return (LineData)


# -输出网络参数--------#
def GetNetData(NodeData, LineData, **Option):
    PQNode = NodeData[np.where(NodeData[:, 1] == 1), 0]  # PQ节点
    PVNode = NodeData[np.where(NodeData[:, 1] == 2), 0]  # PV节点
    SlackNode = NodeData[np.where(NodeData[:, 1] == 3), 0]  # 平衡节点
    # SlackNode = SlackNode[0]
    # print(SlackNode)
    P_Real = -NodeData[:, 2] + NodeData[:, 4]  # 节点输入有功功率
    Q_Real = -NodeData[:, 3] + NodeData[:, 5]  # 节点输入无功功率

    return PQNode, PVNode, SlackNode, P_Real, Q_Real