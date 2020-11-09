# coding=UTF-8
def ArgOption(**Option):  # 解析可选参数
    if 'string' in Option:
        String = Option['string']
    else:
        String = '输出的矩阵：\n'  # 默认值
    if 'path' in Option:
        FilePath = Option['path']
    else:
        FilePath = 'untitled.txt'  # 默认为未命名的
    if 'width' in Option:
        N = Option['width']
    else:
        N = 8  # 默认值
    if 'fmt' in Option:
        Fmt = Option['fmt']
    else:
        Fmt = 'a'  # 默认值
    return (String, FilePath, N, Fmt)


# ---------------------实数矩阵-----------------------#
def Real(Matrix, **Option):
    string, FilePath, N, Fmt = ArgOption(**Option)
    with open(FilePath, Fmt) as file:
        file.write(string)
        Ndim = Matrix.ndim
        if Ndim == 1:  # (Num)
            Num = Matrix.shape[0]
            for i in range(Num):
                file.write(str(Dot10(Matrix[i])).ljust(N + 1)[0:N - 1])
                file.write('  ')
            file.write('\n')
        elif Ndim == 2:  # ([NumR,NumC])
            NumR, NumC = Matrix.shape
            for i in range(NumR):
                for j in range(NumC):
                    file.write(str(Dot10(Matrix[i, j])).ljust(N + 1)[0:N - 1])
                    file.write('  ')
                file.write('\n')


# -------------------复数矩阵-----------------------#
def Complex(Matrix, **Option):
    string, FilePath, N, Fmt = ArgOption(**Option)
    RealM = Matrix.real
    ImagM = Matrix.imag
    # print(FilePath)
    with open(FilePath, Fmt) as file:
        file.write(string)
        Ndim = Matrix.ndim
        if Ndim == 1:  # (Num)
            Num = Matrix.shape[0]
            for i in range(Num):
                file.write(str(Dot10(RealM[i])).ljust(N + 1)[0:N - 1])
                file.write('+')
                file.write('j' + str(Dot10(ImagM[i])).ljust(N + 1)[0:N - 1])
                file.write('  ')
            file.write('\n')
        elif Ndim == 2:  # ([NumR,NumC])
            NumR, NumC = Matrix.shape
            for i in range(NumR):
                for j in range(NumC):
                    file.write(str(Dot10(RealM[i, j])).ljust(N + 1)[0:N - 1])
                    file.write('+')
                    file.write('j' + str(Dot10(ImagM[i, j])).ljust(N + 1)[0:N - 1])
                    file.write('  ')
                file.write('\n')


# --单变量-----------------------#
def SingleTxt(Number, **Option):
    string, FilePath, N, Fmt = ArgOption(**Option)
    with open(FilePath, Fmt) as file:
        file.write(string)
        file.write(str(Dot10(Number)).ljust(N + 1)[0:N - 1])
        file.write('\n')


# --字符串-----------------------#
def StringTxt(**Option):
    string, FilePath, N, Fmt = ArgOption(**Option)
    with open(FilePath, Fmt) as file:
        file.write(string)
        file.write('\n')


# --小数点后保留位数---
def Dot10(x):
    if isinstance(x, int):  # 整数的话，就不用变为浮点数
        return (x)
    else:
        return ('{:.10f}'.format(x))