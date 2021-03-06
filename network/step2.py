# coding=utf-8
import h5py, os
import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime


class Linear_Layer(keras.layers.Layer):
    def __init__(self, num, activation="linear", **kwargs):
        super(Linear_Layer, self).__init__(**kwargs)
        self.activation_function = activation
        self.num = num

    def build(self, input_shape):
        # Weights
        self.W = self.add_weight("W_",
                                  shape=(int(input_shape[0][-1]), self.num),
                                  initializer='random_normal',
                                  trainable=True)
        self.bias = self.add_weight("bias",
                                    shape=[self.num],
                                    initializer='random_normal', trainable=True)

    def call(self, inputs):
        input_ = inputs[0]
        y0 = keras.backend.dot(input_, self.W) + self.bias
        if self.activation_function == 'relu':
            return keras.backend.relu(y0)
        elif self.activation_function == "linear":
            return y0
        else:
            return keras.backend.sigmoid(y0)


class node_Conv_Layer(keras.layers.Layer):
    def __init__(self, G, B,**kwargs):
        super(node_Conv_Layer, self).__init__(**kwargs)
        self.calculate_kernal(G, B)

    def calculate_kernal(self, G, B):
        """
        计算电压相角层和电压幅值层的卷积核
        :param G:
        :param B:
        :param P:
        :param Q:
        :return:
        """
        a = 3 * np.sum(G, axis=0) - np.diag(G)
        b = 3 * np.sum(B, axis=0) - np.diag(B)
        c = np.sum(G, axis=0) - np.diag(G)
        d = np.sum(B, axis=0) - np.diag(B)

        I = np.eye(G.shape[0])
        G = G - I * np.diag(G)
        B = B - I * np.diag(B)
        E = c / (a * c + b * d)
        F = d / (a * c + b * d)
        M = b / (a * c + b * d)
        N = a / (a * c + b * d)

        # 电压幅值的卷积核
        self.a = a
        self.VKP = tf.constant(np.array(2 * E * I, dtype='float32'))

        self.VKV = tf.constant(np.array(F * B + E * G, dtype='float32'))
        self.VKSETA = tf.constant(np.array(2 * F * G - 2 * E * B, dtype='float32'))
        self.VKQ = tf.constant(np.array(-F * I, dtype='float32'))

        # 电压相角的卷积核
        self.SETAKSETA = tf.constant(np.array(N * G + M * B, dtype='float32'))
        self.SETAKV = tf.constant(np.array(0.5 * N * B - 0.5 * M * G, dtype='float32'))
        self.SETAKP = tf.constant(np.array(-M * I, dtype='float32'))
        self.SETAKQ = tf.constant(np.array(-N * I, dtype='float32'))


    def call(self, inputs):
        V = tf.reshape(inputs[0], [-1, 1, 14])
        seta = tf.reshape(inputs[1], [-1, 1, 14])
        P = tf.reshape(inputs[2], [-1, 1, 14])
        Q = tf.reshape(inputs[3], [-1, 1, 14])

        out_v = keras.backend.dot(seta, self.VKSETA)+keras.backend.dot(V, self.VKV) + \
                keras.backend.dot(P, self.VKP) + keras.backend.dot(Q, self.VKQ)

        out_seta = keras.backend.dot(P, self.SETAKP) + keras.backend.dot(Q, self.SETAKQ) \
                   + keras.backend.dot(out_v, self.SETAKV)+keras.backend.dot(seta, self.SETAKSETA)

        return tf.reshape(out_seta, [-1, 14, 1]), tf.reshape(out_v, [-1, 14, 1])


class un_Ztrans(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(un_Ztrans, self).__init__(**kwargs)

    def call(self, inputs):
        # 输出三维，第一维为归一化的值，第二位为均值，第三维为标准差
        input_ = inputs[:, :, 0]
        mu = inputs[:, :, 1]
        sgma = inputs[:, :, 2]
        output = input_*sgma + mu

        return tf.reshape(output, [-1, 14, 1])


def forward_Block(input_V, input_Seta, input_P, input_Q, G, B, block, iteration=5):
    out_Seta_temp, out_v_temp = node_Conv_Layer(G=G, B=B, name=block + '_layer_0')\
        ([input_V, input_Seta, input_P, input_Q])
    out_v = out_v_temp
    out_Seta = out_Seta_temp
    for x in range(1, iteration):
        out_Seta_temp, out_v_temp = node_Conv_Layer(G=G, B=B, name=block + '_layer_' + str(x))\
            ([out_v_temp, out_Seta_temp,  input_P, input_Q])
        out_v = keras.layers.Concatenate(axis=-1)([out_v, out_v_temp])
        out_Seta = keras.layers.Concatenate(axis=-1)([out_Seta, out_Seta_temp])
    out = keras.layers.Concatenate(axis=-1)([out_Seta, out_v])

    l1 = Linear_Layer(num=100, activation='relu', name=block+'_l1')([out])
    l2 = Linear_Layer(num=100, activation='relu', name=block+'_l2')([l1])
    # 不归一化的输出
    V = Linear_Layer(num=1, activation='linear', name=block + '_V')([l2])
    Seta = Linear_Layer(num=1, activation='linear', name=block + '_Seta')([l2])

    # 输出归一化的参数
    # l3_v = Linear_Layer(num=3, activation='linear', name=block + 'l3_v')([l2])
    # l3_seta = Linear_Layer(num=3, activation='linear', name=block + '_l3_seta')([l2])
    #
    # V = un_Ztrans(name=block + '_V')(l3_v)
    # Seta = un_Ztrans(name=block + '_V')(l3_seta)
    return V, Seta


def load_step_data(data_dir):
    data = np.load(data_dir, allow_pickle=True)
    G = data['G']
    B = data['B']
    input_P = data['input_P'][:, :, np.newaxis]
    input_Q = data['input_Q'][:, :, np.newaxis]

    input_V = data['input_V'][:, :, np.newaxis]
    input_Seta = data['input_Seta'][:, :, np.newaxis]
    output_V = data['output_V'][:, :, np.newaxis]
    output_Seta = data['output_Seta'][:, :, np.newaxis]
    # MU_V = np.mean(output_V)
    # sgma_V = np.std(output_V)
    # MU_seta = np.mean(output_Seta)
    # sgma_seta = np.std(output_Seta)

    return input_V, input_Seta, input_P, input_Q, G, B, output_V, output_Seta


def data_generator(input_V, input_Seta, input_P, input_Q, output_V, output_Seta, batch_size):
    while True:
        idx = np.random.permutation(input_V.shape[0])
        for k in range(int(np.ceil(input_V.shape[0] / batch_size))):
            from_idx = k * batch_size
            to_idx = (k + 1) * batch_size
            yield {'input_V': input_V[idx[from_idx:to_idx], :, :],
                   'input_Seta': input_Seta[idx[from_idx:to_idx], :, :],
                   'input_P': input_P[idx[from_idx:to_idx], :, :],
                   'input_Q': input_Q[idx[from_idx:to_idx], :, :]}, \
                  {'output_V': output_V[idx[from_idx:to_idx], :, :],
                   'output_Seta': output_Seta[idx[from_idx:to_idx], :, :]}


def compute_acc(res, labels, tolerance, V=True):
    """
    calculate the accuracy of data with a given tolerance
    :param res: test data
    :param labels: desirable outcome
    :param tolerance: error tolerance
    :return: accuracy of res
    """
    if V:
        deta = np.abs(res - labels)
    else:
        deta = np.abs(res - labels)
    deta = deta[:, :, 0]
    deta[deta <= tolerance] = 1
    deta[deta != 1] = 0
    return np.mean(deta)


def calculate_kernal(G, B, P, Q):
    a = 3 * np.sum(G, axis=0) - np.diag(G)
    b = 3 * np.sum(B, axis=0) - np.diag(B)
    c = np.sum(G, axis=0) - np.diag(G)
    d = np.sum(B, axis=0) - np.diag(B)

    I = np.eye(G.shape[0])
    G = G - I * np.diag(G)
    B = B - I * np.diag(B)
    E = c / (a * c + b * d)
    F = d / (a * c + b * d)
    M = b / (a * c + b * d)
    N = a / (a * c + b * d)

    # 电压幅值的卷积核
    VKP = np.dot(2 * E * I, P)

    VKV = (F * B + E * G).T
    VKSETA = (2 * F * G - 2 * E * B).T
    VKQ = np.dot(-F * I, Q)

    # 电压相角的卷积核
    SETAKSETA = (N * G + M * B).T
    SETAKV = (0.5 * N * B - 0.5 * M * G).T
    SETAKP = np.dot(-M * I, P)
    SETAKQ = np.dot(-N * I, Q)

    return


def Z_Score(trainData):
    trainData = np.array(trainData)
    mean_train = np.mean(trainData, axis=0)
    std_train = np.std(trainData, axis=0)
    std_train[std_train<0.0001] = 0
    trainData = (trainData - mean_train)
    trainData = trainData/std_train
    trainData[np.isnan(trainData)] = 0
    trainData[np.isinf(trainData)] = 0
    return trainData, mean_train, std_train


def Z_Score_recover(trainData, mean_train, std_train):
    trainData = np.array(trainData)
    trainData = trainData*std_train
    trainData = trainData+mean_train
    return trainData


if __name__ == '__main__':
    # 准备训练数据
    input_V0, input_Seta0, input_P0, input_Q0, G, B, output_V0, output_Seta0 = \
        load_step_data('./sampleData/data1.npz')
    dataGEN = data_generator(input_V0, input_Seta0, input_P0, input_Q0, output_V0, output_Seta0, batch_size=20)

    # # 归一化处理
    # output_V0, mu_v, sgma_v = Z_Score(output_V0)
    # output_Seta0, mu_Seta, sgma_Seta = Z_Score(output_Seta0)

    # 搭建模型
    input_V = keras.Input(shape=(14, 1), name='input_V')
    input_Seta = keras.Input(shape=(14, 1), name='input_Seta')
    input_P = keras.Input(shape=(14, 1), name='input_P')
    input_Q = keras.Input(shape=(14, 1), name='input_Q')

    V1, Seta1 =forward_Block(input_V, input_Seta, input_P, input_Q, G, B, block='a', iteration=4)
    V2, Seta2 = forward_Block(V1, Seta1, input_P, input_Q, G, B, block='b', iteration=4)
    V3, Seta3 = forward_Block(V2, Seta2, input_P, input_Q, G, B, block='output', iteration=4)

    model = keras.Model(inputs=[input_V, input_Seta, input_P, input_Q],
                        outputs=[V3, Seta3])
    optimizer = keras.optimizers.Adam(lr=0.001, decay=1e-6)
    # optimizer = keras.optimizers.SGD(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='mse',
                  loss_weights={'output_V': 1.0,
                                'output_Seta': 1.0})
    keras.utils.plot_model(model, 'step2.png', show_shapes=True)

    # 开始训练
    model.load_weights("./logs/model.h5")
    # model.fit(dataGEN,
    #           steps_per_epoch=1000,
    #           epochs=10,
    #           shuffle=True,
    #           verbose=1)
    # model.save("./logs/model.h5")

    # 验证集测试
    input_V, input_Seta, input_P, input_Q, G, B, output_V, output_Seta= \
        load_step_data('./sampleData/test1.npz')

    res_v,  res_seta = model.predict([input_V, input_Seta, input_P, input_Q])

    # 还原数据
    # res_v = Z_Score_recover(res_v, mu_v, sgma_v)
    # res_seta = Z_Score_recover(res_seta, mu_Seta, sgma_Seta)

    print(compute_acc(output_V, res_v, 0.001, True))
    print(compute_acc(output_Seta, res_seta, 0.01, False))