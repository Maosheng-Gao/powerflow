"""
训练时，输出以归一化之后的数据进行训练，
测试直接使用完整结果预测。
"""
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
        termp = np.array(F * B + E * G, dtype='float32')
        self.a = a
        self.VKP = tf.constant(np.array(2 * E * I, dtype='float32').T)

        self.VKV = tf.constant(np.array(F * B + E * G, dtype='float32').T)
        self.VKSETA = tf.constant(np.array(2 * F * G - 2 * E * B, dtype='float32').T)
        self.VKQ = tf.constant(np.array(-F * I, dtype='float32').T)

        # 电压相角的卷积核
        self.SETAKSETA = tf.constant(np.array(N * G + M * B, dtype='float32').T)
        self.SETAKV = tf.constant(np.array(0.5 * N * B - 0.5 * M * G, dtype='float32').T)
        self.SETAKP = tf.constant(np.array(-M * I, dtype='float32').T)
        self.SETAKQ = tf.constant(np.array(-N * I, dtype='float32').T)


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
        output = tf.multiply(input_,  sgma) + mu
        output = tf.expand_dims(output, axis=-1)

        return output


class pooling(keras.layers.Layer):
    """
    根据系统的物理特性进行池化，
    也就是将PV节点的电压幅值固定，
    电压相角转化成跟平衡节点的相对角度
    """
    def __init__(self, activate_neurn, ref_bus, is_magnitude, **kwargs):
        super(pooling, self).__init__(**kwargs)
        # 固定输出层PV节点的电压电压幅值
        self.activate_neurn = tf.constant(activate_neurn, dtype='float32')
        self.factor = activate_neurn.copy()
        self.factor[self.factor > 0] = 1
        self.factor = tf.expand_dims(tf.constant(1 - self.factor, dtype='float32'), axis=0)
        self.is_magnitude = is_magnitude
        self.ref_bus = ref_bus

    def call(self, input_):
        if self.is_magnitude:
            # 将PV节点的电压幅值池化到固定值
            temp = input_*self.factor + self.activate_neurn
            return temp
        else:
            # 将电压相角转化成跟平衡节点的相对角度
            temp = input_ - tf.expand_dims(input_[:, self.ref_bus, :], axis=1)
            return temp


class graph_power_sys():
    def __init__(self, PV_V, G, B, ref, is_training=False):
        self.is_training = is_training
        self.PV_V = PV_V
        self.G = G
        self.B = B
        self.ref = ref
        self.build_net()

    def build_net(self):
        input_V = keras.Input(shape=(14, 1), name='input_V')
        input_Seta = keras.Input(shape=(14, 1), name='input_Seta')
        input_P = keras.Input(shape=(14, 1), name='input_P')
        input_Q = keras.Input(shape=(14, 1), name='input_Q')

        V1, Seta1 = self.forward_Block(input_V, input_Seta, input_P, input_Q, self.G, self.B, block='a', iteration=14)
        V2, Seta2 = self.forward_Block(V1, Seta1, input_P, input_Q, self.G, self.B, block='b', iteration=14)
        if self.is_training:
            V3, Seta3 = self.forward_Block(V2, Seta2, input_P, input_Q, self.G, self.B, block='output',
                                           iteration=4, is_last=True)
        else:
            V3, Seta3 = self.forward_Block(V2, Seta2, input_P, input_Q, self.G, self.B, block='output',
                                           iteration=4, is_last=False)

        self.model = keras.Model(inputs=[input_V, input_Seta, input_P, input_Q],
                            outputs=[V3, Seta3])
        optimizer = keras.optimizers.Adam(lr=0.0001, decay=1e-6)
        # optimizer = keras.optimizers.SGD(lr=0.001)
        self.model.compile(optimizer=optimizer,
                      loss='mse',
                      loss_weights={'output_V': 1.0,
                                    'output_Seta': 1.0})
        keras.utils.plot_model(self.model, 'step2.png', show_shapes=True)

    def forward_Block(self, input_V, input_Seta, input_P, input_Q, G, B, block, iteration=5, is_last=False):
        out_Seta_temp, out_v_temp = node_Conv_Layer(G=G, B=B, name=block + '_layer_0')\
            ([input_V, input_Seta, input_P, input_Q])
        out_v = out_v_temp
        out_Seta = out_Seta_temp
        for x in range(1, iteration):
            out_Seta_temp, out_v_temp = node_Conv_Layer(G=G, B=B, name=block + '_layer_' + str(x))\
                ([out_v_temp, out_Seta_temp,  input_P, input_Q])
            # 添加激活函数
            out_Seta_temp = pooling(self.PV_V, ref_bus=self.ref, is_magnitude=False)(out_Seta_temp)
            out_v_temp = pooling(self.PV_V, ref_bus=self.ref, is_magnitude=True)(out_v_temp)
            out_v = keras.layers.Concatenate(axis=-1)([out_v, out_v_temp])
            out_Seta = keras.layers.Concatenate(axis=-1)([out_Seta, out_Seta_temp])

        out = keras.layers.Concatenate(axis=-1)([out_Seta, out_v])

        # 添加BN归一化层
        out = keras.layers.BatchNormalization(axis=2)(out)

        l1 = Linear_Layer(num=100, activation='relu', name=block+'_l1')([out])
        l2 = Linear_Layer(num=100, activation='relu', name=block+'_l2')([l1])

        if not is_last:
            V = Linear_Layer(num=3, name=block + '_V0')([l2])
            Seta = Linear_Layer(num=3, name=block + '_Seta0')([l2])
            # 对归一化的数据进行还原， 然后进行pooling
            V = un_Ztrans(name=block + '_V1')(V)
            V = pooling(self.PV_V, ref_bus=self.ref, is_magnitude=True, name=block + '_V')(V)
            Seta = un_Ztrans(name=block + '_Seta1')(Seta)
            Seta = pooling(self.PV_V, ref_bus=self.ref, is_magnitude=False, name=block + '_Seta')(Seta)
        else:
            V = Linear_Layer(num=3, name=block + '_V')([l2])
            Seta = Linear_Layer(num=3, name=block + '_Seta')([l2])

        return V, Seta

    def train(self, dataGEN, model_dir):
        # 开始训练
        self.model.fit(dataGEN,
                      steps_per_epoch=1000,
                      epochs=100,
                      shuffle=True,
                      verbose=1)
        self.model.save(model_dir)

    def load_model(self, model_dir):
        self.model.load_weights(model_dir)

    def predict(self, input_V, input_Seta, input_P, input_Q):
        res_v, res_seta = self.model.predict([input_V, input_Seta, input_P, input_Q])
        return res_v, res_seta


def load_step_data(data_dir):
    data = np.load(data_dir, allow_pickle=True)
    G = data['G']
    B = data['B']
    input_P = data['input_P'][:, :, np.newaxis]
    input_Q = data['input_Q'][:, :, np.newaxis]

    input_V = data['input_V'][:, :, np.newaxis]**2
    input_Seta = data['input_Seta'][:, :, np.newaxis]
    output_V = data['output_V'][:, :, np.newaxis]**2
    output_Seta = data['output_Seta'][:, :, np.newaxis]
    # MU_V = np.mean(output_V)
    # sgma_V = np.std(output_V)
    # MU_seta = np.mean(output_Seta)
    # sgma_seta = np.std(output_Seta)

    return input_V, input_Seta, input_P, input_Q, G, B, output_V, output_Seta


def data_generator(input_V, input_Seta, input_P, input_Q, output_V, output_Seta, batch_size):
    data_v, mean_v, std_v = Z_Score(output_V)
    data_seta, mean_seta, std_seta = Z_Score(output_Seta)
    while True:
        idx = np.random.permutation(input_V.shape[0])
        for k in range(int(np.ceil(input_V.shape[0] / batch_size))):
            from_idx = k * batch_size
            to_idx = (k + 1) * batch_size
            index = idx[from_idx:to_idx]
            length_ = len(index)

            temp = output_V[index, :, :]
            data_ = (temp - mean_v) / std_v
            data_[np.isnan(data_)] = 0
            data_[np.isinf(data_)] = 0
            # data_, mean_, std_ = Z_Score(temp)
            mean_ = np.repeat(mean_v[np.newaxis, :, :], [length_], axis=0)
            std_ = np.repeat(std_v[np.newaxis, :, :], [length_], axis=0)
            output_z_v = np.concatenate([data_, mean_, std_], axis=2)

            # data_, mean_, std_ = Z_Score(output_Seta[index, :, :])
            temp = output_Seta[index, :, :]
            data_ = (temp - mean_seta) / std_seta
            data_[np.isnan(data_)] = 0
            data_[np.isinf(data_)] = 0
            mean_ = np.repeat(mean_seta[np.newaxis, :, :], [length_], axis=0)
            std_ = np.repeat(std_seta[np.newaxis, :, :], [length_], axis=0)
            output_z_seta = np.concatenate([data_, mean_, std_], axis=2)

            yield {'input_V': input_V[idx[from_idx:to_idx], :, :],
                   'input_Seta': input_Seta[idx[from_idx:to_idx], :, :],
                   'input_P': input_P[idx[from_idx:to_idx], :, :],
                   'input_Q': input_Q[idx[from_idx:to_idx], :, :]}, \
                  {'output_V': output_z_v,
                   'output_Seta': output_z_seta}


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


def Z_Score(trainData):
    trainData = np.array(trainData)
    mean_train = np.mean(trainData, axis=0)
    std_train = np.std(trainData, axis=0)
    std_train[std_train < 0.0001] = 0
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
        load_step_data('./sampleData/test1.npz')
    dataGEN = data_generator(input_V0, input_Seta0, input_P0, input_Q0, output_V0, output_Seta0, batch_size=30)

    # 计算PV节点的电压幅值
    from pypower.loadcase import loadcase
    from sampleData.utils import makePV_V
    ppc = loadcase('./pypower/case14')

    PV_V, ref = makePV_V(ppc)

    model = graph_power_sys(PV_V, G, B, ref, is_training=True)

    # model.load_model('./logs/model.h5')
    model.train(dataGEN, './logs/model.h5')
    # model.load_model('./logs/model.h5')
    # # 归一化处理
    # output_V0, mu_v, sgma_v = Z_Score(output_V0)
    # output_Seta0, mu_Seta, sgma_Seta = Z_Score(output_Seta0)

    # 验证集测试
    input_V, input_Seta, input_P, input_Q, G, B, output_V, output_Seta= \
        load_step_data('./sampleData/data1.npz')

    model = graph_power_sys(PV_V, G, B, ref, is_training=False)
    model.load_model('./logs/model.h5')
    res_v, res_seta = model.predict(input_V, input_Seta, input_P, input_Q)

    print(compute_acc(output_V, res_v, 0.0001, True))
    print(compute_acc(output_Seta, res_seta, 0.01, False))
