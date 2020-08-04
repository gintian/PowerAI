# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, layers
from tensorflow.keras.models import Model

from data.ghnet_data import GHData
from core.power import Power
from common.time_util import timer
from model.ghnet_model import MultiRegLossLayer
from model.ghnet_util import save_model, write_input, write_output


def build_mlp(inputs, shapes, name, dtype='float32',
              activation='relu', last_activation='relu'):
    """ 多层感知机网络。

    :param inputs: Input. 输入量，
    :param shapes: list[int]. 网络每层神经元数量，包含输出层。
    :param name: str. 网络名称。
    :param dtype: str. 数据类型。
    :param activation: str or activation func. 中间层激活函数。
    :param last_activation: str or activation func. 最后一层激活函数。
    :return: Model. MLP网络模型。
             [tf.Tensor]. MLP每层的Tensor，含输入层和输出层。
    """
    all_layers = [inputs]
    x = inputs
    for i in range(len(shapes) - 1):
        x = layers.Dense(shapes[i], activation=activation, dtype=dtype)(x)
        all_layers.append(x)
    y = layers.Dense(shapes[-1], activation=last_activation)(x)
    all_layers.append(y)
    model = Model(inputs, y, name=name)
    return model, all_layers


if __name__ == '__main__':
    path = os.path.join(os.path.expanduser('~'), 'data', 'wepri36', 'gen')
    net_path = os.path.join(path, 'net')
    res_type = 'cct'
    res_path = os.path.join(path, res_type)
    input_dic = {'generator': ['p'],
                 'load': ['p', 'q']}
    fmt = 'off'
    power = Power(fmt=fmt)
    power.load_power(net_path, fmt, lf=True, lp=False, st=False, station=True)
    input_layer = []
    for etype in input_dic:
        for dtype in input_dic[etype]:
            t = '_'.join((etype, dtype))
            input_layer.extend([(t, n) for n in power.data[etype]['name']])

    data_set = GHData(path, net_path, input_layer)
    data_set.load_x(x_ratio_thr=-1.0, dt_idx=False)
    data_set.load_y(res_type)
    data_set.normalize()
    data_set.column_valid = np.ones((data_set.input_data.shape[1], ), dtype=np.bool)
    """
    y_columns = list(range(data_set.y.shape[1]))
    column_names = data_set.y.columns[y_columns]
    print("targets:", column_names)

    prelu = tf.keras.layers.ReLU(negative_slope=0.1)
    x = Input(shape=(len(input_layer),), dtype='float32', name='x')
    fault = Input(shape=(len(y_columns),), dtype='float32', name='fault')
    y_ = Input(shape=(len(y_columns),), dtype='float32', name='y_')
    pre_model, all_layers = build_mlp(x, [64, 32, len(y_columns)], 'wepri36',
                                      activation=prelu, last_activation=prelu)
    feature_model = Model(x, all_layers[-2])
    loss = MultiRegLossLayer(name='multi_loss_layer')([pre_model.output, y_, fault])
    train_model = Model([x, fault, y_], loss)
    train_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=None)

    n_batch = 16
    n_epochs = 1000
    ids = data_set.split_dataset_random(ratios=[0.9, 0.05, 0.05])
    data_set.make_dataset_tensors(y_columns=y_columns, only_real=True)
    train_data, train_labels, _ = data_set.get_dataset(0)
    train_sample_size = train_labels.shape[0]
    steps_per_epoch = train_sample_size // n_batch
    val_data, val_labels, _ = data_set.get_dataset(1)
    val_sample_size = val_labels.shape[0]
    validation_steps = val_sample_size // n_batch
    train_gen = data_set.dataset_generator_multi(train_data, train_labels, n_batch)
    val_gen = data_set.dataset_generator_multi(val_data, val_labels, n_batch)
    with timer("Timer training"):
        history = train_model.fit_generator(train_gen,
                                            epochs=n_epochs,
                                            steps_per_epoch=steps_per_epoch,
                                            validation_data=val_gen,
                                            validation_steps=validation_steps)
    save_model(os.path.join(res_path, 'predict'), '', pre_model, suffix='tf')
    write_input(data_set, os.path.join(res_path, 'predict', 'input.txt'))
    write_output(data_set.y.columns[y_columns], os.path.join(res_path, 'predict', 'output.txt'))
    save_model(os.path.join(res_path, 'feature'), '', feature_model, suffix='tf')
    """
