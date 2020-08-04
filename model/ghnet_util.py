# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import matplotlib.pyplot as plt

from core.power import Power
from data.ghnet_data import mm_normalize, mm_denormalize
from core.action import set_values


def write_input(data_set, file_name):
    """
    Write input infos to file, including type, elem, min, max

    :param data_set: GHData. Use its input_layer and column_*.
    :param file_name: str.
    """
    with open(file_name, "w") as f:
        for i, (t, elem) in enumerate(data_set.input_layer):
            if data_set.column_valid[i]:
                f.write("%s %s %f %f\n" % (t, elem,
                                           data_set.column_min[i],
                                           data_set.column_max[i]))


def load_input(file_name, input_mode=False):
    input_fmt = pd.read_table(file_name, names=['etype', 'name', 'min', 'max'],
                              encoding='gbk', index_col=False, sep=' ')
    input_fmt['dtype'] = ''
    has_sub = input_fmt['etype'].str.contains('_')
    ss = input_fmt.loc[has_sub, 'etype'].str.split('_')
    ss = pd.DataFrame(ss.values.tolist(), index=ss.index, columns=['etype', 'dtype'])
    input_fmt.loc[has_sub, ['etype', 'dtype']] = ss
    if input_mode:
        input_fmt.loc[input_fmt['dtype'] == 'p', 'dtype'] = 'p0'
        input_fmt.loc[input_fmt['dtype'] == 'q', 'dtype'] = 'q0'
    return input_fmt


def write_output(targets, file_name):
    """
    Write output (predictions) to file.

    :param targets: iterable.
    :param file_name: str.
    """
    with open(file_name, "w") as f:
        f.write('\n'.join(targets))


def write_adjust(adjusts, file_name):
    """
    Write adjust infos to file.

    :param adjusts: {elem: (value, delta)}.
    :param file_name: str.
    :return:
    """
    with open(file_name, "w") as f:
        for k, (v, dv) in adjusts.items():
            f.write("%s %.4f %.4f\n" % (k, v, dv))


def save_model(path, name, model, suffix='tf'):
    """
    Save model for h5/json/pb/tf/frozen

    :param path: str.
    :param name: str.
    :param model: tf.keras.models.Model.
    :param suffix: str. For different format.
    """
    if not os.path.exists(path):
        os.mkdir(path)
    if suffix == 'h5':
        model.save(path + "/" + name + ".h5")
        # tf.keras.models.save_model(model, path, save_format='h5')
    elif suffix == 'json':
        model_json = model.to_json()
        with open(path + "/" + name + ".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(path + "/" + name + ".h5")
    elif suffix == 'pb' or suffix == 'tf':
        tf.keras.models.save_model(model, path, save_format='tf')
    elif suffix == 'frozen':
        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(
            x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                          logdir=path,
                          name=name + '.pb',
                          as_text=False)
    else:
        raise TypeError('suffix=\'%s\'' % suffix)


def load_model(path, name, suffix='tf'):
    """
    Load model from h5/json/pb/tf/frozen

    :param path: str.
    :param name: str.
    :param suffix: str. For different format.
    :return: tf.keras.models.Model.
    """
    model = None
    if suffix == 'h5':
        model = tf.keras.models.load_model(path + "/" + name + ".h5")
    elif suffix == 'json':
        json_file = open(path + "/" + name + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = tf.keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(path + "/" + name + ".h5")
    elif suffix == 'pb' or suffix == 'tf':
        model = tf.keras.models.load_model(path)
    elif suffix == 'frozen':
        raise NotImplementedError('suffix=\'%s\'' % suffix)
    else:
        raise TypeError('suffix=\'%s\'' % suffix)
    return model


def dataset_predict(model, data_set, role):
    """
    Predict for data_set.

    :param model: tf.keras.models.Model. Pre_model.
    :param data_set: GHData.
    :param role: int. Data set id.
    :return: tuple (labels, predictions). Labels and predictions of the same shape.
    :raise: ValueError.
    """
    (data, labels, _) = data_set.get_dataset(role)
    if data is None or labels is None:
        raise ValueError('no data')
    pre = model.predict(data)
    return labels, pre


def save_data_plt(data, path, y=None):
    """
    Save curve for each column of data and y, [data[i], y]

    :param data: DataFrame.
    :param path: str.
    :param y: 1D np.array.
    """
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(data.shape[1]):
        plt.plot(np.arange(data.shape[0]), data.iloc[:, i])
        if y is not None:
            plt.plot(np.arange(y.shape[0]), y)
        # plt.ylim((-1, 1))
        plt.title(data.columns[i], fontproperties="SimHei")
        plt.savefig(path + "/%d.jpg" % i)
        plt.close()


def plt_loss(history):
    """
    Plot loss curve.

    :param history: Keras history infos for training.
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Testing loss')
    plt.title('Training and Testing loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()


def load_power_input(input_fmt, power_or_path, fmt='on', normalize=True):
    if isinstance(power_or_path, str):
        power = Power(fmt=fmt)
        power.load_power(power_or_path, fmt=fmt)
        power.extend_lp_info(real_only=False)
    else:
        power = power_or_path
    values = np.empty(input_fmt.shape[0], dtype=np.float32) + np.nan
    for i, (idx, etype, dtype, name) in \
            enumerate(input_fmt[['etype', 'dtype', 'name']].itertuples()):
        if etype == 'ed':
            # TODO: deal with ed
            continue
        else:
            try:
                v = power.data[etype].loc[power.data[etype]['name'] == name, dtype]
                if len(v) > 0:
                    values[i] = v.values[0]
                # values.append(v.values[0] if len(v) > 0 else np.nan)
            except KeyError:
                # print(etype, dtype, name)
                # values.append(np.nan)
                continue
    if normalize:
        values = mm_normalize(values, input_fmt['min'].values, input_fmt['max'].values)
    values[np.isnan(values)] = -1.
    return values


def restore_power_input(input_fmt, power_or_path, fmt='on', normalize=True):
    if isinstance(power_or_path, str):
        power = Power(fmt=fmt)
        power.load_power(power_or_path, fmt=fmt)
        power.extend_lp_info(real_only=False)
    else:
        power = power_or_path
    if normalize:
        input_fmt['value'] = mm_denormalize(input_fmt['value'].values,
                                            input_fmt['min'].values, input_fmt['max'].values)
    group = input_fmt.groupby(['etype', 'dtype'], sort=False)
    for (e, d), idx in group.indices.items():
        values = power.data[e][[d, 'name']].copy()
        indices = values.index
        values.set_index('name', inplace=True)
        values = input_fmt.loc[idx, ['name', 'value']].set_index('name')
        values.index = indices
        set_values(power.data, e, d, values['value'])
    return power


if __name__ == '__main__':
    input_file = 'C:/Users/sdy/data/db/2018_11/cct/input.txt'
    data_path = 'C:/Users/sdy/data/db/2018_11/net'
    ret = load_power_input(input_file, data_path)
