# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

import numpy as np

EPS = 1e-8


def set_values(data, etype, column, values, delta=False):
    """ 设置指定设备类型指定列的数据。

    :param data: dict of pd.DataFrame. 数据集合。
    :param etype: str. 设备类型。
    :param column: str. 列名。
    :param values: dict. 用于修改指定设备；
                    or np.array 全部修改的数值。
    :param delta: bool. False表示values为指定值；True表示values为变化量。
    """
    if etype not in data or column not in data[etype]:
        raise ValueError('Data incomplete! [%s, %s]' % (etype, column))
    if isinstance(values, dict):
        for k in values:
            if delta:
                data[etype].loc[k, column] += values
            else:
                data[etype].loc[k, column] = values
    else:
        if delta:
            data[etype][column] += values
        else:
            data[etype][column] = values


def distribute_generators_p(generators, delta, indices=None, sigma=None):
    """ 投运机组按比例承担总有功变化量，修改机组表的p0列。

    :param generators: pd.DataFrame. 机组数据表。
    :param delta: float. 总有功变化量（p.u.）。
    :param indices: list. 指定参与机组的索引列表；
                    or None. 全部投运机组参与。
    :param sigma: float. 随机变化率的方差，变化率~N(1.0, sigma)。
                    or None. 不随机变化。
    :return: float. 分配后的剩余功率，若为0.0代表全部分配完毕。
    """
    sub = generators[generators['mark'] == 1]
    if indices:
        indices = [i for i in indices if i in sub.index]
        sub = sub.loc[indices]
    if delta > 0.:
        sub = sub[sub['pmax'] > sub['p']]
        margin = sub['pmax'] - sub['p']
        margin_sum = np.sum(margin)
        if margin_sum <= delta:  # not enough
            generators.loc[sub.index, 'p0'] = sub['pmax']
            return delta - margin_sum
    else:
        sub = sub[sub['p'] > sub['pmin']]
        margin = sub['p'] - sub['pmin']
        margin_sum = np.sum(margin)
        if margin_sum <= -delta:  # not enough
            generators.loc[sub.index, 'p0'] = sub['pmin']
            return delta + margin
    if sigma:
        margin *= np.random.normal(loc=1.0, scale=sigma, size=(len(margin),))
        margin_sum = np.sum(margin)
    generators.loc[sub.index, 'p0'] = sub['p0'] + margin / margin_sum * delta
    return 0.


def distribute_loads_p(loads, delta, indices=None, p_sigma=None,
                       keep_factor=False, factor_sigma=None, clip_q0=True):
    """ 负荷按比例承担总有功变化量，修改负荷表的p0和q0列。

    :param loads: pd.DataFrame. 负荷数据表。
    :param delta: float. 总有功变化量（p.u.）。
    :param indices: list. 指定参与负荷的索引列表；
                    or None. 全部投运负荷参与。
    :param p_sigma: float. 随机有功变化率的方差，变化率~N(1.0, p_sigma)。
                    or None. 不随机变化。
    :param keep_factor: bool. 是否保持功率因子不变。
    :param factor_sigma: float. 随机功率因子变化率的方差，变化率~N(1.0, factor_sigma)。
                         or None. 不随机变化。
    :param clip_q0: bool. 是否保持无功在限值之内。
    """
    sub = loads[(loads['mark'] == 1) & (loads['p'] > 0.)]
    if indices:
        indices = [i for i in indices if i in sub.index]
        sub = sub.loc[indices]
    if keep_factor:
        factor = sub['q'] / sub['p']
        if factor_sigma:
            factor *= np.random.normal(loc=1.0, scale=factor_sigma, size=(len(factor),))
    ratio = sub['p']
    if p_sigma:
        ratio *= np.random.normal(loc=1.0, scale=p_sigma, size=(len(ratio),))
    loads.loc[sub.index, 'p0'] = sub['p0'] + ratio / np.sum(ratio) * delta
    if keep_factor:
        loads.loc[sub.index, 'q0'] = loads.loc[sub.index, 'p0'] * factor
        if clip_q0:
            loads['q0'] = np.clip(loads['q0'], loads['qmin'], loads['qmax'])


def random_load_q0(loads, sigma, clip=False):
    """ 修改负荷表的无功初值，修改负荷表q0列。

    :param loads: pd.DataFrame. 负荷数据表。
    :param sigma: float. 随机无功变化率的方差，变化率~N(1.0, sigma)；
                  or None. 在无功限值内以均匀分布进行采样。
    :param clip: bool. 是否保持无功在限值之内。
    """
    if sigma is not None:
        loads['q0'] *= np.random.normal(loc=1.0, scale=sigma, size=(loads.shape[0],))
    else:
        loads['q0'] = loads['qmin'] + \
                      np.random.rand(loads.shape[0]) * (loads['qmax'] - loads['qmin'])
    if clip:
        loads['q0'] = np.clip(loads['q0'], loads['qmin'], loads['qmax'])


def set_gl_p0(data, value, keep_factor=True, clip=True):
    """ 设置机组或负荷的有功初值，修改机组表或负荷表的p0.

    :param data: pd.DataFrame. 机组表或负荷表。
    :param value: np.array. 全部有功数值。
    :param keep_factor: bool. 是否保持功率因子不变。
    :param clip: bool. 是否保持有功和无功在限值以内。
    """
    if keep_factor:
        factor = data['q0'] / (data['p0'] + EPS)
    data['p0'] = value
    if keep_factor:
        data['q0'] = data['p0'] * factor
    if clip:
        data['p0'] = np.clip(data['p0'], data['pmin'], data['pmax'])
        data['q0'] = np.clip(data['q0'], data['qmin'], data['qmax'])


def full_open_generators(generators, indices, v0=None):
    """ 开机并满发。

    :param generators: pd.DataFrame. 机组数据表。
    :param indices: list. 指定机组的索引列表；
    :param v0: float or [float]. 设置电压初值；
               or None. 不修改电压初值。
    """
    pass


def close_all_branches(data):
    """ 闭合所有交流线和变压器支路。

    :param data: dict of pd.DataFrame. 数据集合。
    """
    pass


def random_open_acline(aclines, num, keep_link=True):
    """ 随机开断一定数量的交流线。

    :param aclines: pd.DataFrame. 交流线数据表。
    :param num: int. 开断数量。
    :param keep_link: bool. 是否保持连接状态，即不增加分岛。
    """
    pass
