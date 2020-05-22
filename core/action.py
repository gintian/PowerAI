# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

import numpy as np
import random

from core.power import Power
from core.topo import PowerGraph

EPS = 1e-8


def set_values(data, etype, column, values, delta=False):
    """ ����ָ���豸����ָ���е����ݡ�

    :param data: dict of pd.DataFrame. ���ݼ��ϡ�
    :param etype: str. �豸���͡�
    :param column: str. ������
    :param values: dict. �����޸�ָ���豸��
                    or np.array ȫ���޸ĵ���ֵ��
    :param delta: bool. False��ʾvaluesΪָ��ֵ��True��ʾvaluesΪ�仯����
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
    """ Ͷ�˻��鰴�����е����й��仯�����޸Ļ�����p0�С�

    :param generators: pd.DataFrame. �������ݱ�
    :param delta: float. ���й��仯����p.u.����
    :param indices: list. ָ���������������б�
                    or None. ȫ��Ͷ�˻�����롣
    :param sigma: float. ����仯�ʵķ���仯��~N(1.0, sigma)��
                    or None. ������仯��
    :return: float. ������ʣ�๦�ʣ���Ϊ0.0����ȫ��������ϡ�
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
    """ ���ɰ������е����й��仯�����޸ĸ��ɱ��p0��q0�С�

    :param loads: pd.DataFrame. �������ݱ�
    :param delta: float. ���й��仯����p.u.����
    :param indices: list. ָ�����븺�ɵ������б�
                    or None. ȫ��Ͷ�˸��ɲ��롣
    :param p_sigma: float. ����й��仯�ʵķ���仯��~N(1.0, p_sigma)��
                    or None. ������仯��
    :param keep_factor: bool. �Ƿ񱣳ֹ������Ӳ��䡣
    :param factor_sigma: float. ����������ӱ仯�ʵķ���仯��~N(1.0, factor_sigma)��
                         or None. ������仯��
    :param clip_q0: bool. �Ƿ񱣳��޹�����ֵ֮�ڡ�
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
    """ �޸ĸ��ɱ���޹���ֵ���޸ĸ��ɱ�q0�С�

    :param loads: pd.DataFrame. �������ݱ�
    :param sigma: float. ����޹��仯�ʵķ���仯��~N(1.0, sigma)��
                  or None. ���޹���ֵ���Ծ��ȷֲ����в�����
    :param clip: bool. �Ƿ񱣳��޹�����ֵ֮�ڡ�
    """
    if sigma is not None:
        loads['q0'] *= np.random.normal(loc=1.0, scale=sigma, size=(loads.shape[0],))
    else:
        loads['q0'] = loads['qmin'] + \
                      np.random.rand(loads.shape[0]) * (loads['qmax'] - loads['qmin'])
    if clip:
        loads['q0'] = np.clip(loads['q0'], loads['qmin'], loads['qmax'])


def set_gl_p0(data, value, keep_factor=True, clip=True):
    """ ���û���򸺺ɵ��й���ֵ���޸Ļ����򸺺ɱ��p0.

    :param data: pd.DataFrame. �����򸺺ɱ�
    :param value: np.array. ȫ���й���ֵ��
    :param keep_factor: bool. �Ƿ񱣳ֹ������Ӳ��䡣
    :param clip: bool. �Ƿ񱣳��й����޹�����ֵ���ڡ�
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
    """ ������������

    :param generators: pd.DataFrame. �������ݱ�
    :param indices: list. ָ������������б�
    :param v0: float or [float]. ���õ�ѹ��ֵ��
               or None. ���޸ĵ�ѹ��ֵ��
    """
    generators.loc[indices, 'mark'] = 1
    generators.loc[indices, 'p0'] = generators.loc[indices, 'pmax']
    if v0 is not None:
        generators.loc[indices, 'v0'] = v0


def close_all_branches(data):
    """ �պ����н����ߺͱ�ѹ��֧·��

    :param data: dict of pd.DataFrame. ���ݼ��ϡ�
    """
    data['acline']['mark'] = 1
    data['transformer']['mark'] = 1


def random_open_acline(power, num, keep_link=True):
    """ �������һ�������Ľ����ߡ�

    :param power: Power. Powerʵ������Ҫ�����ͨ�ԡ�
    :param num: int. ����������
    :param keep_link: bool. �Ƿ񱣳�����״̬���������ӷֵ���
    :return list[index]. ������·�������б�
    """
    ret = []
    aclines = power.data['acline']
    indices = aclines[(aclines['mark'] == 1) & (aclines['ibus'] != aclines['jbus'])].index
    if keep_link:
        graph = PowerGraph(power, graph_type='multi', node_type='bus', on_only=True)
    while num > 0:
        if len(indices) < num:
            raise ValueError('Not plenty of aclines to be off.')
        idx = indices[random.sample(range(len(indices)), 1)[0]]
        indices = indices.drop(idx)
        if keep_link:
            edge = (aclines.loc[idx, 'ibus'], aclines.loc[idx, 'jbus'], idx)
            if graph.is_connected(edge[0], edge[1], [edge]):
                continue
            graph.G.remove_edge(*edge)
        aclines.loc[idx, 'mark'] = 0
        ret.append(idx)
        num = num - 1
    return ret


if __name__ == '__main__':
    path = '../dataset/wepri36'
    fmt = 'off'
    power = Power(fmt)
    power.load_power(path, fmt=fmt, lp=False, st=False, station=True)
    random_open_acline(power, 2)