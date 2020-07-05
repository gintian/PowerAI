# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

import numpy as np
import pandas as pd
import warnings
import os

from common.time_util import timer

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

EPS = 1e-8


def generate_y_matrix(power, island, node_type='bus', dtype=np.float32,
                      on_only=True, ignore_ground_branch=True, ignore_tk=True, x_only=True,
                      ignore_nodes=None, diag_coef=1.01):
    """ ��Powerʵ�����ɵ�����Y��

    :param power: Power. ��ȡLF�ļ����Powerʵ����
    :param island: int. �������š�
    :param node_type: str. 'bus'��ʾ��ĸ��Ϊ��Ԫ��'station'��ʾ�Գ�վΪ��Ԫ��
    :param dtype: dtype. �γɾ���ʱ�ĸ��������ͣ�Ĭ��Ϊfloat32.
    :param on_only: bool. ֻ����Ͷ��֧·��
    :param ignore_ground_branch: bool. �����ݿ���֧·��
    :param ignore_tk: bool. ���Ա�ѹ���ķǱ�׼���
    :param x_only: bool. ֻ���ǵ翹�����Ե��衣
    :param ignore_nodes: list. ���Խڵ������������ƽ��ڵ㡣
    :param diag_coef: float. �Խ���ϵ����������󲻿��档
    :return: pd.DataFrame. Y����
    """
    if node_type == 'bus':
        if 'island' not in power.data['bus'].columns:
            raise ValueError("Island info not generated.")
        nodes = power.data['bus'].index[power.data['bus'].island == island].to_list()
        columns = ['mark', 'ibus', 'jbus', 'r', 'x', 'no']
        columns_tran = ['mark', 'ibus', 'jbus', 'r', 'x', 'tk']
        branches = np.vstack((power.data['acline'][columns].values,
                              power.data['transformer'][columns_tran].values)).astype(dtype)
    elif node_type == 'station':
        if 'island' not in power.stations.columns:
            raise ValueError("Island info not generated.")
        nodes = power.stations.index[power.stations.island == island].to_list()
        columns = ['mark', 'st_i', 'st_j', 'r', 'x']
        branches = power.data['acline'][columns].values.astype(dtype)
    else:
        return None
    valid = np.isin(branches[:, [1, 2]], nodes).all(axis=1)
    branches = branches[valid]
    if on_only:
        branches = branches[branches[:, 0] == 1.]
    if ignore_ground_branch:
        branches = branches[branches[:, 1] != branches[:, 2]]
    else:
        branch = branches
        branches = branch[branch[:, 1] != branch[:, 2]]
        ground_branch = branch[branch[:, 1] == branch[:, 2]]
        # raise NotImplementedError("Ground branches are not considered.")
    # valid = np.isin(branches[:, [1, 2]], nodes).all(axis=1)
    # branches = branches[valid]

    if branches.shape[0] == 0:
        raise ValueError(
            "Branches is empty, while node size = %d" %
            len(nodes))

    branches = branches.copy()
    nodes = list(set(branches[:, [1, 2]].astype(np.int32).flatten()))
    n = len(nodes)
    node_idx = pd.Series(data=range(n), index=nodes)
    branches[:, 1] = node_idx.loc[branches[:, 1]].values
    branches[:, 2] = node_idx.loc[branches[:, 2]].values
    if x_only:
        gb = -1. / (branches[:, 4] + EPS)  # jb = -1/jx
        y = np.zeros((n, n), dtype=dtype)
    else:
        gb = np.vectorize(np.complex)(branches[:, 3].astype(dtype),
                                      branches[:, 4].astype(dtype))
        gb = 1. / (gb + EPS)  # g+jb=1/(r+jx)
        y = np.zeros((n, n), dtype=gb.dtype)
    for i, [ii, jj] in enumerate(branches[:, [1, 2]].astype(np.int32)):
        y[ii][jj] = y[jj][ii] = y[ii][jj] - gb[i]
        # y[i, j] = -(g+jb) in non-diagonal element
    for i in range(y.shape[0]):
        y[i, i] = 0.0
        y[i, i] = -np.sum(y[i,]) * diag_coef
    if not ignore_tk:
        tran_branches = branches[branches[:, 5] < 2]
        if x_only:
            gbt = -1. / (tran_branches[:, 4] + EPS)  # jb = -1/jx
        else:
            gbt = np.vectorize(np.complex)(tran_branches[:, 3].astype(dtype),
                                            tran_branches[:, 4].astype(dtype))
            gbt = 1. / (gbt + EPS)  # g+jb=1/(r+jx)
        tk = tran_branches[:, 5].astype(dtype)
        for i, [ii, jj] in enumerate(tran_branches[:, [1, 2]].astype(np.int32)):
            y[ii][jj] = y[jj][ii] = y[ii][jj] + gbt[i] - gbt[i] / tk[i]
            y[jj][jj] = y[jj][jj] - gbt[i] + gbt[i] / (tk[i] * tk[i])
    if not ignore_ground_branch:
        ground_branch[:, 1] = node_idx.loc[ground_branch[:, 1]].values
        if x_only:
            grb = -1. / (ground_branch[:, 4] + EPS)
        else:
            grb = np.vectorize(np.complex)(ground_branch[:, 3].astype(dtype),
                                           ground_branch[:, 4].astype(dtype))
            grb = 1. / (grb + EPS)
        for i, ii in enumerate(ground_branch[:, 1].astype(np.int32)):
            y[ii][ii] = y[ii][ii] + grb[i]

    # y[0][0] = y[0][0] * diag_coef
    y = pd.DataFrame(data=y, index=nodes, columns=nodes)
    if ignore_nodes:
        nodes = [n for n in nodes if n not in ignore_nodes]
        y = y.loc[nodes, nodes]
    return y


def calc_z_from_y(y):
    """ ͨ��Y��������õ�Z����

    :param y: pd.DataFrame or np.array. Y����
    :return: pd.DataFrame or np.array. Z����
    """
    if not isinstance(y, pd.DataFrame) and not isinstance(y, np.array):
        raise NotImplementedError("%s not supported." % type(y))
    if y.shape[0] > 1000:
        warnings.warn("It would be time-consuming: y.shape=%d." % y.shape[0])
    z = np.linalg.inv(y)
    if isinstance(y, pd.DataFrame):
        return pd.DataFrame(data=z, index=y.index, columns=y.columns)
    return z


def calc_ed_from_z(z, indices=None):
    """ ��Z������ȡ�������롣

    :param z: pd.DataFrame. Z����
    :param indices: 1D list. ��վ�б�
                    or 2D list. ��վ���б�
                    or None. ����ȫ����վ��
    :return: pd.DataFrame. ��indicesΪ1D list��Noneʱ������DataFrame��
             np.array. ��indicesΪ2D listʱ��������������
    """
    if indices is None:
        nodes = z.index.to_list()
        zz = z.values
    else:
        indices = np.array(indices)
        nodes = list(set(indices.flatten()))
        node_idx = pd.Series(data=range(len(nodes)), index=nodes)
        zz = z.loc[nodes, nodes].values
    if len(nodes) == 0:
        raise ValueError("Node set is empty.")
    elif len(nodes) > 1000:
        warnings.warn("It would be time-consuming: node size= %d." % len(nodes))

    ed = np.zeros((zz.shape[0], zz.shape[0]))
    for i in range(0, zz.shape[0] - 1):
        j = np.arange(i + 1, zz.shape[0])
        ed[i, j] = np.vectorize(np.abs)(zz[i, i] + zz[j, j] - zz[i, j] - zz[j, i])
        # ed[j, i] = ed[i, j]
    ed = ed + ed.T
    if indices is None or indices.ndim == 1:
        return pd.DataFrame(data=ed, index=nodes, columns=nodes)
    elif indices.ndim == 2:
        assert indices.shape[1] == 2
        i = node_idx.loc[indices[:, 0]].values
        j = node_idx.loc[indices[:, 1]].values
        return ed[i, j]
    return None


def calc_ed_from_power(power, island, node_type='bus', dtype=np.float32,
                       on_only=True, ignore_ground_branch=True, ignore_tk=True, x_only=True,
                       indices=None):
    """ ��Powerʵ����ȡ�������롣��

    :param power: Power. ��ȡLF�ļ����Powerʵ����
    :param island: int. �������š�
    :param node_type: str. 'bus'��ʾ��ĸ��Ϊ��Ԫ��'station'��ʾ�Գ�վΪ��Ԫ��
    :param dtype: dtype. �γɾ���ʱ�ĸ��������ͣ�Ĭ��Ϊfloat32.
    :param on_only: bool. ֻ����Ͷ��֧·��
    :param ignore_ground_branch: bool. �����ݿ���֧·��
    :param ignore_tk: bool. ���Ա�ѹ���ķǱ�׼���
    :param x_only: bool. ֻ���ǵ翹�����Ե��衣
    :param indices: 1D list. ��վ�б�
                    or 2D list. ��վ���б�
                    or None. ����ȫ����վ��
    :return: pd.DataFrame. ��indicesΪ1D list��Noneʱ������DataFrame��
             np.array. ��indicesΪ2D listʱ��������������
    """
    y = generate_y_matrix(power, island, node_type=node_type, dtype=dtype,
                          on_only=on_only, ignore_ground_branch=ignore_ground_branch,
                          ignore_tk=ignore_tk, x_only=x_only)
    z = calc_z_from_y(y)
    return calc_ed_from_z(z, indices)


def ed_map_tsne(ed, n_dim=2, **kwargs):
    """ ͨ��t-SNE�㷨�Ե��������ϵ���н�ά��

    :param ed: pd.DataFrame. �����������
    :param n_dim: int. ��ά��ά�ȡ�
    :param kwargs: ����������
    :return: np.array. ��ά�����ꡣ
    """
    assert isinstance(ed, pd.DataFrame) and ed.shape[0] == ed.shape[1]
    tsne = TSNE(n_components=n_dim, metric='precomputed', **kwargs)
    x = tsne.fit_transform(ed.values)
    x -= np.median(x, axis=0)
    return x


def group_kmeans(ed, n):
    """ ���ڵ��������ϵ��ͨ��kmeans�������з��顣

    :param ed: pd.DataFrame. �����������
    :param n: int. ��������
    :return: dict. �����б��ֵ䡣
    """
    assert isinstance(ed, pd.DataFrame) and ed.shape[0] == ed.shape[1]
    kmeans = KMeans(n_clusters=n)
    clf = kmeans.fit(ed.values)
    groups = {}
    for i in range(n):
        groups[i] = ed.index[np.where(clf.labels_ == i)[0]].to_list()
    return groups


def load_n1_ed(path):
    """ ��path/n1_*��ȡȫ������N-1��ĵ����������

    :param path: std. ����Ŀ¼���������� n1_*.txt �ļ���ÿ���ļ���¼һ�������߿��Ϻ�ȫ����Ҫ��վ��ĵ������룻
                              ���� idx_name.txt �ļ�����¼��վ���������š�
    :return: pd.DataFrame. �����������IndexΪ"i�೧վ��_j�೧վ��"
    """
    subs = []
    files = os.listdir(path)
    for i, file_name in enumerate(files):
        if i % 10 == 0:
            print("%d / %d" % (i, len(files)))
        if not file_name.startswith('n1_'):
            continue
        sub = pd.read_table(os.path.join(path, file_name),
                            sep=' ', encoding='gbk', index_col=[0])
        subs.append(sub)
    df = pd.concat(subs, axis=1)
    st_names = pd.read_table(os.path.join(path, 'idx_name.txt'),
                             sep=' ', encoding='gbk', index_col=[0])
    new_idx = []
    for idx in df.index:
        i, j = int(idx.split('_')[0]), int(idx.split('_')[1])
        new_idx.append('_'.join([st_names['name'][i], st_names['name'][j]]))
    df.index = new_idx
    df = df[(df < 10.).all(axis=1)]
    np.savez(os.path.join(path, 'n1_ed.npz'), index=df.index,
             columns=df.columns, datas=df.values)
    return df


def minset_greedy(df, thr=0.05):
    """ ����̰���㷨����ȡ���Էֱ�����N-1����ĵ���������С�Ӽ���

    :param df: pd.DataFrame. �����������
    :param thr: float. �ֱ�����ֵ��
    :return: Index. ����������������б�
    """
    ret = []
    delta = df.values
    delta = np.abs(delta - delta[:, 0:1]) / (delta + 1e-8)  # ��0��Ϊȫ����״̬�µĵ�������
    delta = (delta[:, 1:] > thr)
    covered = np.zeros((delta.shape[1],), dtype=np.bool_)
    while True:
        remain_sum = np.sum(delta[:, ~covered], axis=1)
        i = remain_sum.argmax()
        if remain_sum[i] == 0:
            break
        # print(df.index[i], np.sum(delta[i,~covered]), np.sum(covered))
        ret.append(i)
        covered += delta[i, :]
    return df.index[ret]


def calc_gsdf(power, island, branches, alpha='single', node_type='bus'):
    """ ����ڵ��й���֧·�й��ķֲ�ϵ��

    :param power: Power. Power����ʾ����
    :param island: int. ���š�
    :param branches: dict. {'acline': [index...], 'transformer': [index...]}
    :param alpha: str. 'single'��ʾƽ��������е��������˻�Ϊ�����GSDF��
    :param node_type: str. 'bus'��ʾ��ĸ��Ϊ��Ԫ��ģ��'station'��ʾ�Գ�վΪ��Ԫ��ģ��
    :return: pd.DataFrame. �����Ⱦ���ÿ�д���һ��֧·��ÿ�д���һ�����飨��ƽ�������
    """
    generators = power.data['generator']
    slack = generators[generators['type'] == 0]['bus']
    slack = power.data['bus'].loc[slack]
    slack = slack[slack['island'] == island].index[0]
    gens = generators[generators['mark'] == 1]['bus']
    gens = power.data['bus'].loc[gens]
    gens = gens[gens['island'] == island].index.to_list()
    if node_type == 'station':
        slack = power.data['bus'].loc[slack, 'st_no']
        gens = power.data['bus'].loc[gens, 'st_no'].drop_duplicates().values.tolist()
    gens.remove(slack)
    y = generate_y_matrix(power, island, node_type=node_type,
                          ignore_nodes=[slack], diag_coef=1.0)
    x = calc_z_from_y(y)

    def gk_one_line(x_, i_, j_, slack_):
        if i_ == slack_:
            return -x_.loc[j_]
        elif j_ == slack_:
            return x_.loc[i_]
        else:
            return x_.loc[i_] - x_.loc[j_]

    # n_branch = len(branches.get('acline', [])) + len(branches.get('transformer', []))
    indices = branches.get('acline', []) + branches.get('transformer', [])
    gk = np.zeros((len(indices), y.shape[1]), dtype=np.float32)
    columns = ['st_i', 'st_j', 'x'] if node_type == 'station' else ['ibus', 'jbus', 'x']
    ii = 0
    for ii, idx in enumerate(branches.get('acline', [])):
        i, j, xk = power.data['acline'].loc[idx, columns]
        if i == j:
            print('acline[' + idx + '] has the same i / j', i, j)
        else:
            gk[ii, :] = gk_one_line(x, i, j, slack) / xk
    columns = ['ibus', 'jbus', 'x', 'tk']
    for ii, idx in enumerate(branches.get('transformer', []), ii+1):
        if node_type == 'station':
            print('Transformer is ignored in station mode.')
            break;
        i, j, xk, tk = power.data['transformer'].loc[idx, columns]
        gk[ii, :] = gk_one_line(x, i, j, slack) / xk
    gk = pd.DataFrame(gk, index=indices, columns=x.columns)[gens]
    gk[slack] = 0.

    if alpha == 'single':
        alpha = pd.Series(0., index=gk.columns, dtype=np.float32)
        alpha.loc[slack] = 1.
    else:
        raise NotImplementedError
    n_gen = len(gk.columns)
    fu = np.eye(n_gen) - np.dot(alpha.values[:, np.newaxis], np.ones((1, n_gen)))
    gkr = np.dot(gk, fu)
    gkr = pd.DataFrame(gkr, index=gk.index, columns=gk.columns)
    return gkr


def calc_vps(power, island, buses, node_type='station'):
    """ ����ڵ��޹��Խڵ��ѹ��������

    :param power: Power. Power����ʾ����
    :param island: int. ���š�
    :param buses: dict. {'node': [index...]}
    :param node_type: str. 'bus'��ʾ��ĸ��Ϊ��Ԫ��ģ��'station'��ʾ�Գ�վΪ��Ԫ��ģ��
    :return: pd.DataFrame. �����Ⱦ���ÿ�д���һ���ڵ㣬ÿ�д���һ���������������ڵ㡣
    """
    generators = power.data['generator']
    slack = generators[generators['type'] == 0]['bus']
    slack = power.data['bus'].loc[slack]
    slack = slack[slack['island'] == island].index[0]
    gens = generators['bus']
    gens = power.data['bus'].loc[gens]
    gens = gens[gens['island'] == island].index.to_list()
    branches = power.data['acline']
    branches = branches[branches['ibus'] == branches['jbus']]
    brans = branches['ibus'].drop_duplicates()
    brans = power.data['bus'].loc[brans]
    brans = brans[brans['island'] == island].index.to_list()
    nodes = power.data['bus'].index[power.data['bus'].island == island].to_list()
    indices = buses.get('bus', [])
    if node_type == 'station':
        slack = power.data['bus'].loc[slack, 'st_no']
        gens = power.data['bus'].loc[gens, 'st_no'].drop_duplicates().values.tolist()
        brans = power.data['bus'].loc[brans, 'st_no'].drop_duplicates().values.tolist()
        nodes = power.data['bus'].loc[nodes, 'st_no'].drop_duplicates().values.tolist()
        indices = power.data['bus'].loc[indices, 'st_no'].values.tolist()
    y = generate_y_matrix(power, island, node_type=node_type, on_only=True,
                          ignore_ground_branch=False, ignore_tk=False, x_only=False,
                          ignore_nodes=[slack], diag_coef=1.00)
    gens.remove(slack)
    nodes.remove(slack)
    if np.isin(slack, brans):
        brans.remove(slack)
    gens = list(set(gens).intersection(set(y.index.tolist())))
    brans = list(set(brans).intersection(set(y.index.tolist())))
    nodes = list(set(nodes).intersection(set(y.index.tolist())))
    no_gen = list(set(gens + brans))
    no_load = list(set(nodes) - set(no_gen))
    no_all = no_load + no_gen
    yp = y.reindex(index=no_all, columns=no_all)
    ya = yp.values
    yab = np.imag(ya)
    ypb = pd.DataFrame(data=yab, index=no_all, columns=no_all)
    rypb = -calc_z_from_y(ypb)
    RDG = rypb.loc[no_load, no_gen]
    RGG = rypb.loc[no_gen, no_gen]
    gk = np.zeros((len(indices), len(no_gen)), dtype=np.float32)
    for ii, idx in enumerate(indices):
        if np.isin(idx, no_load):
            gk[ii, :] = RDG.loc[idx]
        elif np.isin(idx, no_gen):
            gk[ii, :] = RGG.loc[idx]
        else:
            print(idx, "is slackbus")

    gk = pd.DataFrame(gk, index=indices, columns=no_gen)
    return gk


if __name__ == '__main__':
    from core.power import Power

    # path = '../dataset/wepri36'
    # fmt = 'off'
    path = 'C:/Users/sdy/data/db/SA_2'
    fmt = 'on'
    power = Power(fmt)
    with timer('gsdf'):
        power.load_power(path, fmt=fmt, lp=False, st=False)
        # ed = calc_ed_from_power(power, island=0, node_type='bus', x_only=False)
        # gkr = calc_gsdf(power, 0, {'acline':[29, 44]})
        # gkr = calc_gsdf(power, 0, {'acline': [10547, 10317, 10373, 11136],
        #                            'transformer': [51188]})
        vps = calc_vps(power, 0, {'bus': [878, 914]}, 'bus')

