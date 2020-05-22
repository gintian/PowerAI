# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

from core.power import Power
from core.yz import calc_ed_from_power


def test_calc_ed():
    path = '../dataset/wepri36'
    fmt = 'off'
    power = Power(fmt)
    power.load_power(path, fmt=fmt, lp=False, st=False)
    ed = calc_ed_from_power(power, island=0, node_type='bus', x_only=False)
    assert ed[1][1] == 0.
    assert ed[1][36] == 0.07283931941329119


# x = ed_map_tsne(ed)
# groups = group_kmeans(ed, 10)