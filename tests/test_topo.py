# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

from core.power import Power
from core.topo import PowerGraph


def test_topo():
    path = '../dataset/wepri36'
    fmt = 'off'
    power = Power(fmt)
    power.load_power(path, fmt=fmt, lp=False, st=False, station=True)
    graph1 = PowerGraph(power, graph_type='single', node_type='station', on_only=True)
    islands1 = graph1.get_islands(min_num=5)
    print(islands1)
    graph2 = PowerGraph(power, graph_type='multi', node_type='bus',
                         on_only=False, edge_columns=['x'])
    islands2 = graph2.get_islands(min_num=10)
    print(islands2)


if __name__ == '__main__':
    test_topo()