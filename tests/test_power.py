# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

import os
from core.power import Power


def test_power(real_data=False):
    # set_format_version({'mdc': 2.3})
    path = 'dataset/wepri36'
    fmt = 'off'
    power = Power(fmt)
    power.load_power(path, fmt=fmt)
    if real_data:
        path = os.path.join(os.path.expanduser('~'), 'data', 'gd', '2019_12_23T12_15_00')
        fmt = 'on'
        power = Power(fmt)
        power.load_power(path, fmt=fmt)
        power.save_power(path + '/out', fmt=fmt)
        path = 'C:/PSASP_Pro/2020国调年度/冬低731'
        fmt = 'off'
        power = Power(fmt)
        power.load_power(path, fmt=fmt, lp=False)
        power.save_power(path + '/out', fmt=fmt, lp=False)


if __name__ == '__main__':
    test_power(real_data=True)
