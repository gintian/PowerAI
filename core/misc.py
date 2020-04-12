# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

import numpy as np


def call_parallel(x):
    xa = np.array(x)
    return 1./np.sum(1/xa)


def set_isin(sub, whole):
    return all([s in whole for s in sub])


def dict_list_append(data, key, value):
    if key in data:
        data[key].append(value)
    else:
        data[key] = [value]
