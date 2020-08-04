# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn

Active learning
"""

import numpy as np
import pandas as pd
import os

from model.sample.high_dg import HighDG

if __name__ == '__main__':

    path = os.path.join(os.path.expanduser('~'), 'data', 'wepri36', 'gen')
    dg = HighDG(path, fmt='off', res_type='cct', nn_size=5)
    while not dg.done():
        if dg.mode == 'one':
            indices = dg.choose_samples()
            for idx in indices:
                out_path = os.path.join(path, 'gen_%06d' % dg.generated_num)
                dg.generate_one(None, idx, out_path)
            dg.remove_samples()
        elif dg.mode == 'all':
            dg.generate_all()
        else:
            raise NotImplementedError
