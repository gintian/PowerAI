# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn

基于高级特征的分布和生成类
"""

import os
import shutil
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from core.power import Power
from model.sample.adg import ADG
from common.cmd_util import call_wmlf, check_lfcal
from model.ghnet_util import load_model, load_input, load_power_input, restore_power_input
from data.ghnet_data import GHData


class HighDG(ADG):
    def __init__(self, work_path, fmt, res_type, nn_size):
        super().__init__(work_path, fmt)
        self.mode = 'one'
        self.generated_num = 0
        self.nn_size = nn_size
        self.base_path = os.path.join(work_path, 'net')
        self.base_power = Power(fmt=fmt)
        self.base_power.load_power(self.base_path, fmt=fmt)
        self.input_fmt = load_input(os.path.join(work_path, 'cct', 'predict', 'input.txt'),
                                    input_mode=True)
        self.feature_model = load_model(os.path.join(work_path, res_type, 'feature'),
                                        '', suffix='tf')
        self.data_set = GHData(work_path, '', '')
        self.data_set.load_data(os.path.join(work_path, 'data_set.npz'))
        self.features = pd.DataFrame(
            self.feature_model.predict(self.data_set.input_data.values),
            index=self.data_set.input_data.index)
        self.init_assess = self.distribution_assess(nn_size)

    def distribution_assess(self, nn_size=-1):
        dists = pairwise_distances(self.features)
        if nn_size < 1:
            return np.average(dists) / np.max(dists)
        return np.average(np.partition(dists, nn_size + 1)[:, 1:nn_size + 1])

    def choose_samples(self, size=1):
        idx = np.arange(self.features.shape[0])
        np.random.shuffle(idx)
        return self.features.index[idx[:size]]

    def generate_one(self, power, idx, out_path):
        print(idx, out_path)
        if not power:
            self.input_fmt['value'] = self.data_set.ori_data.loc[idx].values
            power = restore_power_input(self.input_fmt, self.base_power, normalize=False)
            print('old=', self.input_fmt['value'])
        # TODO
        self.input_fmt['value'] *= np.random.normal(loc=1.0, scale=0.1,
                                                    size=self.input_fmt.shape[0])
        power = restore_power_input(self.input_fmt, power, normalize=False)
        print('new=', self.input_fmt['value'])
        shutil.rmtree(out_path, ignore_errors=True)
        power.save_power(out_path, fmt=self.fmt, lp=False)
        shutil.copy(os.path.join(self.base_path, 'LF.L0'), out_path)
        shutil.copy(os.path.join(self.base_path, 'ST.S0'), out_path)
        call_wmlf(out_path)
        ret = check_lfcal(out_path)
        if ret:
            idx = os.path.split(out_path)[-1]
            new_data = load_power_input(self.input_fmt, power)
            new_data = self.data_set.normalize_it(new_data)
            new_feature = self.feature_model.predict(new_data[np.newaxis, :])
            self.features.loc[idx] = new_feature.reshape(-1)
            self.generated_num += 1
        return ret

    def remove_samples(self):
        pass

    def done(self):
        assess = self.distribution_assess(self.nn_size)
        print('num=%d, init=%.6f, now=%.6f' % (self.generated_num, self.init_assess, assess))
        return self.generated_num > 0
