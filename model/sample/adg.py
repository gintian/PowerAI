# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn

Abstract distribution and generation class
"""


class ADG(object):
    def __init__(self, work_path, fmt):
        self.work_path = work_path
        self.fmt = fmt
        self.features = None
        self.mode = 'all'

    def distribution_assess(self):
        raise NotImplementedError

    def generate_all(self):
        raise NotImplementedError

    def choose_samples(self, size):
        raise NotImplementedError

    def generate_one(self, power, idx, out_path):
        raise NotImplementedError

    def remove_samples(self):
        raise NotImplementedError

    def done(self):
        raise NotImplementedError
