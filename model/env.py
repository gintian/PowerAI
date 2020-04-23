# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt

from core.power import Power
from core.action import distribute_generators_p, distribute_loads_p, \
    set_gl_p0, random_load_q0
from core.misc import dict_list_append
from common.cmd_util import call_wmlf, call_psa, check_lfcal
from common.efile_util import read_efile, update_table_header


EPS = 1e-8
LOAD_CHANGE_THR = 0.5
PF_NOTCONV_REWARD = -1.2
PF_LOADFULL_REWARD = +1.0
ST_UNSTABLE_REWARD = -1.0
CCT_CHANGE_RATIO = 10.


class OpEnv(object):
    def __init__(self, base_path, work_path, inputs, fmt='on'):
        """ 初始化。

        :param base_path: str. 初始断面存放目录。
        :param work_path: str. Env工作目录。
        :param inputs: {etype: [str]}. 强化学习模型的输入量。
        :param fmt: str. 数据格式类型。
        """
        self.power = Power(fmt)
        self.fmt = fmt
        self.base_path = base_path
        self.work_path = work_path
        self.episode = 0
        self.step = 0
        self.assessments = []
        self.state0 = None
        self.min_max = None
        self.init_load_p = 0.
        self.inputs = inputs

    def get_ep_path(self, ep=None, step=None):
        """ 获取指定episode和step的工作目录。

        :param ep: int. 指定episode；
                   or None. 使用当前episode。
        :param step: int. 指定step；
                     or None. 返回episode对应目录。
        :return: str.
        """
        if ep is None:
            ep = self.episode
        if step is None:
            return os.path.join(self.work_path, 'ep%06d' % ep)
        return os.path.join(self.work_path, 'ep%06d' % ep, str(step))

    def reset(self, random=True, load_path=None):
        """ 重置潮流，并进行评估。

        :param random: bool. 是否随机初始化潮流。
        :param load_path: str. 初始断面目录；
                          or None. 用self.base_path作为初始断面。
        :return: bool. 是否重置成果(not done)
        """
        if load_path is None:
            self.power.load_power(self.base_path, fmt=self.fmt)
        else:
            self.power.load_power(load_path, fmt=self.fmt)
        self.power.data['generator']['p0'] = self.power.data['generator']['p']
        self.episode += 1
        path = self.get_ep_path()
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
        self.step = -1
        self.assessments = []
        if random:
            generators = self.power.data['generator']
            loads = self.power.data['load']
            max_p, gen_p = np.sum(generators[['pmax', 'p']])
            p = max_p * 0.4 + max_p * 0.5 * np.random.rand()  # 40% ~ 90%
            distribute_generators_p(generators, p - gen_p, sigma=0.2)
            generators['p0'] = np.clip(generators['p0'],
                                       generators['pmin'], generators['pmax'])
            gen_p = np.sum(generators['p0'])
            load_p = np.sum(loads['p'])
            distribute_loads_p(loads, 0.9 * gen_p - load_p, p_sigma=0.1, keep_factor=False)
            random_load_q0(loads, sigma=None)
        self.min_max = None
        self.init_load_p = 0.
        self.state0, _, done = self.run_step()
        return not done

    def get_state(self, normalize=True):
        """ 获取当前输入量数值

        :param normalize: bool. True返回归一化数据；False返回原始数据。
        :return: np.array.
        """
        state = []
        for etype, columns in self.inputs.items():
            state.append(self.power.data[etype][columns].values.T.reshape(-1))
        state = np.concatenate(state)
        if normalize:
            state = (state - self.min_max[:, 0]) \
                    / (self.min_max[:, 1] - self.min_max[:, 0] + EPS)
        return state

    def load_init_info(self):
        """ 获取初始状态，包括输入量的上下限和总负荷功率。

        """
        values = []
        for etype, columns in self.inputs.items():
            for col in columns:
                if col == 'p0':
                    values.append(self.power.data[etype][['pmin', 'pmax']].values)
                elif col == 'q0':
                    values.append(self.power.data[etype][['qmin', 'qmax']].values)
                else:
                    continue
        self.min_max = np.concatenate(values)
        loads = self.power.data['load']
        self.init_load_p = np.sum(loads.loc[loads['mark'] == 1, 'p0'])

    @staticmethod
    def make_assessment(path):
        """ 对断面稳定结果进行打分。

        :param path: str. 指定断面目录，包含*.res结果文件。
        :return: (float, bool, np.array). 评分、结束标志、稳定结果。
        """
        headers = {'CCTOUT': 'no desc name cct gen1 gen2 times tmp1 tmp2'}
        update_table_header(path, 'res', headers)
        iwant = {'CCTOUT': ['name', 'cct']}
        results = []
        for file_name in os.listdir(path):
            if file_name.endswith('.res'):
                cct = read_efile(os.path.join(path, file_name), iwant.keys(), iwant)
                results.append(cct['CCTOUT']['cct'])
        results = pd.concat(results)
        values = results.values.reshape(-1,)
        values = values[~np.isnan(values)]
        values = values[values > 0.]
        if len(values) == 0 or np.min(values) < 0.1:
            return ST_UNSTABLE_REWARD, True, results
        # thrs = [0.3, 0.5]
        thrs = [1.0]
        for thr in thrs:
            values_lt = values[values < thr]
            if len(values_lt) > 0:
                return np.average(values_lt), False, results
        return thrs[-1], False, results

    def run_step(self, actions=None):
        """ 按照给定actions运行一步。

        :param actions: str. “random"，随机初始化，仅用于测试；
                        or dict. 动作集合，由load_action函数执行动作。
        :return: (np.array, float, bool). 状态量、回报值、结束标志。
        """
        self.step += 1
        path = self.get_ep_path(step=self.step)
        if actions == 'random':  # just for test
            distribute_generators_p(self.power.data['generator'], 1., sigma=0.1)
            distribute_loads_p(self.power.data['load'], 1., p_sigma=0.1,
                               keep_factor=True, factor_sigma=0.1)
        elif actions is not None:
            self.load_action(actions)
        self.power.save_power(path, self.fmt, lf=True, lp=False, st=True)
        shutil.copy(os.path.join(self.base_path, 'LF.L0'), path)
        shutil.copy(os.path.join(self.base_path, 'ST.S0'), path)
        call_wmlf(path)
        if check_lfcal(path):
            self.power.drop_data(self.fmt, 'lp')
            self.power.load_power(path, self.fmt, lf=False, lp=True, st=False)
            self.power.data['generator']['p0'] = self.power.data['generator']['p']
            if self.step == 0:
                self.load_init_info()
            state = self.get_state()
            if os.name != 'nt':
                call_psa(path)
                assess, done, _ = self.make_assessment(path)
            else:
                assess = np.random.rand()
                done = (assess < 0.1)
        else:
            state = []
            assess = PF_NOTCONV_REWARD
            done = True
        self.assessments.append(assess)
        if self.step == 0:
            reward = 0.
        else:
            reward = self.assessments[-1] - self.assessments[-2]
            if not done:
                reward *= CCT_CHANGE_RATIO
                loads = self.power.data['load']
                load_p = np.sum(loads.loc[loads['mark'] == 1, 'p0'])
                if abs(load_p - self.init_load_p) / self.init_load_p >= LOAD_CHANGE_THR:
                    reward += PF_LOADFULL_REWARD
                    done = True
            else:
                pass
                # reward = assess
        return state, reward, done

    def load_action(self, actions):
        """ 加载潮流修改动作，但不计算潮流。

        :param actions: dict. 动作字典：'load_ratio_p'~按比例调整负荷有功；
                                       'generator_ratio_p'~按比例调整机组有功。
        :return: np.array. 归一化的状态量。
        """
        for k in actions:
            if k == 'load_ratio_p':
                set_gl_p0(self.power.data['load'],
                          self.power.data['load']['p0'] * actions[k],
                          keep_factor=False, clip=False)
            elif k == 'generator_ratio_p':
                set_gl_p0(self.power.data['generator'],
                          self.power.data['generator']['p0'] * actions[k],
                          keep_factor=False, clip=True)
        return self.get_state()

    def print_info(self, state=True, assessment=True):
        """ 打印每步信息。

        :param state: bool. 是否打印状态量。
        :param assessment: bool. 是否打印评分值。
        """
        print('episode = %d, step = %d' % (self.episode, self.step))
        if state:
            print('state =', self.get_state())
        if assessment:
            print('assessment =', self.assessments)


def load_trend(path, fmt, inputs):
    """ 重新加载一个episode中每个step的潮流信息、评分和稳定结果，用于事后分析。

    :param path: str. Episode目录。
    :param fmt: str. 数据格式类型。
    :param inputs: {etype: [str]}. 强化学习模型的输入量，以及'score'和'res'。
    :return: {etype: np.array}.
    """
    data = {}
    for i in range(len(os.listdir(path))):
        sub_path = os.path.join(path, str(i))
        if not os.path.exists(os.path.join(sub_path, 'LF.L1')):
            continue
        if fmt is not None:
            power = Power(fmt)
            power.load_power(sub_path, fmt=fmt, st=False, station=False)
            for etype, columns in inputs.items():
                if etype not in power.data:
                    continue
                df = power.data[etype][columns]
                for col in columns:
                    name = '_'.join((etype, col))
                    dict_list_append(data, name, df[col])
        score, _, res = OpEnv.make_assessment(os.path.join(path, str(i)))
        if 'score' in inputs:
            dict_list_append(data, 'score', pd.Series([score]))
        if 'res' in inputs:
            dict_list_append(data, 'res', res)
    for k in data:
        data[k] = pd.concat(data[k], axis=1)
        data[k].columns = list(range(data[k].shape[1]))
    return data


if __name__ == '__main__':
    '''
    if os.name == 'nt':
        base_path = 'D:/PSASP_Pro/wepri36/wepri36'
        work_path = 'D:/PSASP_Pro/wepri36'
    else:
        base_path = '/home/sdy/data/wepri36/wepri36'
        work_path = '/home/sdy/data/wepri36'
    inputs = {'generator': ['p0'],
              'load': ['p0', 'q0']}

    env = OpEnv(base_path, work_path, inputs, fmt='off')
    print('**********************')
    if env.reset(random=False):
        print('Copy reset success!')
        env.print_info()
    else:
        print('Copy reset failed!')
        env.print_info()

    print('**********************')
    if env.reset(random=True):
        print('Random reset success!')
        env.print_info()
        for i in range(3):
            s, r, done = env.run_step(actions='random')
            env.print_info()
            print(r, done)
            if done:
                break
    else:
        print('Random reset failed!')
        env.print_info()
    '''

    data = []
    inputs = {'generator': ['p'],
              'load': ['p'],
              'score': [], 'res': []}
    work_path = 'D:/PSASP_Pro/wepri36'
    paths = [os.path.join(work_path, 'ep000013'),
             os.path.join(work_path, 'ep000014'),
             os.path.join(work_path, 'ep000015')]
    data.append(load_trend(paths[0], fmt='off', inputs=inputs))
    data.append(load_trend(paths[1], fmt=None, inputs=inputs))
    data.append(load_trend(paths[2], fmt=None, inputs=inputs))
    scores = pd.concat([data[0]['score'], data[1]['score'], data[2]['score']])
    scores.fillna(method='ffill', axis=1, inplace=True)
    scores.index = range(3)
    fig = plt.figure(num=1, figsize=(8, 6))
    x = range(data[0]['score'].shape[1])
    ax_gen = fig.add_subplot(3, 1, 1)
    for idx in data[0]['generator_p'].index:
        ax_gen.plot(x, data[0]['generator_p'].loc[idx], label=idx)
    ax_gen.set_ylabel('Generator P')
    ax_gen.legend(loc='best', ncol=2)
    ax_load = fig.add_subplot(3, 1, 2)
    for idx in data[0]['load_p'].index:
        ax_load.plot(x, data[0]['load_p'].loc[idx], label=idx)
    ax_load.set_ylabel('Load P')
    ax_load.legend(loc='best', ncol=2)
    ax_score = fig.add_subplot(3, 1, 3)
    score_labels = ['PPO', 'NOP', 'DIS']
    for idx in scores.index:
        ax_score.plot(x, scores.loc[idx], 'o-', label=score_labels[idx])
    ax_score.set_xlabel('Round')
    ax_score.set_ylabel('Score')
    ax_score.legend()

