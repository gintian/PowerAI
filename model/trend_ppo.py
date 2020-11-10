# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

from collections import namedtuple
import argparse
import os
import shutil
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, layers
from tensorflow.keras.models import Model
import tensorflow_probability as tfp

from core.action import distribute_generators_p
from model.env import OpEnv

Transition = namedtuple('Transition', ('state', 'value', 'action', 'reward', 'returns'))

EPISODE_MAX = 1000  # total number of episodes for training
TEST_MAX = 10
TAU_LEN = 32  # total number of steps for each tau
NUM_EPOCH = 2
BATCH_SIZE = 1024
MINIBATCH_SIZE = 128
MAX_TAU_RETRY = 10
GAMMA = 0.90  # reward discount
AG_LR = 0.001  # learning rate for actor G
AD_LR = 0.001  # learning rate for actor D
C_LR = 0.005  # learning rate for critic
A_UPDATE_STEPS = 5  # actor update steps
C_UPDATE_STEPS = 5  # critic update steps
GENERATOR_DIM = 8
LOAD_DIM = 10
STATE_DIM = GENERATOR_DIM + LOAD_DIM * 2
G_OUT_DIM, D_OUT_DIM = GENERATOR_DIM, LOAD_DIM
LOAD_CLIP = 0.05  # 95% ~ 105%
GENERATOR_CLIP = 0.2  # 80% ~ 120%
LOAD_SIGMA_COEF = 0.002
GENERATOR_SIGMA_COEF = 0.01
EPS = 1e-8  # epsilon
PPO2_CLIP = 0.2


class TrendPPO(object):
    def __init__(self, work_path):
        """ 初始化。

        :param work_path: str. 工作目录，存储PPO执行过程中的文件。
        """
        self.work_path = work_path
        self.model_path = os.path.join(self.work_path, 'model')
        self.log_path = os.path.join(self.work_path, 'log')
        shutil.rmtree(self.log_path, ignore_errors=True)
        self.log_writer = tf.summary.create_file_writer(self.log_path)
        self.log_writer.set_as_default()

        prelu = tf.keras.layers.ReLU(negative_slope=0.1)

        self.state = Input(shape=(STATE_DIM,), dtype='float32', name='state')
        self.adv = Input(shape=(1,), dtype='float32', name='adv')
        self.actor_d = self.mlp_normal(self.state, [64, 32, D_OUT_DIM], 'actor_d',
                                       activation=prelu,
                                       mu_ratio=LOAD_CLIP, sigma_ratio=0.1)
        self.actor_d_old = self.mlp_normal(self.state, [64, 32, D_OUT_DIM], 'actor_d_old',
                                           activation=prelu,
                                           mu_ratio=LOAD_CLIP, sigma_ratio=0.1)
        self.actor_g = self.mlp_normal(self.state, [64, 32, G_OUT_DIM], 'actor_g',
                                       activation=prelu,
                                       mu_ratio=GENERATOR_CLIP, sigma_ratio=0.1)
        self.actor_g_old = self.mlp_normal(self.state, [64, 32, G_OUT_DIM], 'actor_g_old',
                                           activation=prelu,
                                           mu_ratio=GENERATOR_CLIP, sigma_ratio=0.1)
        # G/D网络输入为state，输出为G/D的动作量
        x = layers.Dense(64, activation=prelu, dtype='float32')(self.state)
        x = layers.Dense(32, activation=prelu)(x)
        critic = layers.Dense(1, activation=None)(x)
        self.critic = Model(self.state, critic, name='critic')
        # critic网络输入为state，输出为v_s。
        self.actor_d_opt = tf.optimizers.Adam(AD_LR)
        self.actor_g_opt = tf.optimizers.Adam(AG_LR)
        self.critic_opt = tf.optimizers.Adam(C_LR)

    @staticmethod
    def mlp_normal(inputs, shape, name, activation='relu',
                   mu_activation='tanh', mu_ratio=0.1,
                   sigma_activation='sigmoid', sigma_ratio=0.1):
        """ 多层感知机网络，输出正态分布的mu和sigma。

        :param inputs: Input. 输入量。
        :param shape: list[int]. 网络每层神经元数量。
        :param name: str. 网络名称。
        :param activation: str or activation func. 中间层激活函数。
        :param mu_activation: str or activation func. mu的激活函数。
        :param mu_ratio: float. mu的变化比例范围。
        :param sigma_activation: str or activation func. sigma的激活函数。
        :param sigma_ratio: float. sigma的变化比例范围。
        :return: Model. 从inputs到[mu, sigma]的网络模型。
        """
        x = inputs
        for i in range(len(shape) - 1):
            x = layers.Dense(shape[i], activation=activation, dtype='float32')(x)
        mu = layers.Dense(shape[-1], activation=mu_activation)(x)
        mu = layers.Lambda(lambda xx: xx * mu_ratio + 1., name=name + '_mu')(mu)
        sigma = layers.Dense(shape[-1], activation=sigma_activation)(x)
        sigma = layers.Lambda(lambda xx: xx * sigma_ratio, name=name + '_sigma')(sigma)
        model = Model(inputs, [mu, sigma], name=name)
        return model

    def actor_train(self, state, act, adv, name):
        """ actor模型训练函数，G/D网络输入为state，输出为G/D的动作量。
        根据优势量adv调整G动作量或D动作量出现的概率，G目标是增加adv，D目标是减小adv。

        :param state: np.array. 电网状态输入量。
        :param act: np.array. 动作量。
        :param adv: np.array. 优势量。adv = r + GAMMA * v_{s+1} - v_s
        :param name str. 'g'表示训练G网络，'d'表示训练D网络。
        :return: pg_loss, diverse_loss. 策略梯度损失和多样性损失。
        """
        if name == 'g':
            actor = self.actor_g
            actor_old = self.actor_g_old
            actor_opt = self.actor_g_opt
            sigma_coef = GENERATOR_SIGMA_COEF
        else:
            actor = self.actor_d
            actor_old = self.actor_d_old
            actor_opt = self.actor_d_opt
            sigma_coef = LOAD_SIGMA_COEF
        with tf.GradientTape() as tape:
            mu, sigma = actor(state)
            pi = tfp.distributions.Normal(mu, sigma)
            mu_old, sigma_old = actor_old(state)
            pi_old = tfp.distributions.Normal(mu_old, sigma_old)
            # ratio = pi.prob(act) / (pi_old.prob(act) + EPS)
            ratio = tf.exp(tf.reduce_sum(pi.log_prob(act) - pi_old.log_prob(act),
                                         axis = 1, keepdims=True))
            clipped_ratio = tf.clip_by_value(ratio, 1. - PPO2_CLIP, 1. + PPO2_CLIP)
            if name == 'g':
                pg_loss = -tf.reduce_mean(tf.minimum(ratio * adv, clipped_ratio * adv))
                # ‘-’代表G网络最大化 adv，同时需要抑制adv不能太大
            else:
                pg_loss = tf.reduce_mean(tf.maximum(ratio * adv, clipped_ratio * adv))
                # '+'代表D网络最小化 adv，同时需要抑制adv不能太小
            diverse_loss = tf.reduce_mean(sigma * sigma_coef)
            loss = pg_loss - diverse_loss
        grad = tape.gradient(loss, actor.trainable_weights)
        actor_opt.apply_gradients(zip(grad, actor.trainable_weights))  # 梯度下降
        return pg_loss, diverse_loss

    def update_old_d(self):
        """ 把D网络参数更新至D_old中。 """
        for w, w0 in zip(self.actor_d.trainable_weights, self.actor_d_old.trainable_weights):
            w0.assign(w)

    def update_old_g(self):
        """ 把G网络参数更新至G_old中。 """
        for w, w0 in zip(self.actor_g.trainable_weights, self.actor_g_old.trainable_weights):
            w0.assign(w)

    def critic_train(self, state, returns):
        """ critic模型训练函数，输入为state，输出为v_s。

        :param state: np.array. 输入状态量。
        :param returns: np.array. Return = r + GAMMA * v_{s+1}
        :return loss. 值评估损失。
        """
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(returns - self.critic(state)))
        grad = tape.gradient(loss, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(grad, self.critic.trainable_weights))
        return loss

    def cal_adv(self, state, returns):
        """ 计算adv，adv = r + GAMMA * v_{s+1} - v_s = Return - v_s

        :param state: np.array. 输入状态量。
        :param returns: np.array. Return值。
        :return: np.array. adv优势量。
        """
        adv = returns - self.critic(state).numpy()
        return adv

    def update(self, state_d, state_g, act_d, act_g, returns_d, returns_g, lock_d=False, lock_g=False):
        self.update_old_d()
        self.update_old_g()
        adv_d = self.cal_adv(state_d, returns_d)
        # adv_d = (adv_d - adv_d.mean())/(adv_d.std()+1e-6)  # sometimes helpful
        adv_g = self.cal_adv(state_g, returns_g)
        # adv_g = (adv_g - adv_g.mean())/(adv_g.std()+1e-6)  # sometimes helpful
        d_pg_loss, d_diverse_loss, g_pg_loss, g_diverse_loss = (0., 0., 0., 0.)
        for _ in range(A_UPDATE_STEPS):
            if not lock_d:
                d_pg_loss, d_diverse_loss = self.actor_train(state_d, act_d, adv_d, name='d')
            if not lock_g:
                g_pg_loss, g_diverse_loss = self.actor_train(state_g, act_g, adv_g, name='g')
        tf.summary.scalar('d_pg_loss', data=d_pg_loss)
        tf.summary.scalar('d_diverse_loss', data=d_diverse_loss)
        tf.summary.scalar('g_pg_loss', data=g_pg_loss)
        tf.summary.scalar('g_diverse_loss', data=g_diverse_loss)
        # tf.summary.scalar('v0', data=tf.reduce_mean(self.critic(state_d)))
        states = np.concatenate([state_d, state_g], axis=0)
        returns = np.concatenate([returns_d, returns_g], axis=0)
        for _ in range(C_UPDATE_STEPS):
            c_loss = self.critic_train(states, returns)
        # print('gt=', gt.reshape(-1), '\ncritic=', self.critic(state_d).numpy().reshape(-1))
        tf.summary.scalar('c_loss', data=c_loss)
        # tf.summary.scalar('v', data=tf.reduce_mean(self.critic(state_d)))
        # tf.summary.scalar('q', data=tf.reduce_mean(q))

    def choose_action(self, state, name, greedy=False):
        state = state[np.newaxis, :]
        if name == 'd':
            mu, sigma = self.actor_d(state)
        else:
            mu, sigma = self.actor_g(state)
        if greedy:
            act = mu[0]
        else:
            pi = tfp.distributions.Normal(mu, sigma)
            act = tf.squeeze(pi.sample(1), axis=0)[0]
        if name == 'd':
            act = np.clip(act, 1. - LOAD_CLIP, 1. + LOAD_CLIP)
        else:
            act = np.clip(act, 1. - GENERATOR_CLIP, 1. + GENERATOR_CLIP)
        return act

    def get_v(self, state):
        if state.ndim < 2:
            state = state[np.newaxis, :]
        return self.critic(state)[0, 0]

    def save_model(self):
        shutil.rmtree(self.model_path, ignore_errors=True)
        os.makedirs(self.model_path)
        self.actor_d.save_weights(os.path.join(self.model_path, 'actor_d'))
        self.update_old_d()
        self.actor_g.save_weights(os.path.join(self.model_path, 'actor_g'))
        self.update_old_g()
        self.critic.save_weights(os.path.join(self.model_path, 'critic'))

    def load_model(self):
        self.actor_d.load_weights(os.path.join(self.work_path, 'model', 'actor_d'))
        self.update_old_d()
        self.actor_g.load_weights(os.path.join(self.work_path, 'model', 'actor_g'))
        self.update_old_g()
        self.critic.load_weights(os.path.join(self.work_path, 'model', 'critic'))


def run_tau(ppo, env, policy_d='ppo', policy_g='ppo', greedy=False,
            lock_d=False, lock_g=False, def_actions_d=None):
    tau = []
    step = min(len(def_actions_d), TAU_LEN) if def_actions_d else TAU_LEN
    for i in range(step):
        loads = env.power.data['load']
        generators = env.power.data['generator']
        # L step
        act_d = None
        state = env.get_state()
        if not lock_d:
            load_p0 = np.sum(loads.loc[loads['mark'] == 1, 'p0'])
            if policy_d == 'ppo':
                act_d = ppo.choose_action(state, 'd', greedy=greedy)
            elif policy_d == 'def':
                act_d = def_actions_d[i]
            env.load_action({'load_ratio_p': act_d})
            if policy_g != 'nop':
                delta = np.sum(loads.loc[loads['mark'] == 1, 'p0']) - load_p0
                distribute_generators_p(generators, delta)
        next_state, r, done = env.run_step()
        tau.append([state, ppo.get_v(state).numpy(), act_d, r, 0.])
        if done:
            break
        # G step
        state = next_state
        act_g = None
        if not lock_g:
            gen_p0 = np.sum(generators.loc[generators['mark'] == 1, 'p0'])
            if policy_g == 'ppo':
                act_g = ppo.choose_action(state, 'g', greedy=greedy)
                env.load_action({'generator_ratio_p': act_g})
                delta = np.sum(generators.loc[generators['mark'] == 1, 'p0']) - gen_p0
                distribute_generators_p(generators, -delta)
        next_state, r, done = env.run_step()
        tau.append([state, ppo.get_v(state).numpy(), act_g, r, 0.])
        if done:
            break
    v_s_ = ppo.get_v(next_state.astype(np.float32)).numpy() if not done else 0.
    for i in reversed(range(len(tau))):
        v_s_ = tau[i][3] + GAMMA * v_s_
        tau[i][4] = v_s_
    mem_d = [tau[i] for i in range(len(tau)) if i % 2 == 0]
    mem_g = [tau[i] for i in range(len(tau)) if i % 2 == 1]
    return mem_d, mem_g


def make_test_stat(assess):
    ep_len = assess.shape[0]
    all_len = assess[['ppo_len', 'nop_len', 'dis_len']]
    idxmax_len = all_len.idxmax(axis=1)
    all_assess = assess[['ppo_assess', 'nop_assess', 'dis_assess']]
    idxmax_assess = all_assess.idxmax(axis=1)
    winner_stat = {'all': ep_len, 'ppo': 0, 'nop': 0, 'agc': 0}
    for i, ser in assess.iterrows():
        if np.sum(ser == ser.loc[idxmax_len[i]]) == 1:
            winner = idxmax_len[i].split('_')[0]
        else:
            winner = idxmax_assess[i].split('_')[0]
        winner_stat[winner] = winner_stat[winner] + 1
    return winner_stat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test trend ppo model.')
    parser.add_argument('--train', dest='train', action='store_true', default=True)
    parser.add_argument('--test', dest='train', action='store_false')
    args = parser.parse_args()

    base_path = os.path.join(os.path.expanduser('~'), 'data', 'wepri36', 'wepri36')
    work_path = os.path.join(os.path.expanduser('~'), 'data', 'wepri36')
    inputs = {'generator': ['p0'],
              'load': ['p0', 'q0']}
    env = OpEnv(base_path, work_path, inputs, fmt='off')

    ppo = TrendPPO(work_path)

    if args.train:
        lock_d, lock_g = False, False
        for ep in range(EPISODE_MAX):
            memory_d, memory_g = [], []
            t0 = time.time()
            n_try = 0
            while len(memory_g) < BATCH_SIZE:
                try:
                    env.reset(random=True)
                    mem_d, mem_g = run_tau(ppo, env, greedy=False,
                                           lock_d=lock_d, lock_g=lock_g)
                except ValueError:
                    env.episode = env.episode - 1
                    n_try = n_try + 1
                    if n_try >= MAX_TAU_RETRY:
                        print('Env run tau failed, stop training.')
                        break
                    else:
                        continue
                else:
                    n_try = 0

                memory_d.extend(mem_d)
                memory_g.extend(mem_g)
                print('Mem={}+{}/{}, Time={:.4f}'.format(len(memory_d), len(memory_g),
                                                         BATCH_SIZE,
                                                         time.time() - t0))
            memory_d = Transition(*zip(*memory_d))
            state_d = np.array(memory_d.state, dtype=np.float32)
            # value_d = np.array(memory_d.value, dtype=np.float32)
            action_d = np.array(memory_d.action, dtype=np.float32)
            returns_d = np.array(memory_d.returns, dtype=np.float32)[:, np.newaxis]
            memory_g = Transition(*zip(*memory_g))
            state_g = np.array(memory_g.state, dtype=np.float32)
            # value_g = np.array(memory_g.value, dtype=np.float32)
            action_g = np.array(memory_g.action, dtype=np.float32)
            returns_g = np.array(memory_g.returns, dtype=np.float32)[:, np.newaxis]

            tf.summary.experimental.set_step(ep + 1)
            for _ in range(NUM_EPOCH):
                inds = np.arange(BATCH_SIZE)
                for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                    mb_inds = inds[start:start+MINIBATCH_SIZE]
                    ppo.update(state_d[mb_inds], state_g[mb_inds],
                               action_d[mb_inds], action_g[mb_inds],
                               returns_d[mb_inds], returns_g[mb_inds],
                               lock_d=lock_d, lock_g=lock_g)
            print('Episode: {}/{} | Running Time: {:.4f}'.format(
                ep + 1, EPISODE_MAX, time.time() - t0))
        ppo.save_model()
        ppo.log_writer.close()
    else:
        ppo.load_model()
        assess = []
        for ep in range(TEST_MAX):
            try:
                env.reset(random=True)
            except ValueError:
                env.episode = env.episode - 1
                continue
            t0 = time.time()
            mem_d, mem_g = run_tau(ppo, env, greedy=True)
            print('PPO: len={}/{} assess={}'.format(len(mem_d), len(mem_g), env.assessments))
            ppo_len, ppo_assess = len(env.assessments), np.average(env.assessments)
            path0 = env.get_ep_path(step=0)
            env.reset(random=False, load_path=path0)
            d_actions = [mem_d[i][2] for i in range(len(mem_d))]
            mem_d, mem_g = run_tau(ppo, env, policy_d='def', policy_g='nop',
                                   def_actions_d=d_actions)
            print('NOP: len={}/{} assess={}'.format(len(mem_d), len(mem_g), env.assessments))
            nop_len, nop_assess = len(env.assessments), np.average(env.assessments)
            env.reset(random=False, load_path=path0)
            mem_d, mem_g = run_tau(ppo, env, policy_d='def', policy_g='agc',
                                   def_actions_d=d_actions)
            print('DIS: len={}/{} assess={}'.format(len(mem_d), len(mem_g), env.assessments))
            dis_len, dis_assess = len(env.assessments), np.average(env.assessments)
            print('Episode: {}/{}  | Running Time: {:.4f}'.format(env.episode, TEST_MAX,
                                                                  time.time() - t0))
            assess.append([ppo_len, ppo_assess,
                           nop_len, nop_assess,
                           dis_len, dis_assess])
        assess = pd.DataFrame(np.array(assess),
                              columns=['ppo_len', 'ppo_assess',
                                       'nop_len', 'nop_assess',
                                       'dis_len', 'dis_assess'])
        assess.to_csv(os.path.join(env.work_path, 'model', 'assess.csv'))
