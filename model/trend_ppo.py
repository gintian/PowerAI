# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

import argparse
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, layers
from tensorflow.keras.models import Model
import tensorflow_probability as tfp

from core.action import distribute_generators_p

EP_TRAIN = 10  # total number of episodes for training
EP_TEST = 10
EP_LEN = 16  # total number of steps for each episode
GAMMA = 0.50  # reward discount
AG_LR = 0.0001  # learning rate for actor G
AD_LR = 0.0001  # learning rate for actor D
C_LR = 0.0002  # learning rate for critic
A_UPDATE_STEPS = 10  # actor update steps
C_UPDATE_STEPS = 10  # critic update steps
GENERATOR_DIM = 8
LOAD_DIM = 10
STATE_DIM = GENERATOR_DIM + LOAD_DIM * 2
G_OUT_DIM, D_OUT_DIM = GENERATOR_DIM, LOAD_DIM
LOAD_CLIP = 0.05  # 95% ~ 105%
GENERATOR_CLIP = 0.2  # 20% ~ 120%
EPS = 1e-8  # epsilon
PPO2_CLIP = 0.2


class TrendPPO(object):
    def __init__(self, work_path):
        self.work_path = work_path
        self.state = Input(shape=(STATE_DIM,), dtype='float32', name='state')
        self.adv = Input(shape=(1,), dtype='float32', name='adv')
        self.actor_d = self.mlp_normal(self.state, [64, 32, D_OUT_DIM], 'actor_d',
                                       mu_ratio=LOAD_CLIP, sigma_ratio=0.1)
        self.actor_d_old = self.mlp_normal(self.state, [64, 32, D_OUT_DIM], 'actor_d_old',
                                           mu_ratio=LOAD_CLIP, sigma_ratio=0.1)
        self.actor_g = self.mlp_normal(self.state, [64, 32, G_OUT_DIM], 'actor_g',
                                       mu_ratio=GENERATOR_CLIP, sigma_ratio=0.1)
        self.actor_g_old = self.mlp_normal(self.state, [64, 32, G_OUT_DIM], 'actor_g_old',
                                           mu_ratio=GENERATOR_CLIP, sigma_ratio=0.1)
        x = layers.Dense(64, activation='relu', dtype='float32')(self.state)
        x = layers.Dense(32, activation='relu')(x)
        critic = layers.Dense(1, activation=None)(x)
        self.critic = Model(self.state, critic, name='critic')
        self.actor_d_opt = tf.optimizers.Adam(AD_LR)
        self.actor_g_opt = tf.optimizers.Adam(AG_LR)
        self.critic_opt = tf.optimizers.Adam(C_LR)

    @staticmethod
    def mlp_normal(inputs, shape, name, activation='relu',
                   mu_activation='tanh', mu_ratio=0.1,
                   sigma_activation='sigmoid', sigma_ratio=0.1):
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
        if name == 'g':
            actor = self.actor_g
            actor_old = self.actor_g_old
            actor_opt = self.actor_g_opt
        else:
            actor = self.actor_d
            actor_old = self.actor_d_old
            actor_opt = self.actor_d_opt
        with tf.GradientTape() as tape:
            mu, sigma = actor(state)
            pi = tfp.distributions.Normal(mu, sigma)
            mu_old, sigma_old = actor_old(state)
            pi_old = tfp.distributions.Normal(mu_old, sigma_old)
            ratio = pi.prob(act) / (pi_old.prob(act) + EPS)
            clipped_ratio = tf.clip_by_value(ratio, 1. - PPO2_CLIP, 1. + PPO2_CLIP)
            # TODO: d net loss? maybe maximum?
            if name == 'g':
                loss = -tf.reduce_mean(tf.minimum(ratio * adv, clipped_ratio * adv))
            else:
                loss = tf.reduce_mean(tf.minimum(ratio * adv, clipped_ratio * adv))
        grad = tape.gradient(loss, actor.trainable_weights)  # maximize adv for g
        actor_opt.apply_gradients(zip(grad, actor.trainable_weights))

    def update_old_d(self):
        for w, w0 in zip(self.actor_d.trainable_weights, self.actor_d_old.trainable_weights):
            w0.assign(w)

    def update_old_g(self):
        for w, w0 in zip(self.actor_g.trainable_weights, self.actor_g_old.trainable_weights):
            w0.assign(w)

    def critic_train(self, state, q):
        with tf.GradientTape() as tape:
            adv = q - self.critic(state)
            loss = tf.reduce_mean(tf.square(adv))
        grad = tape.gradient(loss, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(grad, self.critic.trainable_weights))

    def cal_adv(self, state, q):
        adv = q - self.critic(state)
        return adv.numpy()

    def update(self, state_d, state_g, act_d, act_g, q):
        self.update_old_d()
        self.update_old_g()
        adv = self.cal_adv(state_d, q)
        # adv = (adv - adv.mean())/(adv.std()+1e-6)  # sometimes helpful
        for _ in range(A_UPDATE_STEPS):
            self.actor_train(state_d, act_d, adv, name='d')
            self.actor_train(state_g, act_g, adv, name='g')
        for _ in range(C_UPDATE_STEPS):
            self.critic_train(state_d, q)

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
        path = os.path.join(self.work_path, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        self.actor_d.save_weights(os.path.join(path, 'actor_d'))
        self.update_old_d()
        self.actor_g.save_weights(os.path.join(path, 'actor_g'))
        self.update_old_g()
        self.critic.save_weights(os.path.join(path, 'critic'))

    def load_model(self):
        self.actor_d.load_weights(os.path.join(self.work_path, 'model', 'actor_d'))
        self.update_old_d()
        self.actor_g.load_weights(os.path.join(self.work_path, 'model', 'actor_g'))
        self.update_old_g()
        self.critic.load_weights(os.path.join(self.work_path, 'model', 'critic'))


def run_episode(ppo, env, policy_d='ppo', policy_g='ppo', greedy=False,
                def_actions_d=None):
    states_d, states_g, actions_d, actions_g, rewards = [], [], [], [], []
    step = min(len(def_actions_d), EP_LEN) if def_actions_d else EP_LEN
    for i in range(step):
        loads = env.power.data['load']
        generators = env.power.data['generator']
        s_d = env.get_state()
        load_p = np.sum(loads.loc[loads['mark'] == 1, 'p0'])
        if policy_d == 'ppo':
            act_d = ppo.choose_action(s_d, 'd', greedy=greedy)
            actions_d.append(act_d)
            env.load_action({'load_ratio_p': act_d})
        elif policy_d == 'def':
            env.load_action({'load_ratio_p': def_actions_d[i]})
        s_g = env.get_state()
        delta = np.sum(loads.loc[loads['mark'] == 1, 'p0']) - load_p
        if policy_g == 'ppo':
            act_g = ppo.choose_action(s_g, 'g', greedy=greedy)
            actions_g.append(act_g)
            act_g = {'generator_ratio_p': act_g}
        else:
            if policy_g == 'dis':
                distribute_generators_p(generators, delta)
            act_g = None

        s_, r, done = env.run_step(act_g)
        states_d.append(s_d)
        states_g.append(s_g)
        rewards.append(r)
        if done:
            break
    return done, s_, states_d, states_g, actions_d, actions_g, rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test trend ppo model.')
    parser.add_argument('--train', dest='train', action='store_true', default=True)
    parser.add_argument('--test', dest='train', action='store_false')
    args = parser.parse_args()

    if os.name == 'nt':
        base_path = 'D:/PSASP_Pro/wepri36/wepri36'
        work_path = 'D:/PSASP_Pro/wepri36'
    else:
        base_path = '/home/sdy/data/wepri36/wepri36'
        work_path = '/home/sdy/data/wepri36'
    inputs = {'generator': ['p0'],
              'load': ['p0', 'q0']}
    env = OpEnv(base_path, work_path, inputs, fmt='off')

    ppo = TrendPPO(work_path)

    if args.train:
        all_ep_r = []
        for ep in range(EP_TRAIN):
            if not env.reset(random=True):
                print('Episode: {}/{}, reset failed!'.format(ep + 1, EP_TRAIN))
                all_ep_r.append(0.)
                continue
            t0 = time.time()
            done, s_, states_d, states_g, actions_d, actions_g, rewards = \
                run_episode(ppo, env, greedy=False)

            v_s_ = ppo.get_v(s_.astype(np.float32)).numpy() if not done else 0.
            q = []
            for r in rewards[::-1]:
                v_s_ = r + GAMMA * v_s_
                q.append(v_s_)
            q.reverse()
            # print('Done =', done, 'assess =', env.assessments)
            # print('Q =', q)
            ppo.update(np.vstack(states_d).astype(np.float32),
                       np.vstack(states_g).astype(np.float32),
                       np.vstack(actions_d).astype(np.float32),
                       np.vstack(actions_g).astype(np.float32),
                       np.array(q).astype(np.float32)[:, np.newaxis])

            ep_r = sum(rewards)
            if ep == 0:
                all_ep_r.append(ep_r)
            else:
                all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
            print(
                'Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    ep + 1, EP_TRAIN, ep_r,
                    time.time() - t0
                )
            )
        ppo.save_model()
    else:
        ppo.load_model()
        for ep in range(EP_TEST):
            if not env.reset(random=True):
                print('Episode: {}/{}, reset failed!'.format(ep + 1, EP_TEST))
                continue
            t0 = time.time()
            done, _, _, _, actions_d, _, rewards = run_episode(ppo, env, greedy=True)
            print('PPO: done = {}, step = {}, assess = {}'.format(done, len(rewards),
                                                                  env.assessments))
            path0 = env.get_ep_path(step=0)
            env.reset(random=False, load_path=path0)
            done, _, _, _, _, _, rewards = \
                run_episode(ppo, env, policy_d='def', policy_g='nop',
                            def_actions_d=actions_d)
            print('NOP: done = {}, step = {}, assess = {}'.format(done, len(rewards),
                                                                  env.assessments))
            env.reset(random=False, load_path=path0)
            done, _, _, _, _, _, rewards = \
                run_episode(ppo, env, policy_d='def', policy_g='dis',
                            def_actions_d=actions_d)
            print('DIS: done = {}, step = {}, assess = {}'.format(done, len(rewards),
                                                                  env.assessments))

            print('Episode: {}/{}  | Running Time: {:.4f}'.format(ep + 1, EP_TEST,
                                                                  time.time() - t0))
