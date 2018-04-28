import gym
import gym_mosquitoes
from mem_buffer import MemBuffer
import numpy as np
import random
from net import Net
import tensorflow as tf

BATCH_SIZE = 32
MEM_SIZE = 1000
TARGET_UPDATE_FREQ = 50
GAMMA = 0.9
REPLAY_START_SIZE = MEM_SIZE

NUM_EPISODES = 100

DISCRETE_ACTIONS = 20


class Learner(object):
    def __init__(self):
        self.env = gym.make('Mosquitoes-v0')
        self.init_discretized_actions(DISCRETE_ACTIONS)
        self.init_mem_buffer()

    def init_mem_buffer(self):
        self.mem_buffer = MemBuffer(MEM_SIZE)
        s = self.env.reset()
        features = np.hstack([s, s])
        while not self.mem_buffer.full:
            while self.env.t < 10000:
                a = random.choice(self.actions)
                a_i = self._action_to_i(a)
                for j in range(50):
                    s_n, r, _, _ = self.env.step(a)
                features_n = np.hstack([s, s_n])
                self.mem_buffer.add(features, a_i, r, features_n)
                s = s_n
                features = features_n
            s = self.env.reset()
            features = np.hstack([s, s])

    def learn(self):
        layers = 2
        Q = Net('net1', layers=layers)
        Q_hat = Net('net2', layers=layers, copy=Q)
        Q_hat.copy()

        epsilon = 0.1
        c = 0

        for episode_i in range(NUM_EPISODES):
            s = self.env.reset()
            features = np.hstack([s, s])

            while self.env.t < 10000:
                if np.random.random() >= epsilon:
                    a_i = Q.get_action(np.expand_dims(features, axis=0))[0]
                else:
                    a = random.choice(self.actions)
                    a_i = self._action_to_i(a)
                for j in range(50):
                    s_n, r, _, _ = self.env.step(self.actions[a_i])
                features_n = np.hstack([s, s_n])
                self.mem_buffer.add(features, a_i, r, features_n)

                batch = self.get_batch(Q_hat)
                Q.optimize(*batch)

                s = s_n
                features = features_n

                c += 1

                if c >= TARGET_UPDATE_FREQ:
                    c = 0
                    Q_hat.copy()

            print(episode_i, r)

        Q.save('2_layers_simple')

    def get_batch(self, Q_hat):
        f, a, r, f_n = self.mem_buffer.sample(BATCH_SIZE)
        v_max = Q_hat.get_value(f_n)
        t = r + GAMMA * v_max
        target = np.zeros((BATCH_SIZE, DISCRETE_ACTIONS))
        target[np.arange(BATCH_SIZE), a] = t
        mask = np.zeros((BATCH_SIZE, DISCRETE_ACTIONS))
        mask[np.arange(BATCH_SIZE), a] = 1
        return f, target, mask

    def init_discretized_actions(self, bins):
        low = self.env.action_space.low
        high = self.env.action_space.high
        self.actions = [
            low + i * (high - low) / (bins - 1) for i in range(bins)
        ]
        self.action_indeces = {
            tuple(a): i for i, a in enumerate(self.actions)
        }

    def _action_to_i(self, action):
        return self.action_indeces[tuple(action)]


if __name__ == '__main__':
    l = Learner()
    l.learn()
