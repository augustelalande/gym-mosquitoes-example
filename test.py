import gym
import gym_mosquitoes
from net import Net
import numpy as np
import random


def discretize_actions(env, bins):
    low = env.action_space.low
    high = env.action_space.high
    actions = [
        low + i * (high - low) / (bins - 1) for i in range(bins)
    ]
    return actions


if __name__ == '__main__':
    env = gym.make('Mosquitoes-v0')
    actions = discretize_actions(env, 20)
    Q = Net('net1', layers=1)
    Q.load('1_layer_simple')

    s = env.reset()
    env.render()
    features = np.hstack([s, s])
    while env.t < 100000:
        a_i = Q.get_action(np.expand_dims(features, axis=0))[0]

        for j in range(50):
            print(actions[0], a_i)
            s_n, r, _, _ = env.step(actions[0])
        env.render()
        features_n = np.hstack([s, s_n])

        s = s_n
        features = features_n
