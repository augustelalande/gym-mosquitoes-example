from __future__ import division

import gym
import gym_mosquitoes
from net import Net
from test import discretize_actions
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random


ACTIONS = None
DISCRETE_ACTIONS = 20


def eval_model(env, net, max_t):
    results = []
    s = env.reset(soft_reset=True)
    features = np.hstack([s, s])
    while env.t < max_t:
        a_i = net.get_action(np.expand_dims(features, axis=0))[0]
        for j in range(50):
            s_n, r, _, _ = env.step(ACTIONS[a_i])
        results.append(-r)
        features = np.hstack([s, s_n])
        s = s_n
    return results


def eval_model_learning(env, net, max_t):
    results = []
    s = env.reset(soft_reset=True)
    features = np.hstack([s, s])

    while env.t < max_t:
        if np.random.random() >= 0:
            a_i = net.get_action(np.expand_dims(features, axis=0))[0]
            a = ACTIONS[a_i]
        else:
            a = random.choice(ACTIONS)
            a_i = ACTIONS.index(a)
        for j in range(50):
            s_n, r, _, _ = env.step(a)
        results.append(-r)
        features_n = np.hstack([s, s_n])

        target = np.zeros((1, DISCRETE_ACTIONS))
        mask = np.zeros((1, DISCRETE_ACTIONS))
        target[0, a_i] = r
        mask[0, a_i] = 1
        batch = [
            np.expand_dims(features, axis=0),
            target, mask
        ]
        for i in range(10):
            net.optimize(*batch)

        s = s_n
        features = features_n
    return results


def eval_constant_action(env, a, max_t):
    results = []
    env.reset(soft_reset=True)
    while env.t < max_t:
        for j in range(50):
            _, r, _, _ = env.step(a)
        results.append(-r)
    return results


def plot_simple(env):
    max_t = 3000

    net = Net('net1', layers=1)
    net.load('1_layer_simple')
    r1 = eval_model(env, net, max_t)

    net = None
    tf.reset_default_graph()

    net = Net('net1', layers=2)
    net.load('2_layers_simple')
    r2 = eval_model(env, net, max_t)

    net = None
    tf.reset_default_graph()

    r3 = eval_constant_action(env, ACTIONS[6], max_t)
    r4 = eval_constant_action(env, ACTIONS[-1], max_t)
    x = [50 * i for i in range(1, int(max_t / 50 + 1))]

    plt.plot(x, r1, label="Linear Approximator")
    plt.plot(x, r2, label="1 Hidden Layer NN")
    plt.plot(x, r3, label="Best Action")
    plt.plot(x, r4, label="Worst Action")
    plt.ylim(.11, .14)
    plt.title("Offline - Same parametrization")
    plt.xlabel("step")
    plt.ylabel("reward (Total Symptomatic Infections)")
    plt.legend()
    plt.savefig("offline_simple.png", bbox_inches='tight')
    plt.clf()


def plot_random(env):
    max_t = 3000

    cum_results = []

    for i in range(50):
        env.reset()

        net = Net('net1', layers=1)
        net.load('1_layer_random')
        r1 = eval_model(env, net, max_t)

        net = None
        tf.reset_default_graph()

        net = Net('net1', layers=2)
        net.load('2_layers_random')
        r2 = eval_model(env, net, max_t)

        net = None
        tf.reset_default_graph()

        a_i = get_best_constant(env)
        r3 = eval_constant_action(env, ACTIONS[a_i], max_t)

        cum_results.append([r1, r2, r3])

    r1, r2, r3 = avg_results(cum_results)
    x = [50 * i for i in range(1, int(max_t / 50 + 1))]

    plt.plot(x, r1, label="Linear Approximator")
    plt.plot(x, r2, label="1 Hidden Layer NN")
    plt.plot(x, r3, label="Constant Action")
    # plt.ylim(.11, .14)
    plt.title("Offline - Different parametrization")
    plt.xlabel("step")
    plt.ylabel("reward (Total Symptomatic Infections)")
    plt.legend()
    plt.savefig("offline_random.png", bbox_inches='tight')
    plt.clf()


def plot_random_learning(env):
    max_t = 3000

    cum_results = []

    for i in range(50):
        env.reset()

        net = Net('net1', layers=1)
        net.load('1_layer_random')
        print('--')
        r1 = eval_model_learning(env, net, max_t)
        print('---')
        net = None
        tf.reset_default_graph()

        net = Net('net1', layers=2)
        net.load('2_layers_random')
        r2 = eval_model_learning(env, net, max_t)

        net = None
        tf.reset_default_graph()

        a_i = get_best_constant(env)
        r3 = eval_constant_action(env, ACTIONS[a_i], max_t)

        cum_results.append([r1, r2, r3])

    r1, r2, r3 = avg_results(cum_results)
    x = [50 * i for i in range(1, int(max_t / 50 + 1))]

    plt.plot(x, r1, label="Linear Approximator")
    plt.plot(x, r2, label="1 Hidden Layer NN")
    plt.plot(x, r3, label="Constant Action")
    # plt.ylim(.11, .14)
    plt.title("Online - Different parametrization")
    plt.xlabel("step")
    plt.ylabel("reward (Total Symptomatic Infections)")
    plt.legend()
    plt.savefig("online_random.png", bbox_inches='tight')
    plt.clf()


def plot_random_both(env):
    max_t = 3000

    cum_results1 = []
    cum_results2 = []

    for i in range(50):
        env.reset()

        net = Net('net1', layers=1)
        net.load('1_layer_random')
        r11 = eval_model(env, net, max_t)
        r12 = eval_model_learning(env, net, max_t)

        net = None
        tf.reset_default_graph()

        net = Net('net1', layers=2)
        net.load('2_layers_random')
        r21 = eval_model(env, net, max_t)
        r22 = eval_model_learning(env, net, max_t)


        net = None
        tf.reset_default_graph()

        a_i, b_i = get_best_constant(env)
        r3 = eval_constant_action(env, ACTIONS[a_i], max_t)
        r4 = eval_constant_action(env, ACTIONS[b_i], max_t)

        cum_results1.append([r11, r21, r3, r4])
        cum_results2.append([r12, r22, r3, r4])

    r1, r2, r3, r4 = avg_results(cum_results1)
    x = [50 * i for i in range(1, int(max_t / 50 + 1))]

    plt.plot(x, r1, label="Linear Approximator")
    plt.plot(x, r2, label="1 Hidden Layer NN")
    plt.plot(x, r3, label="Best Action")
    plt.plot(x, r4, label="Worst Action")
    # plt.ylim(.11, .14)
    plt.title("Offline - Different parametrization")
    plt.xlabel("step")
    plt.ylabel("reward (Total Symptomatic Infections)")
    plt.legend()
    plt.savefig("offline_random_1.png", bbox_inches='tight')
    plt.clf()

    r1, r2, r3, r4 = avg_results(cum_results2)
    x = [50 * i for i in range(1, int(max_t / 50 + 1))]

    plt.plot(x, r1, label="Linear Approximator")
    plt.plot(x, r2, label="1 Hidden Layer NN")
    plt.plot(x, r3, label="Best Action")
    plt.plot(x, r4, label="Worst Action")
    # plt.ylim(.11, .14)
    plt.title("Online - Different parametrization")
    plt.xlabel("step")
    plt.ylabel("reward (Total Symptomatic Infections)")
    plt.legend()
    plt.savefig("online_random_1.png", bbox_inches='tight')
    plt.clf()


def avg_results(cum_results):
    n = len(cum_results)
    avg_results = []
    for i in range(len(cum_results[0])):
        res = [r[i] for r in cum_results]
        avg = [sum(r[i] for r in res) / n for i in range(len(res[0]))]
        avg_results.append(avg)

    return avg_results


def get_best_constant(env):
    min_i = 0
    max_i = 0
    min_score = float('inf')
    max_score = float('-inf')
    for i, a in enumerate(ACTIONS):
        r = eval_constant_action(env, a, 3000)
        if r[-1] < min_score:
            min_score = r[-1]
            min_i = i
        if r[-1] > max_score:
            max_score = r[-1]
            max_i = i
    return min_i, max_i


if __name__ == '__main__':
    env = gym.make('Mosquitoes-v0')
    global ACTIONS
    ACTIONS = discretize_actions(env, DISCRETE_ACTIONS)

    # plot_simple(env)

    env = gym.make('RandMosquitoes-v0')
    plot_random_both(env)

    # for i, a in enumerate(ACTIONS):
    #     r = eval_constant_action(env, a, 10000)
    #     print(i, a, r[-1])
