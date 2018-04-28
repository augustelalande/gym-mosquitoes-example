import numpy as np


class MemBuffer(object):
    def __init__(self, length):
        self.length = length
        self.features = [None for _ in range(length)]
        self.next_features = [None for _ in range(length)]
        self.actions = [0 for _ in range(length)]
        self.rewards = [0 for _ in range(length)]
        self.index = 0
        self.full = False

    def add(self, feature, action, reward, next_feature):
        self.features[self.index] = feature
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_features[self.index] = next_feature
        self.index += 1
        self.index %= self.length
        if self.index == self.length - 1:
            self.full = True

    def sample(self, n):
        indices = np.random.randint(0, self.length, n)
        s = [self.features[i] for i in indices]
        a = [self.actions[i] for i in indices]
        r = [self.rewards[i] for i in indices]
        s_n = [self.next_features[i] for i in indices]
        return np.stack(s), np.array(a), np.array(r), np.stack(s_n)
