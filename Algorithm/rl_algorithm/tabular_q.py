import csv
import random

import line_profiler
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# from Environments.DeepSeaTreasure.DiscreteDST import DeepSeaTreasureEnvironment
from scipy.special import softmax

ACTIONS = {0: "up", 1: "down", 2: "left", 3: "right"}
Training_visualization_path = "C://Users//19233436//PycharmProjects//DWPI//Train_Agent//Train Result//discreteDST_Result//Learning_curve//"


class Tabular_Q_Agent:
    def __init__(self, env):
        self.env = env
        self.actions = [i for i in range(env.action_space_dim)]
        self.Q_table = self.initialise_q_values()

    def vanilla_epsilon_greedy(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.actions)
        else:
            return np.argmax(self.Q_table[state])

    def initialise_q_values(self):
        return np.random.rand(self.env.image_space[0] * self.env.image_space[1], self.env.action_space_dim)

    def q_learning(self, episodes, reset_to, traj):
        epsilon0 = 0.8
        alpha = 0.2
        gamma = 1
        phase_1_reward = self.env.reward_calc(traj)
        for i in range(episodes):
            state = self.env.reset_to_state(reset_to)
            epsilon = max(epsilon0 - i / episodes, 0.0001)
            terminal = False
            while not terminal:
                action = self.vanilla_epsilon_greedy(state, epsilon)
                reward, image, terminal, next_state = self.env.step(action)
                TD_error = reward + gamma * np.max(self.Q_table[next_state]) - self.Q_table[state][action]
                self.Q_table[state][action] += alpha * TD_error
                print(f"TD_error:{TD_error}\tfrom state:{state} go {ACTIONS[action]} to next_state:{next_state}")
                state = next_state

