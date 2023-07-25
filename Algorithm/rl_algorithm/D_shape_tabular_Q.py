import matplotlib.pyplot as plt
from datetime import datetime
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate, Dense
from collections import deque  # Used for replay buffer and reward tracking
from datetime import datetime  # Used for timing script
import time
from simulators.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure

ACTIONS = {0: "up", 1: "down", 2: "left", 3: "right"}


class Tabular_Q_Agent:
    def __init__(self, env):
        self.env = env
        self.actions = [i for i in range(self.env.action_space)]
        self.Q_table = np.random.rand(self.env.size, self.env.size, self.env.action_space)
        self.Q_goal_table = np.random.rand(self.env.size, self.env.size, self.env.size, self.env.size,
                                           self.env.action_space)
        # print(self.actions)

    def epsilon_greedy(self, state, goal, epsilon, d_shaping):
        if np.random.rand() < epsilon:
            return np.random.choice(self.actions)
        else:
            if not d_shaping:
                return np.argmax(self.Q_table[state[0]][state[1]])
            else:
                return np.argmax(self.Q_goal_table[state[0]][state[1]][goal[0]][goal[1]])

    def q_learning(self, num_of_step=50000, d_shaping=False, demo=None, pref_w=None):
        epsilon = 0.2
        alpha = 0.1
        gamma = 0.99
        steps = 0
        state_list = []
        episode_reward = 0

        image, position = self.env.reset()

        if not d_shaping:
            state = position
            goal = None
        else:
            state = position
            goal = demo[0]

        # print(state)
        all_rews = []

        while steps < num_of_step:

            if d_shaping:
                action = self.epsilon_greedy(state, goal, epsilon, d_shaping=d_shaping)
                d_goal_idx = min(steps, len(demo) - 1)
                n_d_goal_idx = min(d_goal_idx + 1, len(demo) - 1)
                rews, _, terminal, n_pos, _, t_r = self.env.step(action, d_goal=[demo[d_goal_idx], demo[n_d_goal_idx]])
                state_list.append(n_pos)
                reward = np.dot(rews, pref_w)
                n_state = n_pos
                n_goal = demo[n_d_goal_idx]
                TD_target = reward + gamma * np.max(self.Q_goal_table[n_state[0]][n_state[1]][n_goal[0]][n_goal[1]])
                TD_error = TD_target - self.Q_goal_table[state[0]][state[1]][goal[0]][goal[1]][action]
                self.Q_goal_table[state[0]][state[1]][goal[0]][goal[1]][action] += alpha * TD_error

            else:
                action = self.epsilon_greedy(state, epsilon, d_shaping=d_shaping)
                rews, n_image, terminal, n_pos, shaping_reward, t_r = self.env.step(action)
                reward = np.dot(rews, pref_w)
                n_state = n_pos
                self.Q_table[state[0]][state[1]][action] += alpha * (
                        reward + gamma * np.max(self.Q_table[n_state[0]][n_state[1]]) -
                        self.Q_table[state[0]][state[1]][action])

            episode_reward += t_r
            state = n_state

            if terminal:
                for state_idx in range(len(state_list) - 1):
                    n_state_idx = state_idx + 1
                    state = state_list[state_idx]
                    n_state = state_list[n_state_idx]

                    goal_idx = random.randint(0, len(state_list) - 2)
                    n_goal_idx = goal_idx + 1
                    goal = state_list[goal_idx]
                    n_goal = state_list[n_goal_idx]

                    reward = self.env.relabel_d_shape(state=state, n_state=n_state, goal=goal, n_goal=n_goal,
                                                      pref=pref_w)
                    TD_target = reward + gamma * np.max(self.Q_goal_table[n_state[0]][n_state[1]][n_goal[0]][n_goal[1]])
                    TD_error = TD_target - self.Q_goal_table[state[0]][state[1]][goal[0]][goal[1]][action]

                    self.Q_goal_table[state[0]][state[1]][goal[0]][goal[1]][action] += alpha * TD_error

                if len(all_rews) < 100:
                    all_rews.append(episode_reward)
                else:
                    all_rews.append(episode_reward)
                    all_rews[-1] = (np.mean(all_rews[-100:]))
                print(f"episodic r:{all_rews[-1]}")
                episode_reward = 0
                image, position = self.env.reset()

                if not d_shaping:
                    state = position
                    goal = None
                else:
                    state = position
                    goal = demo[0]
            steps += 1
        return all_rews

    def play_a_episode(self, goal):
        episode_reward = 0
        terminal = False
        image, position = self.env.reset()
        state = position
        while not terminal:
            action = self.epsilon_greedy(state, epsilon=0, d_shaping=False, goal=goal)
            rewards, n_image, terminal, n_position, shaping_reward, treasure_reward = self.env.step(action)
            next_state = n_position

            episode_reward += treasure_reward
            state = next_state
        print(f"Play episode, to goal:{goal} \t get episodic reward:{episode_reward} @ {n_position}")


def generate_traj(size):
    traj = []
    r = 0
    for r in range(size):
        for c in range(size):
            traj.append([r, c])
    # traj[-2] = [28,29]
    # traj[-3] = [28,29]
    # traj[-1] = [28,29]
    return traj


if __name__ == '__main__':
    # print(generate_traj(size = 10))
    # size = 30
    # goal_pos = 899
    num_of_step = 1000000

    env = DeepSeaTreasure()
    q_agent = Tabular_Q_Agent(env)
    all_rewards = q_agent.q_learning(num_of_step=num_of_step, pref_w=np.array([0.1, 0.9]))
    print(f"all_rewards:{all_rewards}")

    # # env = ImageGridWorld(size=size, goal_pos=goal_pos, shaping=True)
    # # q_agent = Tabular_Q_Agent(env)
    # # all_rewards_shape = q_agent.q_learning(num_of_step=num_of_step)
    #
    # env = ImageGridWorld(size=size, goal_pos=goal_pos, shaping=False, d_shaping=True)
    # q_agent = Tabular_Q_Agent(env)
    # all_rewards_d_shape = q_agent.q_learning(num_of_step=num_of_step, demonstration=generate_traj(size=size))
    # # print(f"all_rewards:{all_rewards}")
    # plt.plot(all_rewards, color="blue", alpha=0.7)
    # # plt.plot(all_rewards_shape, color = "red")
    # plt.plot(all_rewards_d_shape, color="green")
    # plt.show()
    # # state_pair = env.reset()
    # # state = state_pair[1]
    # # terminal = False
    # # epi_reward = 0
    # # while not terminal:
    # #     action = q_agent.epsilon_greedy(state, 0)
    # #     reward, next_state_pair, terminal = env.step(action)
    # #     next_state = next_state_pair[1]
    # #     state = next_state
    # #     epi_reward +=reward
    # #     # print(f"{ACTIONS[action]}")
    # # print(f"epi reward:{epi_reward}")
