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
from simulators.deep_sea_treasure.preference_space import PreferenceSpace

ACTIONS = {0: "up", 1: "down", 2: "left", 3: "right"}
pref_space = PreferenceSpace()
pref_list = pref_space.iterate()


class Tabular_Q_Agent:
    def __init__(self, env):
        self.env = env
        self.actions = [i for i in range(self.env.action_space)]
        self.Q_table = np.random.rand(self.env.size, self.env.size, self.env.action_space)

    def epsilon_greedy(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.actions)
        else:

            return np.argmax(self.Q_table[state[0]][state[1]])

    def imitate_q(self, num_of_step=50000, pref_w=None, demo=None, agent_list=None):
        epsilon = 0.5
        alpha = 0.1
        gamma = 1
        steps = 0
        reward_list = []
        expected_utility_list = []
        for reset_idx in range(len(demo) - 1)[::-1]:
            # get the sub_demo to calculate the reward bar
            sub_demo = demo[reset_idx + 1:]
            rews_threshold = np.zeros(2)
            for demo_s in sub_demo:
                rews_threshold += self.env.calculate_reward(player_pos=demo_s)
            rew_threshold = np.dot(rews_threshold, pref_w)

            evaluation_rew = -np.inf
            rew_threshold = round(float(rew_threshold), 2)
            print(f"sub_demo:{sub_demo}, reward bar:{rew_threshold}, pref:{pref_w}, reset_to:{demo[reset_idx]}")
            step = 0

            while evaluation_rew < rew_threshold and step < 100:  # while the agent does not learn a policy not worse than the demo
                step += 1

                image, state = self.env.reset_to_state(reset_to=demo[reset_idx])  # reset to the start point
                terminal = False

                while not terminal:
                    steps += 1
                    action = self.epsilon_greedy(state, epsilon)
                    rews, n_image, terminal, n_state, shaping_reward, t_r = self.env.step(action)
                    reward = np.dot(rews, pref_w)

                    # -------------------Train------------------#
                    TD_target = reward + gamma * np.max(self.Q_table[n_state[0]][n_state[1]])
                    TD_error = TD_target - self.Q_table[state[0]][state[1]][action]
                    self.Q_table[state[0]][state[1]][action] += alpha * TD_error
                    state = n_state

                evaluation_rew = self.play_sub_episode(reset_to=demo[reset_idx], pref=pref_w)
        return steps, reward_list, expected_utility_list

    def imitate_q_(self, num_of_step=50000, pref_w=None, demo=None, agent_list=None):
        epsilon = 0.5
        alpha = 0.1
        gamma = 1
        steps = 0
        reward_list = []
        expected_utility_list = []
        # divide the demo into 4 part:
        num_of_parts = len(demo) // 4
        init_s = demo[0]
        reset_to_ss = []
        reset_to_ss.append(init_s)

        for i in range(num_of_parts + 1)[::-1]:
            reset_idx = i * 3
            # reset_to_ss.append(demo[reset_idx*4])
            sub_demo = demo[reset_idx + 1:]
            rews_threshold = np.zeros(2)
            for demo_s in sub_demo:
                rews_threshold += self.env.calculate_reward(player_pos=demo_s)
            evaluation_rew = -np.inf
            rew_threshold = round(np.dot(rews_threshold, pref_w), 2)
            print(f"sub_demo:{sub_demo}, reward bar:{rew_threshold}, pref:{pref_w}, reset_to:{demo[reset_idx]}")
            step = 0

            while evaluation_rew < rew_threshold:  # and step < 50:  # while the agent does not learn a policy not worse than the demo
                step += 1

                image, state = self.env.reset_to_state(reset_to=demo[reset_idx])  # reset to the start point
                terminal = False

                while not terminal:
                    steps += 1
                    action = self.epsilon_greedy(state, epsilon)
                    rews, n_image, terminal, n_state, shaping_reward, t_r = self.env.step(action)
                    reward = np.dot(rews, pref_w)

                    # -------------------Train------------------#
                    TD_target = reward + gamma * np.max(self.Q_table[n_state[0]][n_state[1]])
                    TD_error = TD_target - self.Q_table[state[0]][state[1]][action]
                    self.Q_table[state[0]][state[1]][action] += alpha * TD_error
                    state = n_state

                evaluation_rew, state_list = self.play_sub_episode(reset_to=demo[reset_idx], pref=pref_w)
                if step % 50 == 0:
                    print(f"evaluation_rew:{evaluation_rew}\tstate_list:{state_list}")
            print(f"evaluation_rew:{evaluation_rew}\tstate_list:{state_list}")
        return steps, reward_list, expected_utility_list

    def play_sub_episode(self, reset_to, pref):
        episode_reward = 0
        terminal = False
        image, state = self.env.reset_to_state(reset_to=reset_to)
        state_list = [state]
        while not terminal:
            action = self.epsilon_greedy(state, epsilon=0)
            rewards, n_image, terminal, n_state, shaping_reward, treasure_reward = self.env.step(action)
            # print(f"rewards:{rewards}")
            episode_reward += np.dot(rewards, pref)
            state = n_state
            state_list.append(state)
        # print(f"state_list:{state_list}")
        # print(f"evaluation_rew:{round(episode_reward, 2)}")
        return round(episode_reward, 2), state_list
        # print(f"Play episode, reset_to:{reset_to} \t get episodic reward:{episode_reward} @ {n_state}")

    def play_a_episode(self, pref, agent):
        env_try = DeepSeaTreasure()
        episode_reward = 0
        terminal = False
        image, state = env_try.reset()
        while not terminal:
            action = agent.epsilon_greedy(state, epsilon=0)
            rewards, n_image, terminal, n_state, shaping_reward, treasure_reward = env_try.step(action)

            episode_reward += np.dot(rewards, pref)
            state = n_state
        print(f"Play episode, to goal:{pref} \t get episodic reward:{episode_reward} @ {state}")
        return round(episode_reward, 2)

    def select_agent_from_pref(self, agent_list, w):
        thresholds = [0.51, 0.34, 0.21, 0.17, 0.15, 0.12, 0.1, 0.08, 0.06, 0]
        # for w in pref_list:
        #     print(f"pref:{w} | {sum(w[1] < thresholds)}")
        return agent_list[sum(w[1] < thresholds)]


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
    steps, reward_list = q_agent.imitate_q(
        demo=[(0, 0), (0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)],
        pref_w=np.array([0.84, 0.16]))
    print(reward_list)
    plt.plot(range(steps), reward_list)
    plt.show()
    # q_agent.play_a_episode(pref=np.array([0.84, 0.16]))

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
