import random

import matplotlib.pyplot as plt
import numpy as np
from simulators.deep_sea_treasure.abstract_simulator import AbstractSimulator

GAMMA = 0.99


class DeepSeaTreasure(AbstractSimulator):
    def __init__(self, visualization=False, submarine_pos=None, num_of_row=11, num_of_col=11, img_repr=False):
        # CCDST
        # row_1 = (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0)
        # row_2 = (1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0)
        # row_3 = (0, 34, -1, -1, -1, -1, -1, -1, -1, -1, 0)
        # row_4 = (0, 0, 58, -1, -1, -1, -1, -1, -1, -1, 0)
        # row_5 = (0, 0, 0, 78, 86, 92, -1, -1, -1, -1, 0)
        # row_6 = (0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0)
        # row_7 = (0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0)
        # row_8 = (0, 0, 0, 0, 0, 0, 112, 116, -1, -1, 0)
        # row_9 = (0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0)
        # row_10 = (0, 0, 0, 0, 0, 0, 0, 0, 122, -1, 0)
        # row_11 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 124, 0)

        # row_1 = (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
        # row_2 = (0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
        # row_3 = (0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1)
        # row_4 = (0, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1)
        # row_5 = (0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1)
        # row_6 = (0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1)
        # row_7 = (0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1)
        # row_8 = (0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1)
        # row_9 = (0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1)
        # row_10 = (0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1)
        # row_11 = (0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1)
        # row_12 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        row_1 = (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
        row_2 = (0.7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
        row_3 = (0, 8.2, -1, -1, -1, -1, -1, -1, -1, -1, -1)
        row_4 = (0, 0, 11.5, -1, -1, -1, -1, -1, -1, -1, -1)
        row_5 = (0, 0, 0, 14.0, 15.1, 16.1, -1, -1, -1, -1, -1)
        row_6 = (0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1)
        row_7 = (0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1)
        row_8 = (0, 0, 0, 0, 0, 0, 19.6, 20.3, -1, -1, -1)
        row_9 = (0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1)
        row_10 = (0, 0, 0, 0, 0, 0, 0, 0, 22.4, -1, -1)
        row_11 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 23.7, -1)
        # self.num_of_row = num_of_row
        # self.num_of_col = num_of_col
        self.size = 11
        self.background_map = (row_1, row_2, row_3, row_4, row_5, row_6, row_7, row_8, row_9, row_10, row_11)
        self.img_map = [list(self.background_map[i]) for i in range(len(self.background_map))]

        # self.image = image

        if submarine_pos is None:
            self.submarine_pos = [0, 0]
        else:
            self.submarine_pos = submarine_pos
        self.row = self.submarine_pos[0]
        self.col = self.submarine_pos[1]
        # print(f"submarine:{submarine_pos}\trow:{self.row}\tcol:{self.col}")
        self.num_of_row = len(self.background_map)
        self.num_of_col = len(self.background_map[0])
        # print(f"num_row:{self.num_of_row}\tnum_col:{self.num_of_col}")
        self.observation_space_img = (self.num_of_row, self.num_of_col, 3)
        self.observation_space_discrete = 2
        if img_repr:
            self.observation_space = self.observation_space_img
        else:
            self.observation_space = self.observation_space_discrete
        self.action_space = 4
        self.add_submarine()
        self.visualization = visualization
        self.preference_token = "No pref"
        self.image_path = "DST/" + str(self.preference_token) + "/"
        self.energy = 100

    def render_map(self, img_map):
        image = np.zeros(self.observation_space_img)
        for row in range(0, self.num_of_row):
            for col in range(0, self.num_of_col):
                if img_map[row][col] == 0:
                    image[row][col][0] = 0
                    image[row][col][1] = 0
                    image[row][col][2] = 0
                elif img_map[row][col] == -1:  # available
                    image[row][col][0] = 0
                    image[row][col][1] = 191
                    image[row][col][2] = 255
                elif img_map[row][col] == 99:
                    image[row][col][0] = 255
                    image[row][col][1] = 0
                    image[row][col][2] = 0
                else:
                    image[row][col][0] = 255
                    image[row][col][1] = 255
                    image[row][col][2] = 0
        return image

    def show_available_position(self, row_index, num_sample):
        available_position = []
        # for row in range(self.num_col):
        count = 0
        while count < num_sample:
            col = random.choice(range(self.num_of_col))
            if self.img_map[row_index][col] == -1 or self.img_map[row_index][col] == 99:
                available_position.append([row_index, col])
                count += 1
        return available_position

    def reset(self, put_submarine=True):

        self.row = self.submarine_pos[0]
        self.col = self.submarine_pos[1]
        self.img_map = [list(self.background_map[i]) for i in range(self.num_of_row)]
        if put_submarine:
            self.add_submarine()
        image = self.render_map(self.img_map)
        image /= 255
        self.energy = 100
        position = (self.row, self.col)
        return image, position

    def reset_to_state(self, reset_to, put_submarine=True):
        self.row = reset_to[0]
        self.col = reset_to[1]
        self.img_map = [list(self.background_map[i]) for i in range(self.num_of_row)]
        if put_submarine:
            self.add_submarine()
        image = self.render_map(self.img_map)
        image /= 255
        self.energy = 100
        position = (self.row, self.col)
        # preference = (reset_to[2], reset_to[3])
        return image, position

    def add_submarine(self):
        self.img_map[self.row][self.col] = 99

    def step(self, action, episode=1001, d_goal=None):  # 0:up 1:down 2:left 3:right
        self.img_map = [list(self.background_map[i]) for i in range(self.num_of_row)]
        rewards = np.zeros(2)
        rewards[0] = -1
        treasure_reward = 0
        d_shaping_reward = 0
        terminal = False
        if d_goal:
            manhattan_goal = abs(self.row - d_goal[0][0]) + abs(self.col - d_goal[0][1])
        self.energy -= 1

        if action == 0 and self.row > 0 and not self.background_map[self.row - 1][self.col] == 0:
            self.row = self.row - 1
        if action == 1 and self.row < self.num_of_row - 1 and not self.background_map[self.row + 1][self.col] == 0:
            self.row = self.row + 1
        elif action == 2 and self.col > 0 and not self.background_map[self.row][self.col - 1] == 0:
            self.col = self.col - 1
        elif action == 3 and self.col < self.num_of_col - 1 and not self.background_map[self.row][self.col + 1] == 0:
            self.col = self.col + 1

        if d_goal:
            manhattan_goal_prime = abs(self.row - d_goal[1][0]) + abs(self.col - d_goal[1][1])
            d_shaping_reward = manhattan_goal - manhattan_goal_prime

        if not self.background_map[self.row][self.col] == 0 and not self.background_map[self.row][self.col] == -1:
            rewards[1] = self.background_map[self.row][self.col]
            treasure_reward = rewards[1]
            terminal = True
        if self.energy <= 0:
            terminal = True

        self.add_submarine()
        image = self.render_map(self.img_map)
        image /= 255
        # position = self.row * self.num_of_col + self.col
        position = (self.row, self.col)

        return rewards, image, terminal, position, d_shaping_reward, treasure_reward
        # return rewards, (image, (self.row, self.col)), terminal,shaping_reward

    def visualize(self):
        my_ticks_x = np.arange(0, self.num_of_col, 1)
        my_ticks_y = np.arange(0, self.num_of_row, 1)
        plt.xticks(my_ticks_x)
        plt.yticks(my_ticks_y)
        plt.imshow(self.img_map)
        plt.savefig("try" + ".png")

    def set_visualization(self, visualization=False):
        self.visualization = visualization

    def set_preference(self, preference_token):
        self.preference_token = preference_token
        self.image_path = "/content/drive/MyDrive/Colab Notebooks/Source Code/AAMAS-2023/Route/" + str(
            self.preference_token) + "/"

    def check_forbidden(self, action, player_pos):
        row = player_pos[0]
        col = player_pos[1]
        if action == "up" and row > 0 and not self.background_map[row - 1][col] == 0:
            return False
        if action == "down" and row < self.num_of_row and not self.background_map[row + 1][col] == 0:
            return False
        elif action == "left" and col > 0 and not self.background_map[row][col - 1] == 0:
            return False
        elif action == "right" and col < self.num_of_col and not self.background_map[row][col + 1] == 0:
            return False
        else:
            return True

    def calculate_pos(self, player_pos):
        row = player_pos[0]
        col = player_pos[1]
        return (row * self.num_of_col + col) / 100

    def calculate_reward(self, player_pos):
        row = player_pos[0]
        col = player_pos[1]
        rewards = np.zeros(2)
        if not self.background_map[row][col] == 0 and not self.background_map[row][col] == -1:
            rewards += np.array([-1, self.background_map[row][col]])
        else:
            rewards += np.array([-1, 0])
        return rewards

    def get_settings(self, action=None):
        return [self.img_map, {-1: (0, 191, 255),
                               0: (0, 0, 0),
                               99: (0, 191, 255),
                               1: (255, 255, 0),
                               34: (255, 255, 0),
                               58: (255, 255, 0),
                               78: (255, 255, 0),
                               86: (255, 255, 0),
                               92: (255, 255, 0),
                               112: (255, 255, 0),
                               116: (255, 255, 0),
                               122: (255, 255, 0),
                               124: (255, 255, 0)}]

    def check_terminal(self, player_pos):
        row = player_pos[0]
        col = player_pos[1]
        if not self.background_map[row][col] == 0 and not self.background_map[row][col] == -1:
            return True
        else:
            return False

    def calculate_utility(self, demo, pref_w):
        rewards = np.zeros(2)
        pure_rewards = np.zeros(2)
        gamma = 1
        for pos in demo:
            row = pos[0]
            col = pos[1]
            if not self.background_map[row][col] == 0 and not self.background_map[row][col] == -1:
                rewards += gamma * np.array([-1, self.background_map[row][col]])
                pure_rewards += np.array([-1, self.background_map[row][col]])
                # print(f"reach:{self.background_map[row][col]}")
            else:
                rewards += gamma * np.array([-1, 0])
                pure_rewards += np.array([-1, 0])
            gamma *= GAMMA
        utility = np.dot(rewards, pref_w)
        return utility, pure_rewards

    def calculate_utility_from_actions(self, action_demo, pref_w):
        self.reset()
        gamma = 1
        value_vec = np.zeros(2)
        for action in action_demo:
            rewards, _, terminal, _, _, _ = self.step(action=action)
            value_vec += rewards * gamma
            gamma *= GAMMA
        value_scalar = np.dot(value_vec, pref_w)
        return value_scalar, value_vec

    def state_traj_to_actions(self, state_demo):
        action_list = []
        # print(state_demo)
        for i in range(len(state_demo) - 1):
            move = np.array(state_demo[i + 1]) - np.array(state_demo[i])
            if move[0] == -1 and move[1] == 0:
                action_list.append(0)
            if move[0] == 1 and move[1] == 0:
                action_list.append(1)
            if move[0] == 0 and move[1] == -1:
                action_list.append(2)
            if move[0] == 0 and move[1] == 1:
                action_list.append(3)
            if move[0] == 0 and move[1] == 0:
                if state_demo[i][0] == 0:
                    action_list.append(0)
                if state_demo[i][0] == 10:
                    action_list.append(1)
                if state_demo[i][1] == 10:
                    action_list.append(3)
                if state_demo[i][0] == 5 and state_demo[i][1] == 6:
                    action_list.append(2)
                if state_demo[i][0] == 6 and state_demo[i][1] == 6:
                    action_list.append(2)
                if state_demo[i][0] == 8 and state_demo[i][1] == 8:
                    action_list.append(2)
            # print(np.array(state_demo[i + 1]) - np.array(state_demo[i]))
        # print(action_list)
        return action_list


# 0:up 1:down 2:left 3:right


if __name__ == '__main__':
    dst_env = DeepSeaTreasure(visualization=True)
    dst_env.reset(put_submarine=False)
    traj_to_10_9 = [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
                    (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (8, 9), (9, 9), (10, 9)]

    traj_to_9_8 = [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
                   (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8), (8, 8), (9, 8)]

    traj_to_7_7 = [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 7),
                   (3, 7), (4, 7), (5, 7), (6, 7), (7, 7)]

    traj_to_7_6 = [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 6), (3, 6),
                   (4, 6), (5, 6), (6, 6), (7, 6)]

    traj_to_4_5 = [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 5), (3, 5), (4, 5)]
    traj_to_4_4 = [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (4, 4)]
    traj_to_4_3 = [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (4, 3)]
    traj_to_3_2 = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (3, 2)]
    traj_to_2_1 = [(0, 0), (0, 1), (1, 1), (2, 1)]
    traj_to_1_0 = [(0, 0), (1, 0)]
    trajs = [traj_to_10_9, traj_to_9_8, traj_to_7_7, traj_to_7_6, traj_to_4_5, traj_to_4_4, traj_to_4_3, traj_to_3_2,
             traj_to_2_1, traj_to_1_0]
    treasure_w = 0.0
    sum_utility = 0
    # pref_list = [1, 0.7, 0.67, 0.66, 0.58, 0.54, 0.51, 0.47, 0.39, 0.21]
    for i in range(101):
        # for treasure_w in pref_list:
        treasure_w = round((100 - i) / 100, 2)
        utility_list = []
        for traj in trajs:
            utility = dst_env.calculate_utility(demo=traj[1:], pref_w=np.array([1 - treasure_w, treasure_w]))
            utility_list.append(utility)
        sum_utility += max(utility_list)
        idx = np.argmax(utility_list)
        print(f"pref:{[1 - treasure_w, treasure_w]}\tmax_utility:{max(utility_list)}\tpos:{10 - idx}")
    print(f"avg utility:{sum_utility / 101}")
    dst_env.state_traj_to_actions(traj_to_4_5)
