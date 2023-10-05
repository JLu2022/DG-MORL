import random

import matplotlib.pyplot as plt
import numpy as np
from simulators.abstract_simulator import AbstractSimulator
from util.utils import coord_to_pos, pos_to_coord
from util.utils import ACTIONS


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

        row_1 = (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0)
        row_2 = (0.7, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0)
        row_3 = (0, 8.2, -1, -1, -1, -1, -1, -1, -1, -1, 0)
        row_4 = (0, 0, 11.5, -1, -1, -1, -1, -1, -1, -1, 0)
        row_5 = (0, 0, 0, 14.0, 15.1, 16.1, -1, -1, -1, -1, 0)
        row_6 = (0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0)
        row_7 = (0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0)
        row_8 = (0, 0, 0, 0, 0, 0, 19.6, 20.3, -1, -1, 0)
        row_9 = (0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0)
        row_10 = (0, 0, 0, 0, 0, 0, 0, 0, 22.4, -1, 0)
        row_11 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 23.7, 0)
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
        shaping_reward = 0
        treasure_reward = 0
        d_shaping_reward= 0
        terminal = False
        # print(f"r:{self.row}\tc:{self.col}")
        if d_goal:
            manhattan_goal = abs(self.row - d_goal[0][0]) + abs(self.col - d_goal[0][1])
        self.energy -= 1
        # manhattan_dist = abs(self.row - 10) + abs(self.col - 9)

        if action == 0 and self.row > 0 and not self.background_map[self.row - 1][self.col] == 0:
            self.row = self.row - 1
        if action == 1 and self.row < self.num_of_row - 1 and not self.background_map[self.row + 1][self.col] == 0:
            self.row = self.row + 1  # 0 Down, 1 Right
        elif action == 2 and self.col > 0 and not self.background_map[self.row][self.col - 1] == 0:
            self.col = self.col - 1
        elif action == 3 and self.col < self.num_of_col - 1 and not self.background_map[self.row][self.col + 1] == 0:
            self.col = self.col + 1

        if d_goal:
            manhattan_goal_prime = abs(self.row - d_goal[1][0]) + abs(self.col - d_goal[1][1])
            d_shaping_reward = manhattan_goal - manhattan_goal_prime

        # manhattan_dist_prime = abs(self.row - 10) + abs(self.col - 9)
        # shaping_reward = manhattan_dist - 0.99*manhattan_dist_prime
        # # rewards[1] = shaping_reward
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

    def relabel_d_shape(self, state, n_state, goal, n_goal, pref):
        rewards = np.zeros(2)
        rewards[0] = -1
        row = state[0]
        col = state[1]

        n_row = n_state[0]
        n_col = n_state[1]

        g_row = goal[0]
        g_col = goal[1]

        n_g_row = n_goal[0]
        n_g_col = n_goal[1]
        # print(goal)
        manhattan_goal = abs(row - g_row) + abs(col - g_col)
        manhattan_goal_prime = abs(n_row - n_g_row) + abs(n_col - n_g_col)
        # if n_row == self.goal_coord[0] and n_col == self.goal_coord[1]:
        #     r_task = 1
        # else:
        #     r_task = -1
        if not self.background_map[self.row][self.col] == 0 and not self.background_map[self.row][self.col] == -1:
            rewards[1] = self.background_map[self.row][self.col]
        # rewards/=124
        r_task = np.dot(pref, rewards)
        f_goal = manhattan_goal_prime - manhattan_goal
        reward = r_task + f_goal
        return reward

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


if __name__ == '__main__':
    dst_env = DeepSeaTreasure(visualization=True)
    dst_env.reset(put_submarine=False)
    # print(dst_env.show_available_position())
    print(dst_env.img_map)
    positions = []
    train_set = []
    # for row in range(11):
    #     for col in range(10):
    #         if dst_env.img_map[row][col] == -1:
    #             positions.append([row, col])
    # for position in positions:
    #     dst_env.reset(put_submarine=False)
    #     dst_env.row = position[0]
    #     dst_env.col = position[1]
    #     dst_env.add_submarine()
    #     train_set.append(np.array([dst_env.render_map(dst_env.img_map).flatten() / 255]))
    #
    #     img = dst_env.render_map(dst_env.img_map)
    #     plt.imshow(img)
    #     plt.show(block=False)
    #     plt.pause(0.2)
    #     plt.close()
    terminal = False
    image, position = dst_env.reset()
    print(f"position:{position}")
    plt.imshow(image)
    plt.show()
    # plt.pause(0.2)
    # plt.close()
    # while True:
    #     action = random.randint(0, 3)
    #     rewards, image, terminal, position = dst_env.step(action=action)
    #     print(f"reward:{rewards}\n"
    #           f"action:{ACTIONS[action]}\n"
    #           f"terminal:{terminal}\n"
    #           f"position:{position}\n"
    #           f"======================")
    #     plt.imshow(image)
    #     plt.show(block=False)
    #     plt.pause(0.4)
    #     plt.close()

    # image, pos = dst_env.reset_to_state(reset_to=(1, 3), put_submarine=True)
    rewards, image, terminal, position, d_shaping_reward, treasure_reward = dst_env.step(action=1)
    print(f"terminal:{terminal}")
    plt.imshow(image)
    plt.show()
