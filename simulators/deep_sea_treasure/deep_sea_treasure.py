import random
import numpy as np
from simulators.deep_sea_treasure.abstract_simulator import AbstractSimulator
import matplotlib.pyplot as plt

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
        return position, None

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
        rewards[1] = -1
        terminal = False
        self.energy -= 1

        if action == 0 and self.row > 0 and not self.background_map[self.row - 1][self.col] == 0:
            self.row = self.row - 1
        if action == 1 and self.row < self.num_of_row - 1 and not self.background_map[self.row + 1][self.col] == 0:
            self.row = self.row + 1
        elif action == 2 and self.col > 0 and not self.background_map[self.row][self.col - 1] == 0:
            self.col = self.col - 1
        elif action == 3 and self.col < self.num_of_col - 1 and not self.background_map[self.row][self.col + 1] == 0:
            self.col = self.col + 1

        if not self.background_map[self.row][self.col] == 0 and not self.background_map[self.row][self.col] == -1:
            rewards[0] = self.background_map[self.row][self.col]
            terminal = True
        if self.energy <= 0:
            terminal = True

        self.add_submarine()
        image = self.render_map(self.img_map)
        image /= 255
        position = (self.row, self.col)
        return position, rewards, terminal, None, None

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
            rewards += np.array([self.background_map[row][col]], -1)
        else:
            rewards += np.array([0, -1])
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
                rewards += gamma * np.array([self.background_map[row][col]], -1)
                pure_rewards += np.array([self.background_map[row][col]], -1)
            else:
                rewards += gamma * np.array([0, -1])
                pure_rewards += np.array([0, -1])
            gamma *= GAMMA
        utility = np.dot(rewards, pref_w)
        return utility, pure_rewards

    def calculate_utility_from_actions(self, action_demo, pref_w):
        self.reset()
        gamma = 1
        value_vec = np.zeros(2)
        for action in action_demo:
            position, rewards, terminal, _, _ = self.step(action=action)
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
    action_demo_1 = [1]  # 0.7
    action_demo_2 = [3, 1, 1]  # 8.2
    action_demo_3 = [3, 3, 1, 1, 1]  # 11.5
    action_demo_4 = [3, 3, 3, 1, 1, 1, 1]  # 14.0
    action_demo_5 = [3, 3, 3, 3, 1, 1, 1, 1]  # 15.1
    action_demo_6 = [3, 3, 3, 3, 3, 1, 1, 1, 1]  # 16.1
    action_demo_7 = [3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1]  # 19.6
    action_demo_8 = [3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1]  # 20.3
    action_demo_9 = [3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 22.4
    action_demo_10 = [3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 23.7
    action_demos = [action_demo_1, action_demo_2, action_demo_3, action_demo_4, action_demo_5, action_demo_6,
                    action_demo_7, action_demo_8, action_demo_9, action_demo_10]

    treasure_w = 0.0
    sum_utility = 0
    # pref_list = [1, 0.7, 0.67, 0.66, 0.58, 0.54, 0.51, 0.47, 0.39, 0.21]
    return_list = []
    treasure_ws = []
    value_vecs = []
    for i in range(101):
        # for treasure_w in pref_list:
        treasure_w = round((100 - i) / 100, 2)
        utility_list = []
        for demo in action_demos:
            value_scalar, value_vec = dst_env.calculate_utility_from_actions(action_demo=demo,
                                                                             pref_w=np.array(
                                                                                 [1 - treasure_w, treasure_w]))
            # value_vecs.append(value_vec)
            utility_list.append(value_scalar)
        print(f"w_vec:{[1 - treasure_w, treasure_w]}\tV*S(w):{max(utility_list)}\tdemo:{np.argmax(utility_list)}")
    #     sum_utility += max(utility_list)
    #     idx = np.argmax(utility_list)
    #     print(f"pref:{[1 - treasure_w, treasure_w]}\tmax_utility:{max(utility_list)}\tpos:{10 - idx}")
    #     return_list.append(max(utility_list))
    #     treasure_ws.append(treasure_w)
    # print(f"return_list:{return_list}\ttreasure_ws:{treasure_ws}")
    # return_list = return_list[::-1]
    # treasure_ws = treasure_ws[::-1]
    # print(f"avg utility:{sum_utility / 101}")
    # # dst_env.state_traj_to_actions(traj_to_4_5)
    # plt.plot(treasure_ws, return_list, color="red")
    # plt.show()
