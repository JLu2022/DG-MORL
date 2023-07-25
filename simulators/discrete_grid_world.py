import random

import matplotlib.pyplot as plt
import numpy as np
from simulators.abstract_simulator import AbstractSimulator
from util.utils import coord_to_pos, pos_to_coord
from util.utils import ACTIONS


class ImageGridWorld(AbstractSimulator):
    def __init__(self, size=(5, 5), goal_pos=24, max_steps=30, visualization=False):
        self.size = size
        self.image_space = np.array([self.size[0], self.size[1], 3])
        self.observation_space = np.array([1])
        self.action_space = np.array(range(0, 4))
        self.action_space_dim = 4
        self.goal_coords = pos_to_coord(self.size, goal_pos)
        self.img_map = -np.ones(self.size)
        self.img_map[self.goal_coords[0]][self.goal_coords[1]] = 1

        self.row = 0
        self.col = 0

        # print(f"observation_space:{self.observation_space}")
        # print(f"img_map:{self.img_map}")

        self.add_agent()
        self.visualization = visualization
        self.preference_token = "No pref"
        self.image_path = "DST/" + str(self.preference_token) + "/"
        self.max_steps = max_steps

    def render_map(self, img_map):
        image = np.zeros(self.image_space)
        for row in range(0, self.image_space[0]):
            for col in range(0, self.image_space[1]):
                if img_map[row][col] == 0:
                    image[row][col][0] = 0
                    image[row][col][1] = 0
                    image[row][col][2] = 0
                elif img_map[row][col] == -1:
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

    def reset(self, add_agent=True):
        self.row = 0
        self.col = 0
        self.img_map = -np.ones(self.size)
        self.img_map[self.goal_coords[0]][self.goal_coords[1]] = 1
        if add_agent:
            self.add_agent()
        image = self.render_map(self.img_map)
        image /= 255
        self.max_steps = 30
        return coord_to_pos(self.size, (self.row, self.col)), image

    def reset_to_state(self, reset_to):
        row, col = pos_to_coord(size=self.size, pos=reset_to)
        self.row = row
        self.col = col
        self.max_steps = 30
        # print(f"row:{self.row}, col:{self.col},reset to:{row, col}")
        return reset_to, None

    def add_agent(self):
        self.img_map[self.row][self.col] = 99

    def step(self, action):  # 0:up 1:down 2:left 3:right
        self.img_map = -np.ones(self.size)
        self.img_map[self.goal_coords[0]][self.goal_coords[1]] = 1
        reward = -1
        terminal = False
        self.max_steps -= 1
        if action == 0 and self.row > 0:
            self.row = self.row - 1
        if action == 1 and self.row < self.size[0] - 1:
            self.row = self.row + 1
        elif action == 2 and self.col > 0:
            self.col = self.col - 1
        elif action == 3 and self.col < self.size[1] - 1:
            self.col = self.col + 1

        reward += self.img_map[self.row][self.col]
        if not self.img_map[self.row][self.col] == -1:
            terminal = True

        if self.max_steps <= 0:
            terminal = True
        self.add_agent()
        image = self.render_map(self.img_map)
        image /= 255

        return coord_to_pos(self.size, (self.row, self.col)), reward, terminal, image, None

    def visualize(self):
        my_ticks_x = np.arange(0, self.image_space[0], 1)
        my_ticks_y = np.arange(0, self.image_space[1], 1)
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

    def reward_calc(self, traj):
        reward = 0
        for state in traj:
            coord = pos_to_coord(self.size, state)
            reward += self.img_map[coord[0]][coord[1]]
        return reward


if __name__ == '__main__':
    dst_env = ImageGridWorld(visualization=True, size=(5, 5), goal_pos=24)
    for row in range(5):
        row_str = ""
        for col in range(5):
            row_str += str(coord_to_pos((5, 5), (row, col))) + "\t"
            if col % 4 == 0 and not col == 0:
                print(row_str + "\n-----------------------\n")

    for i in range(25):
        for action in range(4):
            dst_env.reset_to_state(reset_to=i)
            pos, reward, terminal, image = dst_env.step(action)
            print(f"from state:{i} go {ACTIONS[action]} to state:{pos}")
            print("--------------")
    # init_s = 19

    # dst_env.reset_to_state(reset_to=init_s)
    # reward, image, terminal, pos = dst_env.step(3)
    # print(f"from state:{init_s} go {ACTIONS[3]} to state:{pos}")
    # dst_env.reset_to_state(reset_to=init_s)
    # reward, image, terminal, pos = dst_env.step(2)
    # print(f"from state:{init_s} go {ACTIONS[2]} to state:{pos}")
    # dst_env.reset_to_state(reset_to=init_s)
    # reward, image, terminal, pos = dst_env.step(1)
    # print(f"from state:{init_s} go {ACTIONS[1]} to state:{pos}")
    # reward, image, terminal, pos = dst_env.step(0)
    # print(f"from state:{init_s} go {ACTIONS[0]} to state:{pos}")
    # print("--------------")
    # init_s = 23
    #
    # dst_env.reset_to_state(reset_to=init_s)
    # reward, image, terminal, pos = dst_env.step(3)
    # print(f"from state:{init_s} go {ACTIONS[3]} to state:{pos}")
    # dst_env.reset_to_state(reset_to=init_s)
    # reward, image, terminal, pos = dst_env.step(2)
    # print(f"from state:{init_s} go {ACTIONS[2]} to state:{pos}")
    # dst_env.reset_to_state(reset_to=init_s)
    # reward, image, terminal, pos = dst_env.step(1)
    # print(f"from state:{init_s} go {ACTIONS[1]} to state:{pos}")
    # reward, image, terminal, pos = dst_env.step(0)
    # print(f"from state:{init_s} go {ACTIONS[0]} to state:{pos}")
    # image = dst_env.reset(add_agent=True)
    # plt.imshow(image)
    # plt.show()
    # plt.close()
    # for _ in range(20):
    #     reward, image, terminal, pos = dst_env.step(random.randint(0, 4))
    #     plt.imshow(image)
    #     plt.title("position: "+str(pos))
    #     plt.show()
    #     plt.close()
