import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from util.utils import find_best_traj
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
random.seed(121)
np.random.seed(121)
tf.random.set_seed(121)
input_dim = 2
encoding_dim = 16
hidden_dim = 8
latent_dim = 8
utility_list = []
true_corner_weights = [[0.29, 0.71], [0.32, 0.68], [0.79, 0.21], [0.83, 0.17], [0.85, 0.15], [0.88, 0.12],
                       [0.90, 0.10], [0.92, 0.08], [0.94, 0.06]]
corner_reward_list = []
cross = "---------------Policy starts-------------------"


if __name__ == '__main__':
    archive = np.load("archives/archive.npy", allow_pickle=True).item()
    archive = dict(archive)
    pref_traj_score, pref_traj_rews, rew_vec_list, preference_list = find_best_traj(archive=archive)
    print(f"rew_vec_list:{rew_vec_list}")
    points = rew_vec_list

    filter_values = []
    corner_weight_idx = []
    edge_points = []
    for i in range(len(points)):
        p = points[i]
        add = True
        for j in range(len(points)):
            p_ = points[j]
            if p[1] == p_[1] and p_[0] > p[0] and not i == j:
                add = False
            if p[1] in filter_values:
                add = False
        if add:
            edge_points.append(points[i])
            filter_values.append(points[i][1])
            corner_weight_idx.append(i)

        print(f"edge_points:{edge_points}")
        print(f"filter_values:{filter_values}")
        print(f"corner_weight_idx:{corner_weight_idx}")

    corner_weights = []
    for idx in corner_weight_idx:
        corner_weights.append(preference_list[idx])
    print(corner_weights)

    np.save("files/corner_weights.npy", corner_weights)
    x_values = [point[0] for point in rew_vec_list]
    y_values = [point[1] for point in rew_vec_list]
    plt.scatter(x_values, y_values, color='b', marker='o',
                label='corner weights')
    x_points = [point[0] for point in edge_points]
    y_points = [point[1] for point in edge_points]
    plt.scatter(x_points, y_points, color='r', marker='*',
                label='corner weights')

    plt.xlabel('time')
    plt.ylabel('treasure')
    plt.legend()
    plt.show()
