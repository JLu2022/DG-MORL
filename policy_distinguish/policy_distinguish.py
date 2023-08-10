import copy
import random

import tensorflow as tf
import numpy as np
from keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate, Dense, Reshape
from keras import Model, optimizers
from keras import metrics
from Algorithm.rl_algorithm.backward_Q_agent import Tabular_Q_Agent
import matplotlib.pyplot as plt
import os
from simulators.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure
from simulators.deep_sea_treasure.preference_space import PreferenceSpace
from keras import layers
from keras import backend as K
from keras.losses import mse

# from sklearn.cluster import KMeans
# from sklearn.neighbors import LocalOutlierFactor
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
random.seed(121)
np.random.seed(121)
tf.random.set_seed(121)
input_dim = 2
encoding_dim = 16
hidden_dim = 8
latent_dim = 8
utility_list = []
true_corner_weights = [[0.49, 0.51], [0.66, 0.34], [0.79, 0.21], [0.83, 0.17], [0.85, 0.15], [0.88, 0.12],
                       [0.90, 0.10], [0.92, 0.08], [0.94, 0.06]]
corner_reward_list = []
cross = "---------------Policy starts-------------------"


def policy_distinguish(autoencoder, preference_list, trajectory_set, epochs=100, batch_size=32):
    corner_weight_idx = []
    pass_loss = -np.inf
    eval_losses = [-np.inf]
    train_repertoire = np.array([])
    corner_weights = []
    initial_weights = autoencoder.get_weights()
    loss = 0
    max_loss = 0
    for i in range(len(preference_list)):
        # print("trajectory_set[i]", trajectory_set[i])
        print(f"[trajectory_set[i][0]]:{[trajectory_set[i][0]]}")
        predictions = autoencoder(np.array([trajectory_set[i][0]]))
        eval_loss = tf.reduce_mean(mse(np.array([trajectory_set[i][0]]), predictions))
        print(
            f"pref:{preference_list[i]}rews{rew_vec_list[i]}"
            # f"avg_rew:{np.mean(trajectory_set[i], axis=0)}\t "
            f"reconstruct:{np.round(predictions, 2)}"
            f"eval loss:{eval_loss}\t"
            f"train_repertoire content{len(train_repertoire)}"
        )
        if eval_loss > pass_loss:  # adding some scaling loss for the eval loss,say 10% greater than pass loss.
            autoencoder.set_weights(initial_weights)
            eval_losses = []
            print("Meet new")
            corner_weight_idx.append(i)
            corner_weights.append(preference_list[i])
            for epoch in range(epochs + 1):
                with tf.GradientTape() as tape:
                    predictions = autoencoder(trajectory_set[i])
                    losses = mse(trajectory_set[i], predictions)
                    loss = tf.reduce_mean(mse(trajectory_set[i], predictions))
                    max_loss = tf.reduce_max(mse(trajectory_set[i], predictions))
                gradients = tape.gradient(loss, autoencoder.trainable_variables)
                optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))

                if epoch % 100 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, max_loss: {max_loss}")
        pass_loss = max_loss
        eval_losses.append(eval_loss)
    return corner_weights, corner_weight_idx


if __name__ == '__main__':
    # Auto-encoder model
    input_data = Input(shape=(input_dim,))
    hidden_0 = Dense(encoding_dim, activation='relu')(input_data)
    hidden_1 = Dense(hidden_dim, activation='relu')(hidden_0)
    z = Dense(latent_dim)(hidden_1)
    hidden_2 = Dense(hidden_dim, activation='relu')(z)
    hidden_3 = Dense(hidden_dim, activation='relu')(hidden_2)
    output_data = Dense(input_dim)(hidden_3)

    autoencoder = Model(input_data, output_data)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Load in the archive
    archive = np.load("../Algorithm/go_explore/archives/archive.npy", allow_pickle=True).item()
    archive = dict(archive)
    simulator = DeepSeaTreasure(img_repr=True)

    pref_space = PreferenceSpace()
    raw_pref_list = pref_space.iterate()
    pref_traj_score = {}
    pref_traj_rews = {}

    for pref in raw_pref_list:
        pref = tuple(pref)
        max_score = -np.inf
        max_traj = None
        for k in archive[pref].keys():
            if archive[pref][k].score > max_score and archive[pref][k].terminal:
                max_rews = archive[pref][k].reward_vec
                max_score = archive[pref][k].score
                max_traj = archive[pref][k].cell_traj
        # if not pref == (0, 1):
        pref_traj_score[pref] = (max_traj, max_score)
        pref_traj_rews[pref] = tuple(max_rews)

    preference_list = []
    rew_vec_list = []

    for pref, rews in pref_traj_rews.items():  # treasure, step
        preference_list.append(np.array(pref))
        rew_vec_list.append(np.array([rews[0], rews[1]]))
        print(f"pref:{pref}|rews:{rews}|utility:{np.dot(rews, pref)}")
    rew_vec_list = np.array(rew_vec_list)
    rew_data = np.expand_dims(rew_vec_list, axis=0)
    # print(f"reward_vec_list:{rew_data}")

    # Pre-train
    pre_train_dataset = tf.data.Dataset.from_tensor_slices((rew_data, rew_data))
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    autoencoder.fit(pre_train_dataset, epochs=1000)
    rew_data = []
    mask = np.array([0.002, 0.998])  # filter the traj
    for rew in rew_vec_list:
        rew = np.array(rew)
        rews = []
        for i in range(64):
            if i > 0:
                noise = np.random.randint(-2, 1, rew.shape)
                # noise[1] = 0
                rews.append((rew + noise) * mask)
        # print(f"rews:{rews}")
        rew_data.append(rews)
    rew_data = np.array(rew_data)
    # print(rew_data)
    # for rew in rew_data:
    #     print(f"rew:{rew}")
    # rew_data = rew_data * mask
    # rew_data = np.transpose(rew_data, (1, 0, 2))
    # for rew in rew_data:
    #     print(f"rew:{rew}")
    # print(f"rew:{rew_data}")
    # Distinguish policy from the trajs
    corner_weights, corner_weight_idx = policy_distinguish(autoencoder, preference_list, trajectory_set=rew_data,
                                                           epochs=100,
                                                           batch_size=32)

    print(f"corner:{corner_weights}")
    x_values = [point[0] for point in corner_weights]
    y_values = [point[1] for point in corner_weights]
    print(x_values)
    print(y_values)
    true_x_values = [point[0] for point in true_corner_weights]
    true_y_values = [point[1] for point in true_corner_weights]
    print(rew_vec_list)
    for idx in corner_weight_idx:
        print(f"corner weights:{preference_list[idx]}\t"
              f"rew:{list(rew_vec_list[idx])}\t")

        # utility_list.append(np.dot(preference_list[idx], np.array(list(dataset.as_numpy_iterator())[idx])))
        corner_reward_list.append(list(rew_vec_list[idx]))

    print(f"corner_reward_list:{corner_reward_list}")
    corner_reward_list = np.array(corner_reward_list)

    np.save("corner_weights.npy", corner_weights)

    plt.scatter(x_values, y_values, color='b', marker='o',
                label='corner weights')

    plt.scatter(true_x_values, true_y_values, color='r', marker='*', s=30,
                label='ground truth corner weights')
    plt.xlabel('time')
    plt.ylabel('treasure')
    plt.legend()
    plt.show()
    # print(pref_traj_score)
    # pref_list = np.array(pref_list[::-1])
    robustified_rewards = []
    agent_list = []
    trajs = []
    for w in corner_weights:
        print(cross)
        agent = Tabular_Q_Agent(env=simulator)
        agent_list.append(agent)
        pref = tuple(w)
        traj = pref_traj_score[pref][0]
        # expected_utility_lists = []
        # for pref_index in range(len(pref_list)):
        #     agent = agent_list[pref_index]
        print(f"demo traj:{traj}\tpref:{pref}")
        steps, reward_list, expected_utility_list = agent.imitate_q_(demo=traj, pref_w=np.array(w),
                                                                     agent_list=agent_list)
        # expected_utility_lists += expected_utility_list
        # print(f"pref:{pref_list[pref_index]}\treward_list:{reward_list}")
        _, episode_rewards = agent.play_a_episode(pref=np.array(w), agent=agent)
        robustified_rewards.append(episode_rewards)
    # plt.plot(expected_utility_lists)
    # plt.show()
    robustified_rewards = np.array(robustified_rewards)
    plt.scatter(corner_reward_list[:, 0], corner_reward_list[:, 1], color='b', marker='o',
                label='points found by explore')

    plt.scatter(robustified_rewards[:, 0], robustified_rewards[:, 1], color='r', marker='o',
                label='points found by robustification')
    plt.xlabel('time')
    plt.ylabel('treasure')
    plt.legend()
    plt.show()
