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
# 设置 Python 的随机种子
random.seed(121)

# 设置 NumPy 的随机种子
np.random.seed(121)

# 设置 TensorFlow 的随机种子
tf.random.set_seed(121)
input_dim = 2
encoding_dim = 16
hidden_dim = 8
latent_dim = 8

# 创建去噪自编码器模型
input_data = Input(shape=(input_dim,))  # input_dim是输入数据的维度
hidden_0 = Dense(encoding_dim, activation='relu')(input_data)
hidden_1 = Dense(hidden_dim, activation='relu')(hidden_0)
z = Dense(latent_dim)(hidden_1)
hidden_2 = Dense(hidden_dim, activation='relu')(z)
hidden_3 = Dense(hidden_dim, activation='relu')(hidden_2)
output_data = Dense(input_dim)(hidden_3)
autoencoder = Model(input_data, output_data)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

cross = "---------------Policy starts-------------------"
current_directory = os.getcwd()
print(f"cur:{current_directory}")
if __name__ == '__main__':
    archive = np.load("../Algorithm/go_explore/archives/archive.npy", allow_pickle=True).item()
    archive = dict(archive)
    simulator = DeepSeaTreasure(img_repr=True)

    pref_space = PreferenceSpace()
    raw_pref_list = pref_space.iterate()
    pref_traj_score = {}
    pref_traj_rews = {}

    for pref in raw_pref_list:
        # print(f"pref:{pref}")
        pref = tuple(pref)
        max_score = -np.inf
        max_traj = None
        # print(list(archive.keys()))
        for k in archive[pref].keys():
            if archive[pref][k].score > max_score and archive[pref][k].terminal:
                # print(f"pref:{pref}\treward:{archive[pref][k].reward_vec}\tscore:{archive[pref][k].score}")
                max_rews = archive[pref][k].reward_vec
                max_score = archive[pref][k].score
                max_traj = archive[pref][k].cell_traj
        if not pref == (0, 1):
            pref_traj_score[pref] = (max_traj, max_score)
            pref_traj_rews[pref] = tuple(max_rews)

    preference_list = []
    rew_vec_list = []

    for pref, rews in pref_traj_rews.items():  # treasure, step
        preference_list.append(np.array(pref))
        rew_vec_list.append(np.array([rews[0], rews[1]]))
        print(f"pref:{pref}|rews:{rews}|utility:{np.dot(rews,pref)}")

    # 将列表转换为 TensorFlow 数据集
    dataset = tf.data.Dataset.from_tensor_slices((rew_vec_list))


    # 数据增强函数，为每个样本添加随机噪声
    def data_augmentation(item1):
        num_replicas = 32
        # 创建一个数组来保存所有的样本和噪声
        augmented_item1 = []
        # augmented_item2 = []
        for _ in range(num_replicas):
            # 添加随机噪声，这里以高斯分布的噪声为例
            noise1 = np.random.randint(low=-1, high=1, size=item1.shape)
            # noise1 = np.random.normal(loc=-3, scale=0.5, size=item1.shape)
            # noise1[0] = 0
            # noise1[1] = 0

            # item1[0] += random.randint(-3,1)
            # noise2 = np.random.normal(loc=0.0, scale=0.1, size=item2.shape)
            augmented_item1.append(item1 + noise1)
            # augmented_item2.append(item2 + noise2)

        # 使用 tf.stack 将增强后的样本堆叠在一起，并保持原始维度
        augmented_item1 = tf.stack(augmented_item1)
        # augmented_item2 = tf.stack(augmented_item2)

        # 返回增强后的数据
        return augmented_item1


    # 对数据集进行数据增强
    dataset_aug = dataset.map(data_augmentation)
    dataset_aug = list(dataset_aug.as_numpy_iterator())
    # print(f"len:{len(dataset)}")
    # 打印前几个样本
    # for item1 in dataset.take(5):
    #     print("Item 1:", item1.numpy())
    epochs = 100
    batch_size = 32
    threshold = 0.1
    corner_weight_idx = []
    pass_loss = -np.inf
    eval_losses = [-np.inf]
    train_repitore = np.array([])
    utility_list = []
    corner_weights = []
    corner_reward_list = []
    # print(dataset)
    # dataset = np.array(dataset)
    # # 使用LOF算法进行异常检测
    # lof = LocalOutlierFactor(n_neighbors=10, contamination=0.25)  # n_neighbors为邻居数，contamination为异常点比例
    # y_pred = lof.fit_predict(dataset)
    # scores = lof.negative_outlier_factor_
    #
    # # 绘制异常检测结果
    # plt.scatter(dataset[:, 0], dataset[:, 1], color='b', marker='o', label='normal')
    # plt.scatter(dataset[y_pred == -1][:, 0], dataset[y_pred == -1][:, 1], color='r', marker='o', label='abnormal')
    #
    # # 绘制异常点的LOF分数
    # radius = (scores.max() - scores) / (scores.max() - scores.min())  # 归一化LOF分数
    # plt.scatter(dataset[:, 0], dataset[:, 1], s=100 * radius, edgecolors='r', facecolors='none', label='LOF score')
    #
    # for i, label in enumerate(dataset):
    #     plt.annotate(label, (dataset[i, 0], dataset[i, 1]), textcoords="offset points", xytext=(0, 5), ha='center')
    #
    # plt.xlabel('step penalty')
    # plt.ylabel('treasure')
    # plt.legend()
    # plt.show()

    for i in range(len(preference_list) - 1):
        # print(f"data:{dataset[i]}")

        predictions = autoencoder(dataset_aug[i])
        eval_loss = tf.reduce_mean(mse(dataset_aug[i], predictions))

        print(f"pref:{preference_list[i]}rews:{rew_vec_list[i]}avg_rew:{np.mean(dataset_aug[i], axis=0)}\t eval loss:{eval_loss}\t:train_repitore content:{len(train_repitore)}"
              # f"\tavg:{np.mean(eval_losses)}"
              )
        if eval_loss > pass_loss:
            if len(train_repitore) > 5*batch_size:
                train_repitore = np.array([])
            train_repitore = np.append(train_repitore, dataset_aug[i])
            train_repitore = train_repitore.reshape(-1, 2)
            # print(f"avg:{np.mean(eval_losses)}")
            eval_losses = []
            print("Meet new")
            corner_weight_idx.append(i)
            corner_weights.append(preference_list[i])
            for epoch in range(epochs + 1):
                # dataset[i] = np.array(dataset[i])
                np.random.shuffle(train_repitore)
                # 开启 GradientTape，跟踪前向传播过程，计算梯度
                for i in range(0, len(train_repitore), batch_size):
                    with tf.GradientTape() as tape:
                        predictions = autoencoder(train_repitore[i:i+batch_size])
                        loss = tf.reduce_mean(mse(train_repitore[i:i+batch_size], predictions))
                        # print(f"predictions:{predictions}\toriginal:{dataset[i]}")
                        # 计算梯度
                    gradients = tape.gradient(loss, autoencoder.trainable_variables)

                    # 应用梯度，更新模型参数
                    optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))

                if epoch % 100 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        pass_loss = loss
        eval_losses.append(eval_loss)
    for idx in corner_weight_idx:
        # print(np.array(list(dataset.as_numpy_iterator())[idx]).shape)
        print(f"corner weights:{preference_list[idx]}\trew:{list(dataset.as_numpy_iterator())[idx]}\t utility:{np.dot(preference_list[idx], np.array(list(dataset.as_numpy_iterator())[idx]))}")
        utility_list.append(np.dot(preference_list[idx], np.array(list(dataset.as_numpy_iterator())[idx])))
        corner_reward_list.append(list(dataset.as_numpy_iterator())[idx])
    print(f"corner_reward_list:{corner_reward_list}")
    corner_reward_list =np.array(corner_reward_list)


    np.save("corner_weights.npy",corner_weights)
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