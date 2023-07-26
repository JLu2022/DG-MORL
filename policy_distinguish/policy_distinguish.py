import random

import tensorflow as tf
import numpy as np
from keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate, Dense, Reshape
from keras import Model, optimizers
from keras import metrics
import matplotlib.pyplot as plt
import os
from simulators.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure
from simulators.deep_sea_treasure.preference_space import PreferenceSpace
from keras import layers
from keras import backend as K
from keras.losses import mse

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

if __name__ == '__main__':
    archive = np.load("D:/PhD/Project/MOGE/Algorithm/go_explore/archives/grid_world/archive.npy",
                      allow_pickle=True).item()
    archive = dict(archive)
    simulator = DeepSeaTreasure(img_repr=True)

    pref_space = PreferenceSpace()
    pref_list = pref_space.iterate()
    pref_traj_score = {}
    pref_traj_rews = {}

    for pref in pref_list:
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
        print(f"pref:{pref}|rews:{rews}")

    # 将列表转换为 TensorFlow 数据集
    dataset = tf.data.Dataset.from_tensor_slices((rew_vec_list))


    # 数据增强函数，为每个样本添加随机噪声
    def data_augmentation(item1):
        num_replicas = 64
        # 创建一个数组来保存所有的样本和噪声
        augmented_item1 = []
        # augmented_item2 = []
        for _ in range(num_replicas):
            # 添加随机噪声，这里以高斯分布的噪声为例
            noise1 = np.random.normal(loc=0, scale=0.5, size=item1.shape)
            # noise2 = np.random.normal(loc=0.0, scale=0.1, size=item2.shape)
            augmented_item1.append(item1 + noise1)
            # augmented_item2.append(item2 + noise2)

        # 使用 tf.stack 将增强后的样本堆叠在一起，并保持原始维度
        augmented_item1 = tf.stack(augmented_item1)
        # augmented_item2 = tf.stack(augmented_item2)

        # 返回增强后的数据
        return augmented_item1


    # 对数据集进行数据增强
    dataset = dataset.map(data_augmentation)
    dataset = list(dataset.as_numpy_iterator())
    # print(f"len:{len(dataset)}")
    # 打印前几个样本
    # for item1 in dataset.take(5):
    #     print("Item 1:", item1.numpy())
    epochs = 1000
    batch_size = 64
    threshold = 0.1
    corner_weights = []
    pass_loss = -np.inf
    eval_losses = [-np.inf]
    for i in range(len(pref_list)-1):
        # print(f"data:{dataset[i]}")

        predictions = autoencoder(dataset[i])
        eval_loss = tf.reduce_mean(mse(dataset[i], predictions))

        print(f"pref:{pref_list[i]}rews:{rew_vec_list[i]}avg_rew:{np.mean(dataset[i],axis=0)}\t eval loss:{eval_loss}"
              # f"\tavg:{np.mean(eval_losses)}"
              )
        if eval_loss > pass_loss:
            print(f"avg:{np.mean(eval_losses)}")
            eval_losses = []
            print("Meet new")
            corner_weights.append(pref_list[i])
            # 保存初始权重
            initial_weights = autoencoder.get_weights()
            # 重置权重为随机初始化的值
            autoencoder.set_weights([tf.random.normal(w.shape) for w in initial_weights])

            for epoch in range(epochs+1):

                # 开启 GradientTape，跟踪前向传播过程，计算梯度
                with tf.GradientTape() as tape:
                    predictions = autoencoder(dataset[i])
                    loss = tf.reduce_mean(mse(dataset[i], predictions))
                    # print(f"predictions:{predictions}\toriginal:{dataset[i]}")
                    # 计算梯度
                gradients = tape.gradient(loss, autoencoder.trainable_variables)

                # 应用梯度，更新模型参数
                optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))

                if epoch % 100 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        pass_loss = eval_loss
        eval_losses.append(eval_loss)
    print(f"corner weights:{corner_weights}")
    # rew_vec_list = np.array(rew_vec_list)
    # preference_list = np.array(preference_list)
    # min_loss = 0.05
    # threshold = 0.05
    # corner_weights = []
    #
    # for i in range(len(preference_list)):
    #     print(f"The {i}th round, running >>>>>>>>>>>>>>pref{preference_list[i]}")
    #     rew = rew_vec_list[i]/25
    #     reconstructed_rew = autoencoder(np.expand_dims(rew,0))
    #     loss = metrics.mean_squared_error(rew, reconstructed_rew)
    #     train_batch = []
    #     for _ in range(32):
    #         rew_noisy = add_noise(rew)
    #         train_batch.append(rew_noisy)
    #     train_batch = np.array(train_batch)
    #     print(f"train_batch shape:{train_batch.shape}")
    #     print(f"rewards:{rew}\t"
    #           f"reconstructed_reward:{reconstructed_rew}\t"
    #           f"loss:{loss}\t"
    #           f"threshold:{min_loss}")
    #
    #     if loss > threshold:
    #         # print(f"reward:{rew}")
    #         corner_weights.append(preference_list[i])
    #         autoencoder.set_weights([tf.random.normal(w.shape) for w in autoencoder.get_weights()])
    #         print("reset weights")
    #         total_loss = 0
    #         while loss > min_loss:
    #             loss = train_step(train_batch)
    #             # total_loss += loss
    #
    #             # avg_loss = total_loss / num_batches
    #             print(loss)
    #             # loss = train_step(rew)
    #             # print(f"loss:{loss}")
    # print(f"corner weight:{corner_weights}")
