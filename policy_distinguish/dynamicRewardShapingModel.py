import random

import tensorflow as tf
import numpy as np
from keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate, Dense, Reshape
from keras import Model, optimizers
from keras import metrics
# from Env.img_gw_env import ImageGridWorld
import matplotlib.pyplot as plt
import os
from simulators.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure
from simulators.deep_sea_treasure.preference_space import PreferenceSpace
from Algorithm.go_explore.explore import traj_cost_calculate
from keras import layers
from keras import backend as K
from keras.losses import mse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        # override the inherited .call(self, inputs) method @ Dr. James McDermott.
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))  # N(0, 1)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon  # N(mu, sigma)


class Encoder(Model):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__(self)
        self.hidden_0 = Dense(16, activation='relu')
        self.hidden_1 = Dense(16, activation='relu')
        self.hidden_2 = Dense(16, activation='relu')
        self.dense_0 = Dense(latent_dim)

    def call(self, inputs, training=None, mask=None):
        x = self.hidden_0(inputs)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        outputs = self.dense_0(x)
        return outputs


class Decoder(Model):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__(self)
        self.hidden_0 = Dense(16, activation='relu')
        self.hidden_1 = Dense(16, activation='relu')
        self.hidden_2 = Dense(16, activation='relu')
        self.dense_0 = Dense(latent_dim)

    def call(self, inputs, training=None, mask=None):
        x = self.hidden_0(inputs)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        outputs = self.dense_0(x)
        return outputs


class Auto_Encoder(Model):
    def __init__(self, encoder, decoder):
        super().__init__(self)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=None, mask=None):
        z = self.encoder(inputs)
        outputs = self.decoder(z)
        return outputs

    def evaluate_sample(self, sample):
        # print(self(sample))
        return tf.reduce_mean(metrics.mean_squared_error((self(sample)), sample)).numpy()


class VAE_Encoder(Model):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__(self)
        self.hidden_0 = Dense(16, activation='relu')
        self.hidden_1 = Dense(16, activation='relu')
        self.hidden_2 = Dense(16, activation='relu')
        self.z_mean = Dense(latent_dim, name="z_mean")
        self.z_var = Dense(latent_dim, name="z_log_var")
        self.z = Sampling()

    def call(self, inputs, training=None, mask=None):
        x = self.hidden_0(inputs)
        x = self.hidden_1(x)
        x = self.hidden_2(x)

        x = self.dense_0(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_var(x)
        z = self.z((z_mean, z_log_var))
        return z


class VAE_Decoder(Model):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__(self)
        self.hidden_0 = Dense(16, activation='relu')
        self.hidden_1 = Dense(16, activation='relu')
        self.hidden_2 = Dense(16, activation='relu')
        self.dense_0 = Dense(latent_dim)

    def call(self, inputs, training=None, mask=None):
        x = self.hidden_0(inputs)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        outputs = self.dense_0(x)
        return outputs


class VAE_Auto_Encoder(Model):
    def __init__(self, encoder, decoder):
        super().__init__(self)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=None, mask=None):
        z = self.encoder(inputs)
        outputs = self.decoder(z)
        return outputs

    def evaluate_sample(self, sample):
        # print(self(sample))
        return tf.reduce_mean(metrics.mean_squared_error((self(sample)), sample)).numpy()


if __name__ == '__main__':
    archive = np.load("C:/Users/19233436/PycharmProjects/MOGOExplore/simulation/deep_sea_treasure/archive/archive.npy",
                      allow_pickle=True).item()
    simulator = DeepSeaTreasure(img_repr=True)
    cross = "---------------Policy starts-------------------"
    pref_space = PreferenceSpace()
    pref_list = pref_space.iterate()
    pref_traj_score = {}
    pref_traj_rews = {}
    for pref in pref_list:
        # print(f"pref:{pref}")
        pref = tuple(pref)
        max_score = -np.inf
        max_traj = None
        for k in archive[pref].keys():
            if archive[pref][k].score > max_score and archive[pref][k].terminal:
                # print(f"pref:{pref}\treward:{archive[pref][k].reward_vec}\tscore:{archive[pref][k].score}")
                max_rews = archive[pref][k].reward_vec
                max_score = archive[pref][k].score
                max_traj = archive[pref][k].cell_traj
        if not pref == (0, 1):
            pref_traj_score[pref] = (max_traj, max_score)
            pref_traj_rews[pref] = tuple(max_rews)
    # print(f"pref_traj_rews:{pref_traj_rews}")

    # env = ImageGridWorld(size=10, goal_pos=99, max_steps=30, visualization=True)
    original_inputs = tf.keras.Input(shape=(2,), name="encoder_input")
    x = layers.Dense(16, activation="relu")(original_inputs)
    z_mean = layers.Dense(8, name="z_mean")(x)
    z_log_var = layers.Dense(8, name="z_log_var")(x)
    z = Sampling()((z_mean, z_log_var))
    VAE_encoder = tf.keras.Model(inputs=original_inputs, outputs=z, name="encoder")

    # @ Dr. James McDermott.
    # Define decoder model:
    # z -> hidden layer -> output
    latent_inputs = tf.keras.Input(shape=(8,), name="z_sampling")
    x = layers.Dense(16, activation="relu")(latent_inputs)
    outputs = layers.Dense(2)(x)
    VAE_decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name="decoder")

    # Build the VAE model
    outputs = VAE_decoder(z)
    vae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name="vae")
    vae.summary()

    # Loss approach from
    # https://gist.github.com/tik0/6aa42cabb9cf9e21567c3deb309107b7
    # the input of encoder should match the output of the decoder
    reconstruction_loss = mse(original_inputs, outputs)
    reconstruction_loss = 2 * K.mean(reconstruction_loss)
    kl_loss = -0.5 * tf.reduce_mean(
        z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)

    vae.add_loss(kl_loss)
    vae.add_metric(kl_loss, name='kl_loss', aggregation='mean')
    vae.add_loss(reconstruction_loss)
    vae.add_metric(reconstruction_loss, name='mse_loss', aggregation='mean')

    # Optimizer is Adam, learning rate is 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    vae.add_loss(kl_loss)
    vae.add_metric(kl_loss, name='kl_loss', aggregation='mean')
    vae.add_loss(reconstruction_loss)
    vae.add_metric(reconstruction_loss, name='mse_loss', aggregation='mean')
    vae.compile(optimizer)
    preference_list = []
    rew_vec_list = []
    for pref, rews in pref_traj_rews.items():  # treasure, step
        print(f"pref:{pref}\trews:{rews}")
        preference_list.append(np.array(pref))
        rew_vec_list.append(np.array([rews[0], rews[1]]))

    rew_vec_list = np.array(rew_vec_list)
    preference_list = np.array(preference_list)
    min_loss = 0.05
    threshold = 0.01
    corner_weights = []
    for i in range(len(preference_list)):
        print(f"The {i}th round, running >>>>>>>>>>>>>>pref{preference_list[i]}")
        rew = rew_vec_list[i]
        print(rew)
        reconstructed_rew = vae(np.expand_dims(rew, 0))
        loss = metrics.mean_squared_error(rew, reconstructed_rew)
        print(f"rewards:{rew}\treconstructed_reward:{reconstructed_rew}\tloss:{loss}\tthreshold:{min_loss}")
        if loss > threshold:
            corner_weights.append(preference_list[i])
            vae.set_weights([tf.random.normal(w.shape) for w in vae.get_weights()])
            # vae.decoder.set_weights([tf.random.normal(w.shape) for w in decoder.get_weights()])
            rews = np.array(
                [[rew[0] - random.randint(-5, 5) / 1000., rew[1] - random.randint(-5, 5) / 1000.] for i in range(64)])
            print("reset weights")
            # print(rews)
            while loss > min_loss:
                with tf.GradientTape() as tape:
                    reconstructed_rew = vae(rews)
                    loss = metrics.mean_squared_error(rews, reconstructed_rew)
                    loss = tf.reduce_mean(loss)
                    print(f"loss:{loss}")
                grads = tape.gradient(loss, vae.trainable_variables)
                optimizer.apply_gradients(zip(grads, vae.trainable_variables))
            # min_loss = loss
    print(f"corner weight:{corner_weights}")
