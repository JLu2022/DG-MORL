import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate, Dense
from keras import metrics
from keras.optimizers import Adam
from collections import deque  # Used for replay buffer and reward tracking
import matplotlib.pyplot as plt
from datetime import datetime  # Used for timing script

# from simulators.energy_sim.energysimulator import EnergySimulator

SEED = 42
DEBUG = False

BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 1000

GAMMA = 1

TRAINING_EPISODES = 5000
EXPLORATION_RESTARTS = 0

EPSILON_START = 1
EPSILON_END = 0.01
EPSILON_DECAY = 1 / (TRAINING_EPISODES * 0.98)
COPY_TO_TARGET_EVERY = 50  # Steps
START_TRAINING_AFTER = 10  # Episodes
FRAME_STACK_SIZE = 3
NUM_WEIGHTS = 2


class ReplayMemory(deque):
    def sample(self, batch_size):
        indices = np.random.randint(len(self), size=batch_size)
        batch = [self[index] for index in indices]
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch]) for field_index in range(5)]
        return states, actions, rewards, next_states, dones


class DQNAgent:
    def __init__(self, env, model_path=None, checkpoint=True):
        self.dynamic_reward_shaping_optimizer = Adam(learning_rate=1e-3)
        self.env = env
        self.actions = [i for i in range(self.env.action_space)]
        self.gamma = GAMMA  # Discount
        self.eps0 = EPSILON_START  # Epsilon greedy init
        self.model_path = model_path
        self.batch_size = BATCH_SIZE
        self.replay_memory = ReplayMemory(maxlen=REPLAY_MEMORY_SIZE)
        self.checkpoint = checkpoint
        self.input_size = self.env.observation_space_img
        self.output_size = self.env.action_space
        # Build both models
        self.model = self.build_model()
        self.target_model = self.build_model()
        # Make weights the same
        self.target_model.set_weights(self.model.get_weights())

    def build_model(self):
        # Define Layers
        input_layer = Input(shape=self.input_size)
        x = input_layer
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2, name="drop_0")(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2, name="drop_1")(x)
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.2, name="drop_2")(x)
        x = Dense(self.output_size)(x)
        outputs = x

        # Build full model
        model = keras.Model(inputs=input_layer, outputs=outputs)
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        self.loss_fn = keras.losses.mean_squared_error
        model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        return model

    def epsilon_greedy_policy(self, state, epsilon):
        if np.random.rand() < epsilon:
            return random.choice(self.actions)
        else:
            Q_values = self.model(state[np.newaxis], training=False)
            # novelty_potential = self.dynamic_reward_shaping.evaluate_sample(np.expand_dims(state, 0))
            action = np.argmax(Q_values)
            return action

    def play_one_step(self, state, epsilon):
        action = self.epsilon_greedy_policy(state, epsilon)
        self.action_list.append(action)
        next_state, reward, done, _, _ = self.env.step(action)
        self.replay_memory.append([state, action, reward, next_state, done])
        return next_state, reward, done

    def play_one_step_mo(self, state, epsilon, pref):
        action = self.epsilon_greedy_policy(state, epsilon)
        self.action_list.append(action)
        rewards, image, done, next_state = self.env.step(action)
        reward = np.dot(pref, rewards)
        self.replay_memory.append([state, action, reward, next_state, done])
        return next_state, reward, done

    def training_step(self, episode):
        experiences = self.replay_memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = experiences

        # Compute target Q values from 'next_states'
        next_Q_values = self.target_model(next_states, training=False)

        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards + (1 - dones) * self.gamma * max_next_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1)  # Make column vector

        # Mask to only consider action taken
        mask = tf.one_hot(actions, self.output_size)  # Number of actions
        # Compute loss and gradient for predictions on 'states'
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            q_loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(q_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def train_model(self, episodes, save_per=100000, show_detail_per=1000):
        """
        Train the network over a range of episodes.
        """
        steps = 0

        for episode in range(1, episodes + 1):
            self.action_list = []
            eps = max(self.eps0 - episode * EPSILON_DECAY, EPSILON_END)
            # Reset env
            state, _ = self.env.reset()
            state = np.float32(state)  # Convert to float32 for tf

            episode_reward = 0
            while True:
                state, reward, done = self.play_one_step(state, eps)
                steps += 1
                episode_reward += reward
                if done:
                    break
            if episode > START_TRAINING_AFTER:  # Wait for buffer to fill up a bit
                self.training_step(episode)
                if episode % COPY_TO_TARGET_EVERY == 0:
                    self.target_model.set_weights(self.model.get_weights())
                if episode % save_per == 0 and episode >= save_per and self.checkpoint:
                    self.model.save(self.model_path + str(episode))
            if episode % show_detail_per == 0:
                if sum(self.action_list) > 0:
                    indices_of_one = [i for i, x in enumerate(self.action_list) if x == 1]
                else:
                    indices_of_one = "Not run at all"
                print(
                    f"Epoch:{episode}\tEpoch Reward/Cost:{episode_reward}\tEpsilon:{np.round(eps, 2)}\tActions:{sum(self.action_list)}\t@{indices_of_one}")

    def train_model_with_traj(self, episodes, save_per=100000, show_detail_per=1000, traj=None, pref=None,
                              reward_bar=None, reset_to=None):
        """
        Train the network over a range of episodes.
        """
        steps = 0
        for episode in range(1, episodes + 1):
            self.action_list = []
            eps = max(self.eps0 - episode * EPSILON_DECAY, EPSILON_END)
            # Reset env
            image, state = self.env.reset()
            state = np.float32(state)  # Convert to float32 for tf

            episode_reward = 0
            while True:
                next_state, reward, done = self.play_one_step_mo(state, eps, pref=pref)
                steps += 1
                episode_reward += reward
                if done:
                    break

            if episode > START_TRAINING_AFTER:  # Wait for buffer to fill up a bit
                self.training_step(episode)
                if episode % COPY_TO_TARGET_EVERY == 0:
                    self.target_model.set_weights(self.model.get_weights())
                if episode % save_per == 0 and episode >= save_per and self.checkpoint:
                    self.model.save(self.model_path + str(episode))

    def generate_experience(self, day=2):
        self.action_list = []
        # Reset env
        state, _ = self.env.reset_to(reset_to=day)
        state = np.float32(state)  # Convert to float32 for tf

        episode_reward = 0

        while True:
            state, reward, done = self.play_one_step(state, 0.0)
            episode_reward += reward
            if done:
                break
        return episode_reward

# if __name__ == '__main__':
#     np.random.seed(SEED)
#     tf.random.set_seed(SEED)
#     random.seed(SEED)
#     env = EnergySimulator(tariffs_path="../simulators/energy_sim/Dataset/Tariffs.csv",
#                           background_path="../simulators/energy_sim/Dataset/HomeC-meter1_2014.csv",
#                           renewable_path="../simulators/energy_sim/Dataset/HomeC-meter1_2014.csv",
#                           num_day_train=1)
#
#     dqn_ag = DQNAgent(env, model_path="../Agent/AgentModel")
#
#     start_time = datetime.now()
#     dqn_ag.train_model(TRAINING_EPISODES, save_per=10000, show_detail_per=100)
#     total_cost = 0
#     for day in range(1, 31):
#         total_cost += dqn_ag.generate_experience(day=day)
#     print(f"total cost:{total_cost}")
#     # dqn_ag.generate_experience(day=1)
#     # dqn_ag.generate_experience(day=2)
#     # dqn_ag.generate_experience(day=3)
#     # dqn_ag.generate_experience(day=30)
#     run_time = datetime.now() - start_time
#     print(f'Plain DQN_training Run time: {run_time} s')
