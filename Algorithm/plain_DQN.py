import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate, Dense
from keras import metrics
from keras.optimizers import Adam
from collections import deque  # Used for replay buffer and reward tracking
from simulators.minecart.minecart_simulator import Minecart
import matplotlib.pyplot as plt
from datetime import datetime  # Used for timing script

# from simulators.energy_sim.energysimulator import EnergySimulator

SEED = 42
BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 1000
GAMMA = 0.98
TRAINING_EPISODES = 5000
EXPLORATION_RESTARTS = 0
EPSILON_START = 1
EPSILON_END = 0.01
EPSILON_DECAY = 1 / (TRAINING_EPISODES * 0.98)
COPY_TO_TARGET_EVERY = 50  # Steps
START_TRAINING_AFTER = 10  # Episodes
FRAME_STACK_SIZE = 3
NUM_WEIGHTS = 2
SAVE_MODEL_PER = 1000


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
        self.actions = range(6)
        self.gamma = GAMMA  # Discount
        self.eps0 = EPSILON_START  # Epsilon greedy init
        self.model_path = model_path
        self.batch_size = BATCH_SIZE
        self.replay_memory = ReplayMemory(maxlen=REPLAY_MEMORY_SIZE)
        self.checkpoint = checkpoint
        self.input_size = 7
        self.output_size = 6
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
            action = np.argmax(Q_values)
            return action

    def play_one_step(self, state, epsilon, pref_w):
        action = self.epsilon_greedy_policy(state, epsilon)
        # self.action_list.append(action)
        n_state, rews, terminal, _, _ = self.env.step(action)
        reward = np.dot(rews, pref_w)
        self.replay_memory.append([state, action, reward, n_state, terminal])
        return n_state, reward, terminal

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

    def train_model(self, episodes, save_per=100000, show_detail_per=100):
        """
        Train the network over a range of episodes.
        """
        steps = 0
        pref_w = np.array([0.7, 0.1, 0.2])
        for episode in range(1, episodes + 1):
            # self.action_list = []
            eps = max(self.eps0 - episode * EPSILON_DECAY, EPSILON_END)
            # Reset env
            state, _ = self.env.reset()
            state = np.float32(state)  # Convert to float32 for tf

            episode_reward = 0
            while True:
                n_state, reward, terminal = self.play_one_step(state, eps, pref_w)
                steps += 1
                episode_reward += reward
                if terminal:
                    break
            if episode > START_TRAINING_AFTER:  # Wait for buffer to fill up a bit
                self.training_step(episode)
                if episode % COPY_TO_TARGET_EVERY == 0:
                    self.target_model.set_weights(self.model.get_weights())
                if episode % save_per == 0 and episode >= save_per and self.checkpoint:
                    self.model.save(self.model_path + str(episode))
            if episode % show_detail_per == 0:
                print(f"Episode:{episode}\t"
                      f"Episodic utility:{episode_reward}\t"
                      f"Epsilon:{np.round(eps, 2)}\t")

    def jsmoDQN(self, pref_w=None, demo=None):
        epsilon = 0.7
        expected_utility_list = []
        train_cnt = 0
        episode = 0

        overall_utility_thres, _ = self.env.calculate_utility(demo=demo, pref_w=pref_w)
        print(f"overall_utility_thres:{overall_utility_thres}")

        utility = -np.inf
        h_pointer = len(demo) - 1
        while h_pointer >= 0:
            action_list = demo[:h_pointer]
            while utility < overall_utility_thres:
                terminal = False
                state, _ = self.env.reset()

                for action in action_list:  # guide policy takes over
                    n_state, rews, terminal, _, _ = self.env.step(action)
                    reward = np.dot(rews, pref_w)
                    self.replay_memory.append([state, action, reward, n_state, terminal])
                    state = n_state

                while not terminal:  # explore policy takes over
                    action = self.epsilon_greedy_policy(state, epsilon)
                    n_state, rews, terminal, _, _ = self.env.step(action)
                    reward = np.dot(rews, pref_w)
                    self.replay_memory.append([state, action, reward, n_state, terminal])
                    state = n_state
                episode += 1
                if episode > START_TRAINING_AFTER:  # Wait for buffer to fill up a bit
                    self.training_step(episode)
                    train_cnt += 1
                    if episode % COPY_TO_TARGET_EVERY == 0:
                        self.target_model.set_weights(self.model.get_weights())
                    if episode % SAVE_MODEL_PER == 0 and episode >= SAVE_MODEL_PER and self.checkpoint:
                        self.model.save(self.model_path + str(episode))

                utility, traj = self.evaluate(pref_w=pref_w)
                expected_utility_list.append(utility)
                if train_cnt % 1000 == 0:
                    print(f"train cnt:{train_cnt}\tutility:{utility}")
            h_pointer -= 1

        utility, traj = self.evaluate(pref_w=pref_w)
        print(f"good traj:{traj}\tu:{utility}")
        print("==========================================")
        return expected_utility_list

    def evaluate(self, pref_w):
        terminal = False
        state, _ = self.env.reset()
        vec_rewards = np.zeros(3)
        action_list = []
        gamma = 1
        while not terminal:
            action = self.epsilon_greedy_policy(state, epsilon=0)
            n_state, rews, terminal, _, _ = self.env.step(action)
            vec_rewards += gamma * rews
            gamma *= self.gamma
            state = n_state
            action_list.append(action)
        utility = np.dot(vec_rewards, pref_w)
        return utility, action_list


if __name__ == '__main__':
    explore_episodes = 1000
    simulator = Minecart()
    agent = DQNAgent(env=simulator, model_path="../train/minecart/model/")
    agent.train_model(episodes=10000)
    print("Training process finish, start Evaluation...")
    utility, action_list = agent.evaluate(pref_w=np.array([0.7, 0.1, 0.2]))
    print(f"utility:{utility}\nactions:{action_list}")
    np.save("../train/minecart/traj/pure_DQN_tryout/actions.npy", action_list)
