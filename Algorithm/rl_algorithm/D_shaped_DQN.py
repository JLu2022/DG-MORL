import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate, Dense
from keras import metrics
from keras.optimizers import Adam
from collections import deque  # Used for replay buffer and reward tracking
from scipy.special import softmax

ACTIONS = ["up", "down", "left", "right"]
SEED = 42
DEBUG = False

BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 6000

GAMMA = 0.99

TRAINING_EPISODES = 2000
EXPLORATION_RESTARTS = 0
EPSILON_START = 1
EPSILON_END = 0.2
# EPSILON_DECAY = 1 / (TRAINING_EPISODES * 0.98)
COPY_TO_TARGET_EVERY = 100  # Steps
START_TRAINING_AFTER = 100  # Episodes
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
        self.input_size = self.env.observation_space
        self.output_size = self.env.action_space
        # Build both models
        self.model = self.build_plain_model()
        self.target_model = self.build_plain_model()
        # Make weights the same
        self.target_model.set_weights(self.model.get_weights())

    def build_plain_model(self):
        # Define Layers
        input_layer = Input(shape=self.input_size)
        x = input_layer
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        # img_input = Input(shape=self.input_size)
        # # header = keras.models.load_model("C://PhD//2023//Inverse_MORL//MOA2C//E_model")
        # header.trainable = False
        # x = header.layers[2](img_input)
        # x = header.layers[3](x)
        # x = header.layers[4](x)
        # x = header.layers[5](x)
        # x = header.layers[6](x)
        # x = header.layers[7](x)
        x = Dense(16, activation='relu')(x)
        # x = Dropout(0.2, name="drop_0")(x)
        x = Dense(8, activation='relu')(x)
        # x = Dropout(0.2, name="drop_1")(x)
        # x = Dense(8, activation='relu')(x)
        # x = Dropout(0.2, name="drop_2")(x)
        x = Dense(self.output_size)(x)
        outputs = x

        # Build full model
        model = keras.Model(inputs=input_layer, outputs=outputs)
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        self.loss_fn = keras.losses.mean_squared_error
        model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        return model

    def build_goal_conditioned_model(self, goal_size=None):
        # Define Layers
        state_input = Input(shape=self.input_size)  # need revision
        goal_input = Input(shape=(None, goal_size))
        input_layer = Concatenate()([state_input, goal_input])
        x = input_layer
        x = Dense(32, activation='relu')(x)
        # x = Dropout(0.2, name="drop_0")(x)
        x = Dense(32, activation='relu')(x)
        # x = Dropout(0.2, name="drop_1")(x)
        x = Dense(16, activation='relu')(x)
        # x = Dropout(0.2, name="drop_2")(x)
        x = Dense(self.output_size)(x)
        outputs = x

        # Build full model
        model = keras.Model(inputs=[state_input, goal_input], outputs=outputs)
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-4)
        self.loss_fn = keras.losses.mean_squared_error
        model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        return model

    def epsilon_greedy_policy(self, state, epsilon):
        if np.random.rand() < epsilon:
            return random.choice(self.actions)
        else:
            Q_values = self.model(state[np.newaxis], training=False)
            # print(f"Q_values:{Q_values}")
            # novelty_potential = self.dynamic_reward_shaping.evaluate_sample(np.expand_dims(state, 0))
            action = np.argmax(Q_values)
            return action

    def show_Q(self, state):
        Q_values = self.model(state[np.newaxis], training=False)
        print(f"Q_values:{Q_values} | argmax:{np.argmax(Q_values)} | Action:{ACTIONS[np.argmax(Q_values)]}")

    # def goal_conditioned_eps_greedy(self, state, goal, epsilon):
    #     if np.random.rand() < epsilon:
    #         return random.choice(self.actions)
    #     else:
    #         Q_values = self.model([state[np.newaxis], goal[np.newaxis]], training=False)
    #         # novelty_potential = self.dynamic_reward_shaping.evaluate_sample(np.expand_dims(state, 0))
    #         action = np.argmax(Q_values)
    #         return action

    # def step(self, state, epsilon):
    #     action = self.epsilon_greedy_policy(state, epsilon)
    #     # self.action_list.append(action)
    #     next_state, reward, done, _, _ = self.env.step(action)
    #     self.replay_memory.append([state, action, reward, next_state, done])
    #     return next_state, reward, done

    # def mo_step_imitate(self, state, epsilon, pref, goal_state):
    #     action = self.epsilon_greedy_policy(state, epsilon)
    #     rewards, image, done, next_state = self.env.step(action)
    #     reward = np.dot(pref, rewards)
    #     if next_state == goal_state:
    #         reward = 1
    #     else:
    #         reward = 0
    #     self.replay_memory.append([state, action, reward, next_state, done])
    #     return next_state, reward, done

    def mo_step(self, state, epsilon, pref):
        action = self.epsilon_greedy_policy(state, epsilon)
        # action =self.soft_policy(state)
        # self.action_list.append(action)
        rewards, next_state, done, pos, shaped_reward = self.env.step(action)
        # print(f"next_state shape:{next_state.shape}")
        next_state = np.float32(next_state)
        reward = np.dot(pref, rewards)
        self.replay_memory.append([state, action, reward + shaped_reward, next_state, done])
        return next_state, reward, done, pos

    # def goal_conditioned_mo_step(self, state, goal, epsilon, pref):
    #     action = self.goal_conditioned_eps_greedy(state, goal, epsilon)
    #     self.action_list.append(action)
    #     rewards, image, done, next_state = self.env.step(action)
    #     reward = np.dot(pref, rewards)
    #     self.replay_memory.append([state, goal, action, reward, next_state, done])
    #     return next_state, reward, done

    def training_step(self, episode):
        experiences = self.replay_memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = experiences
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

    # def train_model(self, episodes, save_per=100000, show_detail_per=1000):
    #     """
    #     Train the network over a range of episodes.
    #     """
    #     steps = 0
    #
    #     for episode in range(1, episodes + 1):
    #         self.action_list = []
    #         eps = max(self.eps0 - episode / (episodes * 0.99), EPSILON_END)
    #         # eps = 0.7
    #         # Reset env
    #         state, _ = self.env.reset()
    #         state = np.float32(state)  # Convert to float32 for tf
    #
    #         episode_reward = 0
    #         while True:
    #             state, reward, done = self.step(state, eps)
    #             steps += 1
    #             episode_reward += reward
    #             if done:
    #                 break
    #         if episode > START_TRAINING_AFTER:  # Wait for buffer to fill up a bit
    #             self.training_step(episode)
    #             if episode % COPY_TO_TARGET_EVERY == 0:
    #                 self.target_model.set_weights(self.model.get_weights())
    #             if episode % save_per == 0 and episode >= save_per and self.checkpoint:
    #                 self.model.save(self.model_path + str(episode))
    #         if episode % show_detail_per == 0:
    #             if sum(self.action_list) > 0:
    #                 indices_of_one = [i for i, x in enumerate(self.action_list) if x == 1]
    #             else:
    #                 indices_of_one = "Not run at all"
    #             print(
    #                 f"Epoch:{episode}\tEpoch Reward/Cost:{episode_reward}\tEpsilon:{np.round(eps, 2)}\tActions:{sum(self.action_list)}\t@{indices_of_one}")

    def train_model_with_traj(self, episodes=None, save_per=100000, show_detail_per=1000, traj=None, pref=None,
                              reward_bar=None, reset_to=None):
        """
        Train the network over a range of episodes.
        """
        if episodes == None:
            episodes = TRAINING_EPISODES
        all_reward = []
        for episode in range(1, episodes + 1):
            self.action_list = []
            eps = max(self.eps0 - episode / (episodes * 0.9), EPSILON_END)
            # eps = 0.7
            # Reset env
            agent_traj = []
            state, pos = self.env.reset()

            # print(state.shape)
            # print("-------------------------------------------------------------")
            agent_traj.append(pos)
            # state = np.float32(state)  # Convert to float32 for tf

            episode_reward = 0
            steps = 0

            # for traj_element in traj:
            while True:
                # if episode < 2000:
                #     action = traj[steps]
                # else:
                #     action = None
                # print(f"@pos:{pos}")
                state, reward, done, pos = self.mo_step(state, eps, pref=pref)
                # state = np.float32(pos)
                agent_traj.append(pos)
                steps += 1
                episode_reward += reward
                if done:
                    all_reward.append(episode_reward)
                    if len(all_reward) > 100:
                        all_reward[-1] = np.mean(all_reward[-100:])
                    break

            if episode > START_TRAINING_AFTER:  # Wait for buffer to fill up a bit
                self.training_step(episode)
                if episode % COPY_TO_TARGET_EVERY == 0:
                    self.target_model.set_weights(self.model.get_weights())
                # if episode % save_per == 0 and episode >= save_per:
            # if episode % (episodes // 6) == 0 and episode < episodes*0.9:
            #     print(f"@Episode:{episode}, reset weights to avoid overfitting!")
            #     self.model = self.build_plain_model()
            #     self.target_model = self.build_plain_model()
            #     # Make weights the same
            #     self.target_model.set_weights(self.model.get_weights())
            if episode % 100 == 0:
                print(
                    f"Episode:{episode} | AVG:{all_reward[-1] * 124} | epsilon:{eps} | episode_reward:{episode_reward * 124} | agent_traj:{agent_traj}")
        self.model.save(self.model_path + str(pref))

    def generate_experience(self, pref=None):
        state_list = []
        # Reset env
        state, pos = self.env.reset()
        # state = np.float32(state)  # Convert to float32 for tf
        state_list.append(pos)
        episode_reward = 0

        while True:
            state, reward, done, pos = self.mo_step(state, 0.05, pref=pref)
            state_list.append(pos)
            episode_reward += reward
            if done:
                break
        print(f"pref:{pref} | episode_reward:{episode_reward*124} | traj:{state_list}")
        return episode_reward
