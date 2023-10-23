import copy
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate, Dense, LayerNormalization
from keras import metrics
from keras.optimizers import Adam
from collections import deque  # Used for replay buffer and reward tracking
import matplotlib.pyplot as plt
from datetime import datetime  # Used for timing script
from Algorithm.common.morl_algorithm import MOAgent, MOPolicy
from Algorithm.linear_support import LinearSupport
from simulators.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure


# from simulators.minecart.minecart_simulator import Minecart

"""GPI-PD algorithm."""
import os
import random
from itertools import chain
from typing import Callable, List, Optional, Union

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

from common.buffer import ReplayBuffer
from common.evaluation import (
    log_all_multi_policy_metrics,
    log_episode_info,
    policy_evaluation_mo,
)
from common.model_based.probabilistic_ensemble import (
    ProbabilisticEnsemble,
)
from common.model_based.utils import ModelEnv, visualize_eval
from common.morl_algorithm import MOAgent, MOPolicy
from common.networks import (
    NatureCNN,
    get_grad_norm,
    huber,
    layer_init,
    mlp,
    polyak_update,
)
from common.prioritized_buffer import PrioritizedReplayBuffer
from common.utils import linearly_decaying_value, unique_tol
from common.weights import equally_spaced_weights
from linear_support import LinearSupport


class QNet(nn.Module):
    """Conditioned MO Q network."""

    def __init__(self, obs_shape, action_dim, rew_dim, net_arch, drop_rate=0.01, layer_norm=True):
        """Initialize the net.

        Args:
            obs_shape: The observation shape.
            action_dim: The action dimension.
            rew_dim: The reward dimension.
            net_arch: The network architecture.
            drop_rate: The dropout rate.
            layer_norm: Whether to use layer normalization.
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.phi_dim = rew_dim

        self.weights_features = mlp(rew_dim, -1, net_arch[:1])
        if len(obs_shape) == 1:
            self.state_features = mlp(obs_shape[0], -1, net_arch[:1])
        elif len(obs_shape) > 1:  # Image observation
            self.state_features = NatureCNN(self.obs_shape, features_dim=net_arch[0])
        self.net = mlp(
            net_arch[0], action_dim * rew_dim, net_arch[1:], drop_rate=drop_rate, layer_norm=layer_norm
        )  # 128/128 256 256 256

        self.apply(layer_init)

    def forward(self, obs, w):
        """Forward pass."""
        sf = self.state_features(obs)
        wf = self.weights_features(w)
        q_values = self.net(sf * wf)
        return q_values.view(-1, self.action_dim, self.phi_dim)  # Batch size X Actions X Rewards

class PreferenceSpace:
    def __init__(self, fixed_w=None):
        self.fixed_w = fixed_w

    def sample(self):
        # Each preference weight is randomly sampled between -20 and 20 in steps of 5
        p0 = random.choice([x for x in range(0, 101)])  # time
        p1 = 100 - p0
        if self.fixed_w is None:
            preference = np.array([p0, p1], dtype=np.float32) / 100
        else:
            preference = self.fixed_w
        return preference


class ReplayMemory(deque):
    def sample(self, batch_size):
        indices = np.random.randint(len(self), size=batch_size)
        batch = [self[index] for index in indices]
        states, actions, rewards, next_states, dones, weightss = [
            np.array([experience[field_index] for experience in batch]) for field_index in range(6)]
        return states, actions, rewards, next_states, dones, weightss


class ConditionedDQNAgent:
    def __init__(self, env, batch_size=64, copy_to_target_per=200, start_training_after=50, gamma=0.99, model_path=None,
                 replay_mem_size=6000, checkpoint=True, reward_dim=2):
        self.gamma = gamma
        self.global_step = 0
        self.env = env
        self.actions = [i for i in range(self.env.action_space)]
        self.copy_to_target_per = copy_to_target_per
        self.start_training_after = start_training_after
        self.batch_size = batch_size
        self.replay_memory = ReplayMemory(maxlen=replay_mem_size)
        self.checkpoint = checkpoint

        self.input_size = self.env.observation_space
        self.weight_size = reward_dim
        self.output_size = self.env.action_space * reward_dim

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.model_path = model_path

    def build_model(self):
        # Define Layers
        weight_input = Input(shape=(self.weight_size,))
        state_input = Input(shape=self.input_size)
        x = Concatenate()([state_input, weight_input])
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2, name="drop_0")(x)
        # x = LayerNormalization(axis=-1)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2, name="drop_1")(x)
        # x = LayerNormalization(axis=-1)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2, name="drop_2")(x)
        # x = LayerNormalization(axis=-1)(x)
        x = Dense(self.output_size)(x)
        outputs = x

        # Build full model
        model = keras.Model(inputs=[state_input, weight_input], outputs=outputs)
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        loss_fn = keras.losses.mean_squared_error
        model.compile(optimizer=optimizer, loss=loss_fn)
        return model

    def epsilon_greedy_policy(self, state, weights, epsilon, evaluation=False):
        if np.random.rand() < epsilon:
            return random.choice(self.actions)
        else:
            Q_values = self.model([state[np.newaxis], weights[np.newaxis]], training=False)
            Q_values = tf.reshape(Q_values, (-1, 2))
            # print(f"Q_values:{np.dot(Q_values, weights)}")
            action = np.argmax(np.dot(Q_values, weights))
            if evaluation:
                print(f"state:{np.round_(state, 3)}\tQ:{np.round_(Q_values, 3)}\taction:{action}")
            return action

    def _act(self, state, weights, epsilon):
        action = self.epsilon_greedy_policy(state, weights, epsilon, evaluation=False)
        return action

    def update(self, tensor_w):
        tensor_w = tf.reshape(tensor_w,(1,2,1))
        experiences = self.replay_memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones, weightss = experiences

        # Compute target Q values from 'next_states'
        next_Q_values = self.target_model([next_states, weightss], training=False)
        print(f"next_Q:{next_Q_values}")
        next_Q_values = tf.reshape(next_Q_values, (-1, 2, 4))
        print(f"next_Q:{next_Q_values}")
        next_Q_values = tf.reduce_sum(next_Q_values, tensor_w)
        print(f"next_Q:{next_Q_values}")
        max_next_Q_values = np.max(next_Q_values, axis=1)
        print(f"rewards:{rewards}\tmax_Q:{max_next_Q_values}")
        target_Q_values = (rewards + (1 - dones) * self.gamma * max_next_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1)  # Make column vector

        # Mask to only consider action taken
        mask = tf.one_hot(actions, self.output_size)  # Number of actions
        # Compute loss and gradient for predictions on 'states'
        with tf.GradientTape() as tape:
            # print(f"states:{states}\nweightss:{weightss}")
            all_Q_values = self.model([states, weightss])
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            q_loss = tf.reduce_mean(self.model.loss_fn(target_Q_values, Q_values))
        # print(f"q_loss:{q_loss}")
        grads = tape.gradient(q_loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def evaluate_demo(self, demo, eval_env, weights=np.array([1, 0])):
        disc_vec_return = np.zeros_like(weights, dtype=np.float64)
        vec_return = np.zeros_like(weights, dtype=np.float64)
        disc_scalar_return = 0
        scalar_return = 0
        gamma = 1
        eval_env.reset()
        for action in demo:
            _, rewards, _, _, _ = eval_env.step(action)
            vec_return += rewards
            scalar_return += np.dot(rewards, weights)
            disc_vec_return += gamma * rewards
            disc_scalar_return += gamma * np.dot(rewards, weights)
            gamma *= self.gamma
        return disc_scalar_return, disc_vec_return, scalar_return, vec_return

    def jsmorl_train(self, demos, eval_env):
        linear_support = LinearSupport(num_objectives=self.weight_size,
                                       epsilon=0.0)
        for demo in demos:
            utility, vec_return, _, _ = self.evaluate_demo(demo=demo, eval_env=eval_env)
            linear_support.ccs.append(vec_return)
        corners = linear_support.compute_corner_weights()
        utility_thresholds = []
        for w_c in corners:
            demo, utility_threshold, _ = self.weight_to_demo(w_c, demos)
            utility_thresholds.append(utility_threshold)
        utility_thresholds = np.array(utility_thresholds)

        EU_target = np.mean(utility_thresholds)
        EU = -np.inf
        while EU < EU_target:
            self.jsmorl_train_iteration(eval_env=eval_env, eval_freq=100, corners=corners, demos=demos)
            utilities = []
            for w in corners:
                utility, disc_vec_return = self.play_a_episode(env=eval_env, agent=self, demo=[], weights=w)
                utilities.append(utility)
            EU = np.mean(utilities)

    def jsmorl_train_iteration(self,
                               eval_env=None,
                               eval_freq: int = 100,
                               corners=None,
                               demos=None,
                               ):
        """Train the agent for one iteration.
                Args:
                    eval_env (Optional[gym.Env]): Environment to evaluate on
                    eval_freq (int): Number of timesteps between evaluations
        """

        idx = np.random.randint(0, len(corners))
        w = corners[idx]
        demo, utility_threshold, _ = self.weight_to_demo(w, demos)
        pi_g_pointer = 0
        pi_g_horizon = len(demo) - 1

        obs, _ = self.env.reset()
        obs = np.array(obs)
        step = 0
        while step < 3000 and pi_g_horizon > 0:
            step += 1
            self.global_step += 1
            if pi_g_horizon > 0 and pi_g_pointer < pi_g_horizon:
                action = demo[:pi_g_horizon][pi_g_pointer]
                pi_g_pointer += 1
            else:
                action = self._act(state=obs, weights=w, epsilon=0.5)

            next_obs, vec_reward, terminated, truncated, info = self.env.step(action)
            next_obs = np.array(next_obs)

            self.replay_memory.append([obs, action, vec_reward, next_obs, terminated, w])

            if self.global_step >= self.start_training_after:
                self.update(tf.convert_to_tensor(w))
                if self.global_step % self.copy_to_target_per == 0:
                    self.target_model.set_weights(self.model.get_weights())

            if self.global_step % eval_freq == 0:
                u, disc_vec_return = self.play_a_episode(env=eval_env, weights=w, agent=self, demo=demo[:pi_g_horizon])
                if u >= utility_threshold:
                    print(f"@:{w} -- reach threshold")
                    pi_g_horizon -= 2
                    pi_g_horizon = max(pi_g_horizon, 0)

                print(f"guide policy horizon:{pi_g_horizon}\t demo:{demos}\tw_c:{corners}")

            if terminated or truncated:
                obs, _ = self.env.reset()
                obs = np.array(obs)
            else:
                obs = next_obs

    # def train_model(self, steps, save_per=100000, show_detail_per=1000, pref_space=PreferenceSpace(),
    #                 corner_weights=None, eval_env=None):
    #     """
    #     Train the network over a range of episodes.
    #     """
    #
    #     for step in range(1, steps + 1):
    #         self.action_list = []
    #         eps = 0.5
    #         # Reset env
    #         state, _ = self.env.reset()
    #         state = np.array(state)  # Convert to float32 for tf
    #
    #         episode_reward = 0
    #         rewards_vec = np.zeros(3)
    #         # weights = pref_space.sample()
    #         weights = np.array([0.34, 0.36, 0.3])
    #         while True:
    #             state, reward, done, rewards = self.play_one_step(np.array(state), eps, weights)
    #             steps += 1
    #             episode_reward += reward
    #             rewards_vec += rewards
    #             if done:
    #                 break
    #         if step > self.start_training_after:  # Wait for buffer to fill up a bit
    #             self.training_step()
    #             if step % self.copy_to_target_per == 0:
    #                 self.target_model.set_weights(self.model.get_weights())
    #             if step % save_per == 0 and step >= save_per and self.checkpoint:
    #                 self.model.save(self.model_path + str(step))
    #             u = self.play_a_episode(env=eval_env, weights=np.array([0, 1]), agent=self, demo=[])
    #             print(f"tr"
    #                   f"yout_u:{u}")
    #         if step % show_detail_per == 0:
    #             if sum(self.action_list) > 0:
    #                 indices_of_one = [i for i, x in enumerate(self.action_list) if x == 1]
    #             else:
    #                 indices_of_one = "Not run at all"
    #             print(f"Epoch:{step}\t"
    #                   f"Pref:{weights}"
    #                   f"Epoch Reward/Cost:{episode_reward}\t"
    #                   f"Reward Vec:{rewards_vec}\t"
    #                   f"Epsilon:{np.round(eps, 2)}\t"
    #                   f"Actions:{sum(self.action_list)}\t"
    #                   f"@{indices_of_one}")
    #     # u = self.play_a_episode(env=eval_env, pref_w=np.array([0., 1.]), agent=self, demo=[])
    #     # print(f"utility:{u}")

    def weight_to_demo(self, w, demos):
        disc_scalar_returns = []
        vec_returns = []
        for demo in demos:
            disc_scalar_return, disc_vec_return, scalar_return, vec_return = self.evaluate_demo(demo=demo,
                                                                                                eval_env=eval_env,
                                                                                                weights=w)
            disc_scalar_returns.append(disc_scalar_return)
            vec_returns.append(vec_return)
        max_demo_idx = np.argmax(disc_scalar_returns)
        max_demo = demos[max_demo_idx]
        max_utility = disc_scalar_returns[max_demo_idx]
        max_vec_return = vec_returns[max_demo_idx]
        return max_demo, max_utility, max_vec_return

    def play_a_episode(self, env, weights, agent, demo):
        disc_vec_return = np.zeros_like(weights, dtype=np.float64)
        disc_return = 0
        gamma = 1
        terminal = False
        state, _ = env.reset()
        state = np.array(state)
        traj_actions = []
        traj_states = [state]
        action_pointer = 0
        steps = 0
        while not terminal:
            steps += 1
            if action_pointer < len(demo):
                action = demo[action_pointer]
                action_pointer += 1
            else:
                action = agent.epsilon_greedy_policy(state, weights=weights, epsilon=0, evaluation=True)
            traj_actions.append(action)
            n_state, rewards, terminal, _, _ = env.step(action)
            n_state = np.array(n_state)
            traj_states.append(n_state)
            disc_vec_return += gamma * rewards
            disc_return += gamma * np.dot(rewards, weights)
            gamma *= self.gamma
            state = n_state
            if steps > 100:
                break
        print(f"eval action traj:{traj_actions}")
        return disc_return, disc_vec_return


if __name__ == '__main__':
    linear_support = LinearSupport(num_objectives=2,
                                   epsilon=0.0)
    deep_sea_treasure = DeepSeaTreasure()
    eval_env = DeepSeaTreasure()
    agent = ConditionedDQNAgent(env=deep_sea_treasure)
    # corners = np.array([[0., 1.]])
    # demos = [[3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    # u_thresholds = [[19.77]]
    # agent.jsRL_train(total_timesteps=4000,
    #                  eval_env=eval_env,
    #                  eval_freq=100,
    #                  corners=corners,
    #                  demos=demos,
    #                  u_thresholds=u_thresholds)
    # agent.train_model(steps=12000, eval_env=eval_env)

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
    agent.jsmorl_train(demos=action_demos, eval_env=eval_env)
    # weight = np.array([0.67, 0.33])
    # max_demo, max_utility, max_vec_return = agent.weight_to_demo(w=weight, demos=action_demos)
    # print(f"for w:{weight}\tmax_demo:{max_demo}\tmax_u:{max_utility}\tmax_vec_return:{max_vec_return}")
    # for action_demo in action_demos:
    #     _, disc_vec_return = agent.play_a_episode(env=deep_sea_treasure, agent=agent, demo=action_demo,
    #                                               weights=np.array([0, 1]))
    #     print(f"disc vec return:{disc_vec_return}")
    # corner_ws, rews_demo_dict, _ = linear_support.get_support_weight_from_demo(demos=action_demos,
    #                                                                            env=eval_env)
