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
        self.output_size = self.env.action_space

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
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        self.loss_fn = keras.losses.mean_squared_error
        model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        return model

    def epsilon_greedy_policy(self, state, weights, epsilon, evaluation=False):
        if np.random.rand() < epsilon:
            return random.choice(self.actions)
        else:
            Q_values = self.model([state[np.newaxis], weights[np.newaxis]], training=False)
            # print(f"Q_values:{np.dot(Q_values, weights)}")
            action = np.argmax(Q_values)
            if evaluation:
                print(f"state:{np.round_(state, 3)}\tQ:{np.round_(Q_values, 3)}\taction:{action}")
            return action

    def _act(self, state, weights, epsilon):
        action = self.epsilon_greedy_policy(state, weights, epsilon, evaluation=False)
        return action

    def update(self, tensor_w):
        experiences = self.replay_memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones, weightss = experiences

        # Compute target Q values from 'next_states'
        next_Q_values = self.target_model([next_states, weightss], training=False)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards + (1 - dones) * self.gamma * max_next_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1)  # Make column vector

        # Mask to only consider action taken
        mask = tf.one_hot(actions, self.output_size)  # Number of actions
        # Compute loss and gradient for predictions on 'states'
        with tf.GradientTape() as tape:
            # print(f"states:{states}\nweightss:{weightss}")
            all_Q_values = self.model([states, weightss])
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            q_loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        # print(f"q_loss:{q_loss}")
        grads = tape.gradient(q_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

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
        iterations = 0
        while EU < EU_target * 0.95:
            iterations += 1
            self.jsmorl_train_iteration(eval_env=eval_env, eval_freq=100, corners=corners, demos=demos)
            utilities = []
            for w in corners:
                utility, disc_vec_return, _, _ = self.play_a_episode(env=eval_env, agent=self, demo=[], weights=w)
                utilities.append(utility)
            EU = np.mean(utilities)
            print(f"@iteration{iterations}-- EU:{EU}\tEU_target:{EU_target}")
        print(f"reach!!!!@step:{self.global_step}")
        self.model.save(filepath="abc_model")
        corners = sorted(corners, key=lambda x: x[0])
        for w in corners:
            disc_return, disc_vec_return, scalar_return, vec_return = self.play_a_episode(env=eval_env, weights=w,
                                                                                          agent=self, demo=[])
            print(f"for w:{np.round_(w,3)}\tdisc vec return:{vec_return}")

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
            reward = np.dot(vec_reward, w)
            next_obs = np.array(next_obs)

            self.replay_memory.append([obs, action, reward, next_obs, terminated, w])

            if self.global_step >= self.start_training_after:
                self.update(tf.convert_to_tensor(w))
                if self.global_step % self.copy_to_target_per == 0:
                    self.target_model.set_weights(self.model.get_weights())

            if self.global_step % eval_freq == 0:
                u, disc_vec_return,_,_ = self.play_a_episode(env=eval_env, weights=w, agent=self, demo=demo[:pi_g_horizon])
                if u >= utility_threshold:
                    print(f"@:{w} -- reach threshold")
                    pi_g_horizon -= 2
                    pi_g_horizon = max(pi_g_horizon, 0)

                print(f"guide policy horizon:{pi_g_horizon}\t demo:{demo}\tw_c:{w}")

            if terminated or truncated:
                obs, _ = self.env.reset()
                obs = np.array(obs)
            else:
                obs = next_obs

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

    def play_a_episode(self, env, weights, agent, demo, evaluation=False):
        disc_vec_return = np.zeros_like(weights, dtype=np.float64)
        vec_return = np.zeros_like(weights, dtype=np.float64)
        disc_return = 0
        scalar_return = 0
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
                action = agent.epsilon_greedy_policy(state, weights=weights, epsilon=0, evaluation=evaluation)
            traj_actions.append(action)
            n_state, rewards, terminal, _, _ = env.step(action)
            n_state = np.array(n_state)
            traj_states.append(n_state)
            disc_vec_return += gamma * rewards
            vec_return += rewards
            disc_return += gamma * np.dot(rewards, weights)
            scalar_return = np.dot(rewards, weights)
            gamma *= self.gamma
            state = n_state
            if steps > 100:
                break
        # print(f"eval action traj:{traj_actions}")
        return disc_return, disc_vec_return, scalar_return, vec_return


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
