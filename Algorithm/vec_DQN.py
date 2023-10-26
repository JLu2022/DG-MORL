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
from simulators.minecart.minecart_simulator import Minecart
from Algorithm.common.weights import equally_spaced_weights


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
                 replay_mem_size=6000, checkpoint=True, reward_dim=2, actions=4, state_dim=2, learning_rate=1e-3,
                 epsilon=0.5):
        self.gamma = gamma
        self.global_step = 0
        self.env = env
        self.actions = [i for i in range(actions)]
        self.copy_to_target_per = copy_to_target_per
        self.start_training_after = start_training_after
        self.batch_size = batch_size
        self.replay_memory = ReplayMemory(maxlen=replay_mem_size)
        self.checkpoint = checkpoint

        self.input_size = state_dim
        self.weight_size = reward_dim
        self.output_size = actions
        self.epsilon = epsilon

        self.learning_rate = learning_rate
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.model_path = model_path
        self.EU_list = []

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
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_fn = keras.losses.mean_squared_error
        model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        return model

    def epsilon_greedy_policy(self, state, weights, epsilon, evaluation=False):
        if np.random.rand() < epsilon:
            return random.choice(self.actions)
        else:
            Q_values = self.model([state[np.newaxis], weights[np.newaxis]], training=False)
            action = np.argmax(Q_values)
            if evaluation:
                print(f"state:{np.round_(state, 3)}\tQ:{np.round_(Q_values, 3)}\taction:{action}")
            return action

    def _act(self, state, weights, epsilon):
        action = self.epsilon_greedy_policy(state, weights, epsilon, evaluation=False)
        return action

    def update(self):
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
            all_Q_values = self.model([states, weightss])
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            q_loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(q_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return q_loss

    def get_corners(self, demos, eval_env):
        linear_support = LinearSupport(num_objectives=self.weight_size,
                                       epsilon=0.0)
        new_ccs = []
        for demo in demos:
            utility, vec_return, _, _ = self.evaluate_demo(demo=demo, eval_env=eval_env,
                                                           weights=np.zeros(self.weight_size))
            new_ccs.append(vec_return)

        if not len(linear_support.ccs) == len(new_ccs):  # when empty
            linear_support.ccs = new_ccs
        else:
            updated_list = [new_ccs[i] if linear_support.ccs[i] != new_ccs[i] else linear_support.ccs[i] for i in
                            range(len(linear_support.ccs))]
            linear_support.ccs = updated_list
        corners = linear_support.compute_corner_weights()
        return corners

    def get_demos_EU(self, corners, demos, eval_env):
        utility_thresholds = []
        for w_c in corners:
            demo, utility_threshold, _, _ = self.weight_to_demo(w=w_c, demos=demos, eval_env=eval_env)
            utility_thresholds.append(utility_threshold)
        utility_thresholds = np.array(utility_thresholds)
        EU_target = np.mean(utility_thresholds)
        return EU_target

    def get_agent_EU(self, corners, eval_env):
        utilities = []
        for w in corners:
            utility, disc_vec_return, _, _, _ = self.play_a_episode(env=eval_env, agent=self, demo=[], weights=w)
            utilities.append(utility)
        EU = np.mean(utilities)
        return EU

    def evaluate_agent(self, eval_env, eval_weights, show_case=True):
        u = 0

        for weight in eval_weights:
            disc_return, disc_vec_return, scalar_return, vec_return, _ = self.play_a_episode(env=eval_env,
                                                                                             weights=weight,
                                                                                             agent=self, demo=[])
            u += disc_return
            if show_case:
                print(f"for w:{np.round_(weight, 3)}\tdisc vec return:{vec_return}")
        if show_case:
            print(f"100 EU:{u / 100}")
        return u

    def jsmorl_train(self, demos, eval_env, total_timesteps, timesteps_per_iter):
        self.demo_visits = np.zeros(len(demos) + 1)
        corners = self.get_corners(demos=demos, eval_env=eval_env)
        print(f"corner weights:{np.round_(corners, 3)}")
        eval_weights = equally_spaced_weights(self.weight_size, n=100)
        EU_target = self.get_demos_EU(corners=eval_weights, demos=demos, eval_env=eval_env)
        EU = -np.inf
        iterations = 0
        step_list = []
        while EU < EU_target and self.global_step < total_timesteps:
            iterations += 1
            demos = self.jsmorl_train_iteration(eval_env=eval_env, eval_freq=100, corners=corners,
                                                demos=demos,
                                                total_timesteps=timesteps_per_iter, roll_back_step=2)

            corners = self.get_corners(demos=demos, eval_env=eval_env)
            EU = self.get_agent_EU(corners=eval_weights, eval_env=eval_env)
            EU_target = self.get_demos_EU(corners=eval_weights, demos=demos, eval_env=eval_env)

            self.EU_list.append(EU)
            step_list.append(self.global_step)
            print(f"@iteration{iterations}-- EU:{EU}\tEU_target:{EU_target}")
        print(f"reach!!!!@step:{self.global_step}")
        self.model.save(filepath="abc_model")

        corners = self.get_corners(demos=demos, eval_env=eval_env)
        print(f"corner weights:{np.round_(corners, 3)}")
        eval_weights = equally_spaced_weights(self.weight_size, n=100)
        self.evaluate_agent(eval_env=eval_env, eval_weights=eval_weights)
        plt.plot(step_list, self.EU_list)
        plt.show()

    def jsmorl_train_iteration(self,
                               eval_env=None,
                               eval_freq: int = 100,
                               corners=None,
                               demos=None,
                               total_timesteps=4000,
                               roll_back_step=2
                               ):
        """Train the agent for one iteration.
                Args:
                    eval_env (Optional[gym.Env]): Environment to evaluate on
                    eval_freq (int): Number of timesteps between evaluations
        """

        # idx = np.random.randint(0, len(corners))
        # w = corners[idx]
        priorities = []
        demos_ = []
        u_thresholds = []
        for w in corners:
            demo, utility_threshold, max_vec_return, demo_idx = self.weight_to_demo(w, demos, eval_env=eval_env)
            u = self.evaluate_agent(eval_env=eval_env, eval_weights=np.array([w]), show_case=False)
            priority = abs((utility_threshold - u) / utility_threshold)
            priorities.append(priority)
            demos_.append(demo)
            u_thresholds.append(utility_threshold)

        idx = random.choices(range(len(priorities)), weights=priorities, k=1)[0]
        w = corners[idx]
        demo = demos_[idx]

        utility_threshold = u_thresholds[idx]
        print(f"weight:{w} is sampled by priority:{priorities[idx]}")
        pi_g_pointer = 0
        pi_g_horizon = len(demo) - 1

        obs, _ = self.env.reset()
        obs = np.array(obs)
        step = 0
        epsilon_factor = 1
        self.epsilon = 0.9
        while step < total_timesteps and pi_g_horizon >= 0:
        # while pi_g_horizon >= 0:
            step += 1
            self.global_step += 1

            if pi_g_horizon > 0 and pi_g_pointer < pi_g_horizon:
                action = demo[:pi_g_horizon][pi_g_pointer]
                pi_g_pointer += 1
            else:
                action = self._act(state=obs, weights=w, epsilon=self.epsilon)
            self.epsilon -= self.epsilon/1000
            next_obs, vec_reward, terminated, truncated, info = self.env.step(action)
            reward = np.dot(vec_reward, w)
            next_obs = np.array(next_obs)

            self.replay_memory.append([obs, action, reward, next_obs, terminated, w])

            if self.global_step >= self.start_training_after:
                self.update()
                if self.global_step % self.copy_to_target_per == 0:
                    if True:
                        self.target_model.set_weights(self.model.get_weights())

            if self.global_step % eval_freq == 0:
                u, disc_vec_return, _, _, new_demo = self.play_a_episode(env=eval_env, weights=w, agent=self,
                                                                         demo=demo[:pi_g_horizon], evaluation=False)
                if u >= utility_threshold:
                    self.epsilon = 0.9
                    print(
                        f"@:{w} -- reach threshold - u:{np.round_(u, 4)}>u_threshold:{np.round_(utility_threshold, 4)}"
                        f"\nreplace old demo {demo} with better demo:{new_demo}"
                        f"\tepsilon factor:{epsilon_factor}")
                    demo = new_demo
                    demos_[idx] = new_demo
                    if pi_g_horizon >= 1:
                        pi_g_horizon = max(pi_g_horizon - roll_back_step, 0)
                    else:
                        pi_g_horizon -= roll_back_step

                print(
                    f"guide policy horizon:{pi_g_horizon}\t"
                    f"demo:{demo}\t"
                    f"w_c:{w}\t"
                    f"u:{np.round_(u, 4)}\t"
                    f"u_threshold:{np.round_(utility_threshold, 4)}")

            if terminated or truncated:
                obs, _ = self.env.reset()
                obs = np.array(obs)
            else:
                obs = next_obs
        # self.demo_visits[idx] += 1
        for d in demos_:
            print(f"demo:{d}")
        return demos_

    def weight_to_demo(self, w, demos, eval_env):
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
        return max_demo, max_utility, max_vec_return, max_demo_idx

    def evaluate_demo(self, demo, eval_env, weights=np.zeros([1, 0])):
        disc_vec_return = np.zeros(self.weight_size, dtype=np.float64)
        vec_return = np.zeros(self.weight_size, dtype=np.float64)
        disc_scalar_return = 0
        scalar_return = 0
        gamma = 1
        obs, _ = eval_env.reset()
        for action in demo:
            next_obs, rewards, terminated, _, _ = eval_env.step(action)
            disc_scalar_return += gamma * np.dot(rewards, weights)
            disc_vec_return += gamma * rewards

            scalar_return += np.dot(rewards, weights)
            vec_return += rewards
            gamma *= self.gamma
            obs = next_obs
        return disc_scalar_return, disc_vec_return, scalar_return, vec_return

    def play_a_episode(self, env, weights, agent, demo, evaluation=False):
        disc_vec_return = np.zeros(self.weight_size, dtype=np.float64)
        vec_return = np.zeros(self.weight_size, dtype=np.float64)
        disc_return = 0
        scalar_return = 0
        gamma = 1
        terminal = False
        state, _ = env.reset()
        state = np.array(state)
        action_traj = []
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
            action_traj.append(action)
            n_state, rewards, terminal, _, _ = env.step(action)
            n_state = np.array(n_state)
            traj_states.append(n_state)
            disc_vec_return += gamma * rewards
            vec_return += rewards
            disc_return += gamma * np.dot(rewards, weights)
            scalar_return = np.dot(rewards, weights)
            gamma *= self.gamma
            state = n_state
            if steps > 50:
                break
        if evaluation:
            print(f"eval action traj:{action_traj}")

        return disc_return, disc_vec_return, scalar_return, vec_return, action_traj


if __name__ == '__main__':
    linear_support = LinearSupport(num_objectives=2,
                                   epsilon=0.0)
    """ This part is for Experiment of DST"""
    deep_sea_treasure = DeepSeaTreasure()
    eval_env = DeepSeaTreasure()
    agent = ConditionedDQNAgent(env=deep_sea_treasure, learning_rate=1e-3, replay_mem_size=20000,
                                copy_to_target_per=200, batch_size=128, epsilon=0.5)

    action_demo_1 = [2, 1]  # 0.7
    action_demo_2 = [2, 3, 1, 1]  # 8.2
    action_demo_3 = [2, 3, 3, 1, 1, 1]  # 11.5
    action_demo_4 = [2, 3, 3, 3, 1, 1, 1, 1]  # 14.0
    action_demo_5 = [2, 3, 3, 3, 3, 1, 1, 1, 1]  # 15.1
    action_demo_6 = [2, 3, 3, 3, 3, 3, 1, 1, 1, 1]  # 16.1
    action_demo_7 = [2, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1]  # 19.6
    action_demo_8 = [2, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1]  # 20.3
    action_demo_9 = [2, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 22.4
    action_demo_10 = [2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 23.7
    action_demos = [action_demo_1, action_demo_2, action_demo_3, action_demo_4, action_demo_5, action_demo_6,
                    action_demo_7, action_demo_8, action_demo_9, action_demo_10]
    agent.jsmorl_train(demos=action_demos, eval_env=eval_env, total_timesteps=40000, timesteps_per_iter=10000)

    # max_demo, max_utility, max_vec_return, max_demo_idx = agent.weight_to_demo(np.array([0, 1e0]), action_demos, eval_env)
    # print(f"max demo:{max_demo}")
    """ This part is for Experiment of Minecart"""
    # mine_cart_env = Minecart()
    # eval_env = Minecart()
    # agent = ConditionedDQNAgent(env=mine_cart_env, gamma=0.98, reward_dim=3, actions=6, state_dim=7,
    #                             replay_mem_size=20000, copy_to_target_per=2000)
    # human_demos = np.load("../train/minecart/traj/demos.npy", allow_pickle=True)
    # print(f"human demos:{human_demos}")
    # # human_demo = dict(human_demo)
    # # action_demos = []
    # # for k, v in human_demo.items():
    # #     print(f"w:{k}\tv:{v[0]}\tmode:{v[1]}\tactions:{v[2]}")
    # #     action_demos.append(v[2])
    # agent.jsmorl_train(demos=human_demos, eval_env=eval_env)
