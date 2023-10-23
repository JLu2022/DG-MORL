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
# from Algorithm.linear_support import LinearSupport
from simulators.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure
from simulators.minecart.minecart_simulator import Minecart

SEED = 42
DEBUG = False

BATCH_SIZE = 128
REPLAY_MEMORY_SIZE = 6000

GAMMA = 0.98

TRAINING_EPISODES = 5000
EXPLORATION_RESTARTS = 0

EPSILON_START = 1
EPSILON_END = 0.01
EPSILON_DECAY = 1 / (TRAINING_EPISODES * 0.98)
COPY_TO_TARGET_EVERY = 200  # Steps
START_TRAINING_AFTER = 100  # Episodes
FRAME_STACK_SIZE = 3
NUM_WEIGHTS = 2


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
    def __init__(self, env, model_path=None, checkpoint=True):
        self.global_step = 0
        self.dynamic_reward_shaping_optimizer = Adam(learning_rate=1e-4)
        self.env = env
        # self.actions = [i for i in range(self.env.action_space)]
        self.actions = range(5)
        self.gamma = GAMMA  # Discount
        self.eps0 = EPSILON_START  # Epsilon greedy init
        self.model_path = model_path
        self.batch_size = BATCH_SIZE
        self.replay_memory = ReplayMemory(maxlen=REPLAY_MEMORY_SIZE)
        self.checkpoint = checkpoint
        # self.input_size = self.env.observation_space
        self.input_size = 7
        # self.output_size = self.env.action_space
        self.output_size = 5
        # Build both models
        self.model = self.build_model()
        self.target_model = self.build_model()
        # Make weights the same
        self.target_model.set_weights(self.model.get_weights())

    def build_model(self):
        # Define Layers
        weight_input = Input(shape=(3,))
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
            # print(state)
            # print(weights)
            Q_values = self.model([state[np.newaxis], weights[np.newaxis]], training=False)

            action = np.argmax(Q_values)
            if evaluation:
                print(f"state:{np.round_(state,3)}\tQ:{np.round_(Q_values,3)}\taction:{action}")
            return action

    def play_one_step(self, state, epsilon, weights):
        action = self.epsilon_greedy_policy(state, weights, epsilon)
        self.action_list.append(action)
        next_state, rewards, done, _, _ = self.env.step(action)
        next_state = np.array(next_state)
        reward = np.dot(rewards, weights)
        self.replay_memory.append([state, action, reward, next_state, done, weights])
        return next_state, reward, done, rewards

    def _act(self, state, weights, epsilon):
        action = self.epsilon_greedy_policy(state, weights, epsilon, evaluation=False)
        return action

    def training_step(self):
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

    def jsRL_train(self,
                   total_timesteps: int,
                   eval_env=None,
                   eval_freq: int = 100,
                   corners=None,
                   demos=None,
                   u_thresholds=None):
        """Train the agent for one iteration.

                Args:
                    total_timesteps (int): Number of timesteps to train for
                    weight (np.ndarray): Weight vector
                    weight_support (List[np.ndarray]): Weight support set
                    change_w_every_episode (bool): Whether to change the weight vector at the end of each episode
                    reset_num_timesteps (bool): Whether to reset the number of timesteps
                    eval_env (Optional[gym.Env]): Environment to evaluate on
                    eval_freq (int): Number of timesteps between evaluations
                    reset_learning_starts (bool): Whether to reset the learning starts
                """

        self.police_indices = []

        """
        1. For each guide policy, grab the experience
        2. Do and further explore
        3. Do iterative evaluation, if any explore policy meet the criteria, roll-out the guide policy, if any explore 
        policy decay, roll-back the guide policy, do this until all guide policy goes to empty.
        4. Save the model as a good pre-train start point.
        5*. further train with gpi.
        """
        guide_policy_scopes = []
        for demo in demos:
            guide_policy_scopes.append(len(demo) - 1)
        guide_policy_scopes = np.array(guide_policy_scopes)
        max_guide_policy_scopes = copy.deepcopy(guide_policy_scopes)
        try_out = 0
        while sum(guide_policy_scopes) > 0:  # change to random sample a tuple!!
            try_out += 1
            print("------------------------------------------------------------------")
            idx = np.random.randint(0, len(corners))
            print(f"try_out:{try_out}.. start\t "
                  f"guide_scopes:{guide_policy_scopes}\t"
                  f"idx:{idx}\n"
                  f"demo:{demos}")

            w = corners[idx]
            demo = demos[idx]
            guide_policy_scope = guide_policy_scopes[idx]
            obs, _ = self.env.reset()
            obs = np.array(obs)
            guide_policy_pointer = 0
            states_traj = [obs]
            action_traj = []
            for i in range(1, total_timesteps + 1):
                explore_policy = True
                self.global_step += 1
                if guide_policy_scope > 0 and guide_policy_pointer < guide_policy_scope:
                    action = demo[:guide_policy_scope][guide_policy_pointer]
                    guide_policy_pointer += 1
                else:
                    explore_policy = True
                    action = self._act(state=obs, weights=w, epsilon=0.5)
                action_traj.append(action)
                next_obs, vec_reward, terminated, truncated, info = self.env.step(action)
                states_traj.append(next_obs)
                reward = np.dot(vec_reward, w)
                next_obs = np.array(next_obs)
                if explore_policy == True:
                    self.replay_memory.append([obs, action, reward, next_obs, terminated, w])

                if self.global_step >= START_TRAINING_AFTER:
                    # print("Train starts -----------------------")
                    self.training_step()
                    if self.global_step % COPY_TO_TARGET_EVERY == 0:
                        self.target_model.set_weights(self.model.get_weights())

                if self.global_step % eval_freq == 0:
                    rolling_ops = []
                    utility_losses = []

                    for c_i in range(len(corners)):
                        w = corners[c_i]
                        u_threshold = u_thresholds[c_i]
                        u = self.play_a_episode(env=eval_env, pref_w=w, agent=self, demo=demo[:guide_policy_scope])
                        print(f"u:{u}\tu_thres:{u_threshold}")
                        utility_loss = abs(u - u_threshold)
                        utility_losses.append(utility_loss)
                        if u >= u_threshold:
                            print(f"@:{w} -- reach threshold")
                            rolling_ops.append(-2)  # roll out
                        else:
                            rolling_ops.append(0)
                    rolling_ops = np.array(rolling_ops)
                    guide_policy_scopes += rolling_ops
                    guide_policy_scopes = np.minimum(guide_policy_scopes, max_guide_policy_scopes)
                    guide_policy_scopes = np.maximum(guide_policy_scopes, np.zeros_like(guide_policy_scopes))
                    print(guide_policy_scopes)
                    guide_policy_scope = guide_policy_scopes[idx]
                    print(f"guide_policy_scope:{guide_policy_scope}\t demo:{demos[idx]}\tw_c:{corners[idx]}")
                    print(f"utility loss:{np.average(utility_losses)}\t timestep:{i}"
                          f"\n------------------------------")
                    if guide_policy_scope == 0:
                        break

                if terminated or truncated:

                    print(
                        # f"state_traj_train:{states_traj}\t"
                        f"action_traj:{action_traj}"
                        f"rewards:{vec_reward}\t"
                        # f"utility:{np.dot(vec_reward, w)}"
                    )
                    states_traj = []
                    action_traj = []
                    guide_policy_pointer = 0
                    obs, _ = self.env.reset()

                else:
                    obs = next_obs

    def train_model(self, steps, save_per=100000, show_detail_per=1000, pref_space=PreferenceSpace(),
                    corner_weights=None, eval_env=None):
        """
        Train the network over a range of episodes.
        """

        for step in range(1, steps + 1):
            self.action_list = []
            eps = 0.5
            # Reset env
            state, _ = self.env.reset()
            state = np.array(state)  # Convert to float32 for tf

            episode_reward = 0
            rewards_vec = np.zeros(3)
            # weights = pref_space.sample()
            weights = np.array([0.34, 0.36, 0.3])
            while True:
                state, reward, done, rewards = self.play_one_step(np.array(state), eps, weights)
                steps += 1
                episode_reward += reward
                rewards_vec += rewards
                if done:
                    break
            if step > START_TRAINING_AFTER:  # Wait for buffer to fill up a bit
                self.training_step()
                if step % COPY_TO_TARGET_EVERY == 0:
                    self.target_model.set_weights(self.model.get_weights())
                if step % save_per == 0 and step >= save_per and self.checkpoint:
                    self.model.save(self.model_path + str(step))
                u = self.play_a_episode(env=eval_env, pref_w=np.array([0,1]), agent=self,demo=[])
                print(f"tr"
                      f"yout_u:{u}")
            if step % show_detail_per == 0:
                if sum(self.action_list) > 0:
                    indices_of_one = [i for i, x in enumerate(self.action_list) if x == 1]
                else:
                    indices_of_one = "Not run at all"
                print(f"Epoch:{step}\t"
                      f"Pref:{weights}"
                      f"Epoch Reward/Cost:{episode_reward}\t"
                      f"Reward Vec:{rewards_vec}\t"
                      f"Epsilon:{np.round(eps, 2)}\t"
                      f"Actions:{sum(self.action_list)}\t"
                      f"@{indices_of_one}")
        # u = self.play_a_episode(env=eval_env, pref_w=np.array([0., 1.]), agent=self, demo=[])
        # print(f"utility:{u}")

    def play_a_episode(self, env, pref_w, agent, demo):
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
                action = agent.epsilon_greedy_policy(state, weights=pref_w, epsilon=0, evaluation=True)
            traj_actions.append(action)
            n_state, rewards, terminal, _, _ = env.step(action)
            n_state = np.array(n_state)
            traj_states.append(n_state)
            disc_return += gamma * np.dot(rewards, pref_w)
            gamma *= self.gamma
            state = n_state
            if steps > 100:
                break
        print(f"eval action traj:{traj_actions}"
              # f"state_traj:{traj_states}"

              )
        return disc_return


if __name__ == '__main__':
    # linear_support = LinearSupport(num_objectives=2,
    #                                epsilon=0.0)
    # deep_sea_treasure = DeepSeaTreasure()
    # eval_env = DeepSeaTreasure()
    # agent = ConditionedDQNAgent(env=deep_sea_treasure)
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
    env = Minecart()
    eval_env = Minecart()
    agent = ConditionedDQNAgent(env=env)
    corners = np.array([[0.34, 0.36, 0.3]])
    demos = [[2, 1, 3, 3, 3, 5, 5, 4, 0, 0, 1, 1, 1, 1, 1, 3, 3, 3, 2, 2, 1, 3]]
    u_thresholds = [[-0.0424]]
    # agent.jsRL_train(total_timesteps=4000,
    #                  eval_env=eval_env,
    #                  eval_freq=100,
    #                  corners=corners,
    #                  demos=demos,
    #                  u_thresholds=u_thresholds)
    agent.train_model(steps = 12000, eval_env=eval_env)
