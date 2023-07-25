"""
reference:
https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/ppo/ppo.py
"""
import random

import tensorflow as tf
# import tensorflow_probability as tfp
from keras.layers import Dense, Input
import numpy as np
import matplotlib.pyplot as plt
from Algorithm.rl_algorithm.PPO.buffer import PPOBuffer
from Algorithm.rl_algorithm.PPO.distributions import DiagonalGaussian


class PPO(object):
    def __init__(self, env, h_layers=[64, 64], seed=0, steps_per_epoch=4000, epochs=50, gamma=0.99, lam=0.97,
                 clip_ratio=0.2, lr_a=3e-4, lr_c=1e-3, train_a_iters=80, train_c_iters=80, max_ep_len=1000,
                 kl_target=0.01, ent_weight=0.001, save_freq=100, save_path='', mode="continuous"):

        tf.random.set_seed(seed)
        np.random.seed(seed)
        self.mode = mode
        self.env = env

        if self.mode == "continuous":
            self.state_dim = env.observation_space.shape[0]
            self.action_dim = env.action_space.shape[0]
            action_bound = [env.action_space.low, env.action_space.high]
            action_bound[0] = action_bound[0].reshape(1, self.action_dim)
            action_bound[1] = action_bound[1].reshape(1, self.action_dim)
        else:
            self.state_dim = env.observation_space
            self.action_dim = env.action_space
            action_bound = [0, self.action_dim - 1]
        self.action_bound = action_bound

        self.steps_per_epoch = steps_per_epoch
        self.max_ep_len = max_ep_len
        self.train_a_iters = train_a_iters
        self.train_c_iters = train_c_iters
        self.epochs = epochs
        self.gamma = gamma
        self.lam = lam
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.clip_ratio = clip_ratio
        self.kl_target = kl_target
        self.ent_weight = ent_weight
        self.save_freq = save_freq
        self.save_path = save_path

        self.actor = self.build_actor_net(h_layers)
        self.critic = self.build_critic_net(h_layers)

        # important
        lr_schedule_a = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_a,
                                                                       decay_steps=self.train_a_iters * self.epochs / 5.,
                                                                       decay_rate=0.96, staircase=True)
        lr_schedule_c = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_c,
                                                                       decay_steps=self.train_c_iters * self.epochs / 5.,
                                                                       decay_rate=0.96, staircase=True)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_a)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_c)

    def build_actor_net(self, h_layers, activation=tf.nn.relu6):
        print(self.state_dim)
        inputs = Input(shape=[self.state_dim, ])
        initializer = tf.keras.initializers.Orthogonal()
        for i in range(len(h_layers)):
            if i == 0:
                x = Dense(h_layers[i], activation=activation, kernel_initializer=initializer)(inputs)
            else:
                x = Dense(h_layers[i], activation=activation, kernel_initializer=initializer)(x)
        # mean
        mean = Dense(self.action_dim, activation=None, kernel_initializer=initializer)(x)
        # mean = Dense(self.action_dim, activation=tf.nn.softsign, kernel_initializer=initializer)(x)
        # mean = (mean + 1.) * (self.action_bound[1] - self.action_bound[0]) / 2. + self.action_bound[0]

        # std
        log_std = Dense(self.action_dim, kernel_initializer=initializer)(x)
        std = tf.math.exp(log_std)

        discrete_prob = Dense(self.action_dim, activation="softmax", kernel_initializer=initializer)(x)
        if self.mode == "continuous":
            return tf.keras.Model(inputs=inputs, outputs=[mean, std, log_std])
        if self.mode == "discrete":
            return tf.keras.Model(inputs=inputs, outputs=discrete_prob)

    def build_critic_net(self, h_layers, activation=tf.nn.relu6):
        inputs = Input(shape=[self.state_dim, ])
        initializer = tf.keras.initializers.Orthogonal()
        for i in range(len(h_layers)):
            if i == 0:
                x = Dense(h_layers[i], activation=activation, kernel_initializer=initializer)(inputs)
            else:
                x = Dense(h_layers[i], activation=activation, kernel_initializer=initializer)(x)
        outputs = Dense(1, activation=None, kernel_initializer=initializer)(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    @tf.function
    def sample_action(self, observation):
        # shape (1,act_dim)
        mean, std, log_std = self.actor(observation[np.newaxis, :])
        pi = DiagonalGaussian(mean, std, log_std)
        # action = tf.clip_by_value(pi.sample(), self.action_bound[0], self.action_bound[1])
        action = pi.sample()

        # shape (1,1)
        value = self.critic(observation[np.newaxis, :])
        return action[0], value[0, 0]

    # @tf.function
    def sample_discrete_action(self, observation, test=False):
        # print(f"obs:{observation}")
        prob = self.actor(observation[np.newaxis, :])
        if test:
            print(f"action prob:{prob}\t state:{observation}")
        action = tf.random.categorical(tf.math.log(prob), num_samples=1)
        value = self.critic(observation[np.newaxis, :])
        return action[0], value[0, 0]

    # @tf.function
    def update_actor(self, states, actions, advantages, old_pi):
        with tf.GradientTape() as tape:
            mean, std, log_std = self.actor(states)
            pi = DiagonalGaussian(mean, std, log_std)

            log_pi = pi.log_likelihood(actions)
            log_old_pi = old_pi.log_likelihood(actions)
            ratio = tf.exp(log_pi - log_old_pi)
            surr = tf.math.minimum(ratio * advantages,
                                   tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages)
            loss = -tf.math.reduce_mean(surr)

            approx_ent = tf.math.reduce_mean(-log_pi)
            loss -= approx_ent * self.ent_weight  # maximize the entropy to encourage exploration

            approx_kl = tf.math.reduce_mean(log_old_pi - log_pi)

        grad = tape.gradient(loss, self.actor.trainable_weights)
        # very important to clip gradient
        grad, grad_norm = tf.clip_by_global_norm(grad, 0.5)
        self.actor_optimizer.apply_gradients(zip(grad, self.actor.trainable_weights))
        return approx_kl

    def update_discrete_actor(self, states, actions, advantages, old_pi):
        with tf.GradientTape() as tape:
            pi = self.actor(states)
            actions = tf.cast(tf.reduce_mean(actions, axis=1), tf.int32)
            action_one_hot = tf.one_hot(actions, self.action_dim)
            # print(f"pi:{pi}\taction_one_hot:{action_one_hot}")
            probs = tf.reduce_sum(tf.multiply(pi, action_one_hot), axis=1)
            log_pi = tf.math.log(probs)

            old_probs = tf.reduce_sum(tf.multiply(old_pi, action_one_hot), axis=1)
            log_old_pi = tf.math.log(old_probs)

            ratio = tf.exp(log_pi - log_old_pi)
            surr = tf.math.minimum(ratio * advantages,
                                   tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages)
            loss = -tf.math.reduce_mean(surr)

            approx_ent = tf.math.reduce_mean(-log_pi)
            loss -= approx_ent * self.ent_weight  # maximize the entropy to encourage exploration

            approx_kl = tf.math.reduce_mean(log_old_pi - log_pi)

        grad = tape.gradient(loss, self.actor.trainable_weights)
        # very important to clip gradient
        grad, grad_norm = tf.clip_by_global_norm(grad, 0.5)
        self.actor_optimizer.apply_gradients(zip(grad, self.actor.trainable_weights))
        return approx_kl

    # @tf.function
    def update_critic(self, states, returns):
        with tf.GradientTape() as tape:
            advantage = returns - self.critic(states)
            loss = tf.math.reduce_mean(0.5 * tf.square(advantage))
        grad = tape.gradient(loss, self.critic.trainable_weights)
        # very important to clip gradient
        grad, grad_norm = tf.clip_by_global_norm(grad, 0.5)
        self.critic_optimizer.apply_gradients(zip(grad, self.critic.trainable_weights))

    # it should be commented to use 'kl'.
    # @tf.function
    def update(self, states, actions, returns, advantages):
        mean, std, log_std = self.actor(states)
        old_pi = DiagonalGaussian(mean, std, log_std)

        for i in range(self.train_a_iters):
            kl = self.update_actor(states, actions, advantages, old_pi)
            if kl > tf.constant(1.5 * self.kl_target):
                print('Early stopping at step %d due to reaching max kl.' % i)
                break
        for i in range(self.train_c_iters):
            self.update_critic(states, returns)

    def update_discrete(self, states, actions, returns, advantages):
        prob = self.actor(states)
        old_pi = prob

        for i in range(self.train_a_iters):
            kl = self.update_discrete_actor(states, actions, advantages, old_pi)
            if kl > tf.constant(1.5 * self.kl_target):
                print('Early stopping at step %d due to reaching max kl.' % i)
                break
        for i in range(self.train_c_iters):
            self.update_critic(states, returns)

    def train(self, seed=116):
        # self.env.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        buffer = PPOBuffer(self.state_dim, self.action_dim, size=self.steps_per_epoch, gamma=self.gamma, lam=self.lam)

        all_episode_reward = []
        episode_count = 0

        # epoch is used when training the policy network.
        for epoch in range(self.epochs):
            episode_reward = 0
            ep_len = 0  # episode_length
            state, _ = self.env.reset()
            # state = np.array([state])
            state = state.astype(np.float32)
            for ii in range(self.steps_per_epoch):
                # if RENDER:
                #    self.env.render()

                # ------------------- Play Episode ------------------------#
                ep_len += 1

                # shape of action, state : (1,) and (3,)
                if self.mode == "continuous":
                    action, value = self.sample_action(state)
                if self.mode == "discrete":
                    action, value = self.sample_discrete_action(state)
                action = action.numpy()
                value = value.numpy()
                state_, rew, done, _, _ = self.env.step(action)
                # print(state_)
                # state_ = np.array([state_])
                state_ = state_.astype(np.float32)

                buffer.store(state, action, rew, value)
                state = state_

                episode_reward += rew
                # ------------------- End of Play Episode ------------------------#
                if done or ii == self.steps_per_epoch - 1 or ep_len == self.max_ep_len:
                    # if:
                    # 1. Done
                    # 2. reach maximum training steps
                    # 3. episode reaches the maximum length

                    if done:
                        last_value = 0
                    else:
                        last_value = self.critic(state_[np.newaxis, :])
                        last_value = last_value[0, 0]
                    buffer.finish_path(last_value)

                    episode_count += 1
                    print(f"Training | episode:{episode_count}  | "
                          f"epoch: {epoch} | "
                          f"Episode Reward: {episode_reward} | "
                          f"Episode Length: {ep_len}")

                    if (episode_count + 1) % self.save_freq == 0:
                        self.actor.save_weights(self.save_path + 'actor_checkpoint' + str(episode_count))
                        self.critic.save_weights(self.save_path + 'critic_checkpoint' + str(episode_count))

                    if len(all_episode_reward) < 5:  # record running episode reward
                        all_episode_reward.append(episode_reward)
                    else:
                        all_episode_reward.append(episode_reward)
                        all_episode_reward[-1] = (np.mean(all_episode_reward[-5:]))  # smoothing

                    state, _ = self.env.reset()
                    # state = np.array([state])
                    state = state.astype(np.float32)
                    episode_reward = 0
                    ep_len = 0

            state_buf, act_buf, adv_buf, ret_buf = buffer.get()
            state_tensor = tf.convert_to_tensor(np.vstack(state_buf), dtype=tf.float32)
            act_tensor = tf.convert_to_tensor(np.vstack(act_buf), dtype=tf.float32)
            adv_tensor = tf.convert_to_tensor(np.vstack(adv_buf), dtype=tf.float32)
            ret_tensor = tf.convert_to_tensor(np.vstack(ret_buf), dtype=tf.float32)

            adv_tensor = tf.squeeze(adv_tensor, axis=1)
            if self.mode == "continuous":
                self.update(state_tensor, act_tensor, ret_tensor, adv_tensor)
            if self.mode == "discrete":
                self.update_discrete(state_tensor, act_tensor, ret_tensor, adv_tensor)

        plt.figure()
        plt.plot(all_episode_reward)
        plt.xlabel('episodes')
        plt.ylabel('total reward per episode')
        plt.show()

    def test(self, path):

        # self.actor.load_weights(path)

        # self.critic.load_weights(path)

        for _ in range(10):
            traj = ""
            state, _ = self.env.reset()
            # state = np.array([state])
            # state = state.astype(np.float32)

            print("Trying a new epoch...")
            traj += "(" + str(int(state[-2])) + "," + str(int(state[-1])) + ")-"
            while True:

                # shape of action, state : (1,) and (3,)
                action, value = self.sample_discrete_action(state)
                state_, rew, done, _, _ = self.env.step(action.numpy())
                # state_ = np.array([state_])
                # state_ = state_.astype(np.float32)
                state = state_
                traj += "(" + str(int(state[-2])) + "," + str(int(state[-1])) + ")-"
                # print(f"{state} - ")
                if done:
                    print("Done " + traj)
                    break

    def train_with_traj(self, seed=116, traj=None, pref=None):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        buffer = PPOBuffer(self.state_dim, self.action_dim, size=self.steps_per_epoch, gamma=self.gamma, lam=self.lam)
        all_episode_reward = []
        episode_count = 0

        for epoch in range(self.epochs):
            state_list = []
            episode_reward = 0
            ep_len = 0  # episode_length
            done = False

            for i in range(len(traj)):
                demonstration_state = traj[i]
                _, demonstration_state = self.env.reset_to_state(demonstration_state)
                demonstration_state = np.array(demonstration_state)
                demonstration_state = np.concatenate((demonstration_state, pref))
                demonstration_state = demonstration_state.astype(np.float32)

                action, value = self.sample_discrete_action(demonstration_state)
                action = action.numpy()
                value = value.numpy()
                rew, _, done, agent_state = self.env.step(action)  # rewards, image, terminal, position

                state_list.append(agent_state)
                if not i == len(traj) - 1:
                    if agent_state == tuple(demonstration_state[i + 1][:2]):
                        rew = 1
                    else:
                        rew = 0
                else:
                    break
                episode_reward += rew
                buffer.store(demonstration_state, action, rew, value)
            # print(f"episode reward:{episode_reward}")
            last_value = 0
            buffer.finish_path(last_value)
            # print(buffer.ptr)
            if epoch >= 100:
                state_buf, act_buf, adv_buf, ret_buf = buffer.get()
                state_tensor = tf.convert_to_tensor(np.vstack(state_buf), dtype=tf.float32)
                act_tensor = tf.convert_to_tensor(np.vstack(act_buf), dtype=tf.float32)
                adv_tensor = tf.convert_to_tensor(np.vstack(adv_buf), dtype=tf.float32)
                ret_tensor = tf.convert_to_tensor(np.vstack(ret_buf), dtype=tf.float32)

                adv_tensor = tf.squeeze(adv_tensor, axis=1)
                self.update_discrete(state_tensor, act_tensor, ret_tensor, adv_tensor)

                # buffer.clear_cache(self.state_dim, self.action_dim, size=self.steps_per_epoch,
                #                    gamma=self.gamma, lam=self.lam)
        # print(f"episode reward:{episode_reward}")
        print(f"traj:{state_list}")

    def test(self, path):

        # self.actor.load_weights(path)

        # self.critic.load_weights(path)

        for _ in range(10):
            traj = ""
            state, _ = self.env.reset()
            # state = np.array([state])
            # state = state.astype(np.float32)

            print("Trying a new epoch...")
            traj += "(" + str(int(state[-2])) + "," + str(int(state[-1])) + ")-"
            reward = 0
            while True:

                # shape of action, state : (1,) and (3,)
                action, value = self.sample_discrete_action(state, test=True)
                state_, rew, done, _, _ = self.env.step(action.numpy())
                reward += rew
                # state_ = np.array([state_])
                # state_ = state_.astype(np.float32)
                state = state_
                traj += "(" + str(int(state[-2])) + "," + str(int(state[-1])) + ")-"
                # print(f"{state} - ")
                if done:
                    print("Done " + traj + f"cost {reward}")

                    break

    def train_go_explore(self, seed=116, reset_to=None, target_reward=100):
        # self.env.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        buffer = PPOBuffer(self.state_dim, self.action_dim, size=self.steps_per_epoch, gamma=self.gamma,
                           lam=self.lam)

        all_episode_reward = []
        episode_count = 0

        # epoch is used when training the policy network.
        for epoch in range(self.epochs):
            episode_reward = 0
            ep_len = 0  # episode_length
            state, _ = self.env.reset_to_state(reset_to)
            # state = np.array([state])
            # state = state.astype(np.float32)

            for ii in range(self.steps_per_epoch):

                # ------------------- Play Episode ------------------------#
                ep_len += 1

                # shape of action, state : (1,) and (3,)
                if self.mode == "continuous":
                    action, value = self.sample_action(state)
                if self.mode == "discrete":
                    action, value = self.sample_discrete_action(state)
                action = action.numpy()
                value = value.numpy()
                state_, rew, done, _, _ = self.env.step(action)
                # print((state_[-2],state_[-1]))
                # print((state_[-2],state_[-1])==(0,24))
                # print(f"state:{state},state_:{state_},done:{done}")
                # state_ = np.array([state_])
                # state_ = state_.astype(np.float32)

                buffer.store(state, action, rew, value)
                state = state_

                episode_reward += rew
                # ------------------- End of Play Episode ------------------------#
                if done or ii == self.steps_per_epoch - 1 or ep_len == self.max_ep_len:
                    # if:
                    # 1. Done
                    # 2. reach maximum training steps
                    # 3. episode reaches the maximum length

                    if done:
                        last_value = 0
                    else:
                        last_value = self.critic(state_[np.newaxis, :])
                        last_value = last_value[0, 0]
                    buffer.finish_path(last_value)

                    episode_count += 1
                    # print(f"Training | episode:{episode_count}  | "
                    #       f"epoch: {epoch} | "
                    #       f"Episode Reward: {episode_reward} | "
                    #       f"Episode Length: {ep_len}")

                    if (episode_count + 1) % self.save_freq == 0:
                        self.actor.save_weights(self.save_path + 'actor_checkpoint' + str(episode_count))
                        self.critic.save_weights(self.save_path + 'critic_checkpoint' + str(episode_count))

                    if len(all_episode_reward) < 5:  # record running episode reward
                        all_episode_reward.append(episode_reward)
                        # print("if", all_episode_reward[-1])
                    else:
                        all_episode_reward.append(episode_reward)
                        all_episode_reward[-1] = (np.mean(all_episode_reward[-5:]))  # smoothing
                        # print("else", all_episode_reward[-1])
                        if np.round(all_episode_reward[-1], 2) >= np.round(target_reward, 2):
                            print(f"reach standard -- {all_episode_reward[-1]}, moving to earlier state")
                            buffer.clear_cache(self.state_dim, self.action_dim, size=self.steps_per_epoch,
                                               gamma=self.gamma, lam=self.lam)
                            return
                    state, _ = self.env.reset_to_state(reset_to)
                    # state = np.array([state])
                    # state = state.astype(np.float32)
                    episode_reward = 0
                    ep_len = 0

            state_buf, act_buf, adv_buf, ret_buf = buffer.get()
            state_tensor = tf.convert_to_tensor(np.vstack(state_buf), dtype=tf.float32)
            act_tensor = tf.convert_to_tensor(np.vstack(act_buf), dtype=tf.float32)
            adv_tensor = tf.convert_to_tensor(np.vstack(adv_buf), dtype=tf.float32)
            ret_tensor = tf.convert_to_tensor(np.vstack(ret_buf), dtype=tf.float32)

            adv_tensor = tf.squeeze(adv_tensor, axis=1)
            if self.mode == "continuous":
                self.update(state_tensor, act_tensor, ret_tensor, adv_tensor)
            if self.mode == "discrete":
                self.update_discrete(state_tensor, act_tensor, ret_tensor, adv_tensor)

        # plt.figure()
        # plt.plot(all_episode_reward)
        # plt.xlabel('episodes')
        # plt.ylabel('total reward per episode')
        # plt.show()

    def robustification_train_(self, seed=116, reward_bar=0, pref=[0, 1], reset_to=None):
        # self.env.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        buffer = PPOBuffer(self.state_dim, self.action_dim, size=self.steps_per_epoch, gamma=self.gamma,
                           lam=self.lam)

        all_episode_reward = []
        episode_count = 0

        # epoch is used when training the policy network.
        for epoch in range(self.epochs):
            print(f"epoch:{epoch}")
            state_list = []
            episode_reward = 0
            ep_len = 0  # episode_length
            _, state = self.env.reset_to_state(reset_to)
            state = np.array(state)
            state = np.concatenate((state, pref))
            state = state.astype(np.float32)

            for ii in range(self.steps_per_epoch):
                state_list.append(tuple(state[:2]))
                # ------------------- Play Episode ------------------------#
                ep_len += 1

                # shape of action, state : (1,) and (3,)
                action, value = self.sample_discrete_action(state)
                action = action.numpy()
                value = value.numpy()
                rew, _, done, state_ = self.env.step(action)
                rew = np.dot(pref, rew)
                state_ = np.array(state_)
                state_ = np.concatenate((state_, pref))
                state_ = state_.astype(np.float32)

                buffer.store(state, action, rew, value)
                state = state_

                episode_reward += rew
                # ------------------- End of Play Episode ------------------------#
                if done or ii == self.steps_per_epoch - 1 or ep_len == self.max_ep_len:
                    # if:
                    # 1. Done
                    # 2. reach maximum training steps
                    # 3. episode reaches the maximum length
                    state_list.append(tuple(state[:2]))
                    if done:
                        last_value = 0
                    else:
                        last_value = self.critic(state_[np.newaxis, :])
                        last_value = last_value[0, 0]
                    buffer.finish_path(last_value)

                    episode_count += 1
                    if (epoch + 1) % 20 == 0:
                        print(f"Training | episode:{episode_count}  | "
                              f"epoch: {epoch} | "
                              f"Episode Reward: {episode_reward} | "
                              f"Episode Length: {ep_len} |"
                              f"avg r: {np.mean(all_episode_reward[-5:-1])}")

                    # if (episode_count + 1) % self.save_freq == 0:
                    #     self.actor.save_weights(self.save_path + 'actor_checkpoint' + str(episode_count))
                    #     self.critic.save_weights(self.save_path + 'critic_checkpoint' + str(episode_count))

                    if len(all_episode_reward) < 5:  # record running episode reward
                        all_episode_reward.append(episode_reward)
                        # print("if", all_episode_reward[-1])
                    else:
                        all_episode_reward.append(episode_reward)
                        # all_episode_reward[-1] = (np.mean(all_episode_reward[-5:]))  # smoothing
                        # print("else", all_episode_reward[-1])
                        # if np.round(np.mean(all_episode_reward[-5:-1]), 1) >= np.round(reward_bar, 2):
                        epi_rew_vec = 0
                        for _ in range(5):
                            epi_rew, _, _ = self.generate_experience(reset_to=reset_to, pref=pref, show_detail=False)
                            epi_rew_vec += np.dot(epi_rew, pref)
                        print(f"avg achievement:{int(epi_rew_vec / 5*100)/100.0}  |  bar:{int(reward_bar*100)/100.0}")
                        # if self.generate_experience(reset_to=reset_to, pref=pref) >= np.round(reward_bar, 2):
                        # print(f"avg performance:{epi_rew_vec / 5}")
                        if int(epi_rew_vec / 5*100)/100.0 >= int(reward_bar*100)/100.0:
                            print(
                                f"reach standard -- achieve: {int(epi_rew_vec / 5*100)/100.0}\t"
                                f"bar:{int(reward_bar*100)/100.0}")
                            self.generate_experience(reset_to=reset_to, pref=pref)
                            # buffer.clear_cache(self.state_dim, self.action_dim, size=self.steps_per_epoch,
                            #                    gamma=self.gamma, lam=self.lam)
                            return
                    _, state = self.env.reset_to_state(reset_to)
                    state = np.array(state)
                    state = np.concatenate((state, pref))
                    state = state.astype(np.float32)
                    episode_reward = 0
                    ep_len = 0

            state_buf, act_buf, adv_buf, ret_buf = buffer.get()
            state_tensor = tf.convert_to_tensor(np.vstack(state_buf), dtype=tf.float32)
            act_tensor = tf.convert_to_tensor(np.vstack(act_buf), dtype=tf.float32)
            adv_tensor = tf.convert_to_tensor(np.vstack(adv_buf), dtype=tf.float32)
            ret_tensor = tf.convert_to_tensor(np.vstack(ret_buf), dtype=tf.float32)

            adv_tensor = tf.squeeze(adv_tensor, axis=1)
            self.update_discrete(state_tensor, act_tensor, ret_tensor, adv_tensor)

    def robustification_train(self, seed=116, reward_bar=0, pref=[0, 1], reset_to=None):

        np.random.seed(seed)
        tf.random.set_seed(seed)
        buffer = PPOBuffer(self.state_dim, self.action_dim, size=self.steps_per_epoch, gamma=self.gamma, lam=self.lam)
        all_episode_reward = []
        episode_count = 0

        for epoch in range(self.epochs):
            state_list = []
            episode_reward = 0
            ep_len = 0  # episode_length
            _, state = self.env.reset_to_state(reset_to)
            state = np.array(state)
            state = np.concatenate((state, pref))
            state = state.astype(np.float32)
            done = False
            while not done:
                state_list.append(tuple(state[:2]))
                # ------------------- Play Episode ------------------------#
                ep_len += 1
                action, value = self.sample_discrete_action(state)
                action = action.numpy()
                value = value.numpy()
                rew, _, done, state_ = self.env.step(action)  # rewards, image, terminal, position
                rew = np.dot(pref, rew)

                state_ = np.array(state_)
                state_ = np.concatenate((state_, pref))
                state_ = state_.astype(np.float32)

                buffer.store(state, action, rew, value)
                state = state_

                episode_reward += rew
                # ------------------- End of Play Episode ------------------------#
                if done:
                    state_list.append(tuple(state[:2]))

                    last_value = 0
                    buffer.finish_path(last_value)
                    episode_count += 1
                    if (epoch + 1) % 200 == 0:
                        print(f"Training | episode:{episode_count}|"
                              f"Episode Reward: {episode_reward}|"
                              f"Episode Length: {ep_len}|"
                              f"avg r:{np.mean(all_episode_reward[-20:])}|"
                              )

                    all_episode_reward.append(episode_reward)

                    _, state = self.env.reset_to_state(reset_to)

                    state = np.array(state)
                    state = np.concatenate((state, pref))
                    state = state.astype(np.float32)

                    episode_reward = 0
                    ep_len = 0

                    # buffer.clear_cache(self.state_dim, self.action_dim, size=self.steps_per_epoch,
                    #                    gamma=self.gamma, lam=self.lam)

                    if np.mean(all_episode_reward[-20:]) > reward_bar:
                        print(f"mean learn reward:{np.mean(all_episode_reward[-20:])}\treward_bar:{reward_bar}\t"
                              f"traj:{state_list}\tavg r:{all_episode_reward[-20:]}")
                        return

            if buffer.ptr == buffer.max_size:
                state_buf, act_buf, adv_buf, ret_buf = buffer.get()
                state_tensor = tf.convert_to_tensor(np.vstack(state_buf), dtype=tf.float32)
                act_tensor = tf.convert_to_tensor(np.vstack(act_buf), dtype=tf.float32)
                adv_tensor = tf.convert_to_tensor(np.vstack(adv_buf), dtype=tf.float32)
                ret_tensor = tf.convert_to_tensor(np.vstack(ret_buf), dtype=tf.float32)

                adv_tensor = tf.squeeze(adv_tensor, axis=1)
                self.update_discrete(state_tensor, act_tensor, ret_tensor, adv_tensor)
        print(f"mean learn reward:{np.mean(all_episode_reward[-5:])}\t agent traj:{state_list}")

    def generate_experience(self, reset_to=1, pref=None, show_detail=True):
        self.action_list = []
        state_list = []
        _, state = self.env.reset_to_state(reset_to)
        state_list.append(state)
        state = np.array(state)
        state = np.concatenate((state, pref))
        state = np.float32(state)  # Convert to float32 for tf

        episode_reward = 0

        while True:
            action, value = self.sample_discrete_action(state)
            self.action_list.append(int(action))
            rew, _, done, state = self.env.step(action.numpy())
            state_list.append(state)
            state = np.array(state)
            state = np.concatenate((state, pref))
            state = state.astype(np.float32)
            episode_reward += rew
            if done:
                break
        if show_detail:
            print(
                f"pref:{pref}\tstart from:{reset_to}\tstate_list:{state_list}\tepisode_reward:{episode_reward}\tscalar:{np.dot(episode_reward, pref)}")
        return episode_reward, state_list, pref
