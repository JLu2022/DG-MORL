import matplotlib.pyplot as plt
import numpy as np
from simulators.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure
from simulators.deep_sea_treasure.preference_space import PreferenceSpace

ACTIONS = {0: "up", 1: "down", 2: "left", 3: "right"}
pref_space = PreferenceSpace()
pref_list = pref_space.iterate()


class Tabular_Q_Agent:
    def __init__(self, env, gamma):
        self.gamma = gamma
        self.env = env
        self.actions = [i for i in range(self.env.action_space)]
        self.Q_table = np.random.rand(self.env.size, self.env.size, self.env.action_space)

    def epsilon_greedy(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.actions)
        else:
            return np.argmax(self.Q_table[state[0]][state[1]])

    def imitate_q_(self, pref_w=None, demo=None):
        epsilon = 0.5
        alpha = 0.1
        expected_utility_list = []
        num_of_parts = len(demo)
        train_cnt = 0

        overall_utility_thres = self.get_utility_threshold(demo=demo[1:], pref_w=pref_w)
        print(f"overall_utility_thres:{overall_utility_thres}")

        utility, _ = self.play_sub_episode(pref_w=pref_w, reset_to=demo[0])
        while utility < overall_utility_thres:
            terminal = False
            h_pointer = len(demo) - 2
            action_list = self.env.state_traj_to_actions(state_demo=demo[:h_pointer])
            image, state = self.env.reset_to_state(reset_to=demo[0])

            for action in action_list:  # guide policy takes over
                rews, n_image, terminal, n_state, shaping_reward, t_r = self.env.step(action)
                reward = np.dot(rews, pref_w)

                # -------------------Train------------------#
                TD_target = reward + self.gamma * np.max(self.Q_table[n_state[0]][n_state[1]])
                TD_error = TD_target - self.Q_table[state[0]][state[1]][action]

                # print(f"TD_error:{TD_error}")
                self.Q_table[state[0]][state[1]][action] += alpha * TD_error
                state = n_state
                train_cnt+=1

            while not terminal:  # explore policy takes over
                action = self.epsilon_greedy(state, epsilon)
                rews, n_image, terminal, n_state, shaping_reward, t_r = self.env.step(action)
                reward = np.dot(rews, pref_w)

                # -------------------Train------------------#
                TD_target = reward + self.gamma * np.max(self.Q_table[n_state[0]][n_state[1]])
                TD_error = TD_target - self.Q_table[state[0]][state[1]][action]

                # print(f"TD_error:{TD_error}")
                self.Q_table[state[0]][state[1]][action] += alpha * TD_error
                state = n_state
                train_cnt += 1
            utility, _ = self.play_sub_episode(pref_w=pref_w, reset_to=demo[0])

        # for i in range(1, num_of_parts - 1)[::-1]:
        #     # after find a better traj, reset the state index one step backward
        #     reset_idx = i
        #     sub_demo = demo[reset_idx:]
        #     utility_threshold = self.get_utility_threshold(demo=sub_demo[1:], pref_w=pref_w)
        #     eval_utility = -np.inf
        #     step = 0
        #     print(f"sub_demo:{sub_demo}\tutility_thres:{utility_threshold}")
        #     # while the agent does not learn a policy not worse than the demo
        #     while eval_utility < utility_threshold and step < 500:
        #         terminal = False
        #         step += 1
        #         # reset to the start point
        #         image, state = self.env.reset_to_state(reset_to=demo[reset_idx])
        #         print(f"reset to:{state}")
        #         while not terminal:
        #             action = self.epsilon_greedy(state, epsilon)
        #             rews, n_image, terminal, n_state, shaping_reward, t_r = self.env.step(action)
        #             reward = np.dot(rews, pref_w)
        #
        #             # -------------------Train------------------#
        #             TD_target = reward + self.gamma * np.max(self.Q_table[n_state[0]][n_state[1]])
        #             TD_error = TD_target - self.Q_table[state[0]][state[1]][action]
        #
        #             # print(f"TD_error:{TD_error}")
        #             self.Q_table[state[0]][state[1]][action] += alpha * TD_error
        #             state = n_state
        #
        #             train_cnt += 1
        #             if train_cnt % 10 == 0:
        #                 utility, _ = self.play_sub_episode(pref_w=pref_w, reset_to=demo[0])
        #                 expected_utility_list.append(utility)
        #         eval_utility, traj = self.play_sub_episode(pref_w=pref_w, reset_to=demo[reset_idx])
        #         print(f"utility_thres:{utility_threshold}\teval_utility:{eval_utility}\ttraj:{traj}")
        #         print("-------------------")
        return expected_utility_list

    def get_utility_threshold(self, demo, pref_w):
        utility = self.env.calculate_utility(demo=demo, pref_w=pref_w)
        return utility

    def play_sub_episode(self, reset_to, pref_w):
        terminal = False
        image, state = self.env.reset_to_state(reset_to=reset_to)
        vec_rewards = np.zeros(2)
        state_list = [state]
        gamma = 1
        while not terminal:
            action = self.epsilon_greedy(state, epsilon=0)
            rewards, n_image, terminal, n_state, shaping_reward, treasure_reward = self.env.step(action)
            vec_rewards += gamma * rewards
            gamma *= self.gamma
            state = n_state
            state_list.append(state)
        utility = np.dot(vec_rewards, pref_w)
        return utility, state_list

    def play_a_episode(self, pref_w, agent):
        env_try = DeepSeaTreasure()
        disc_return = 0
        gamma = self.gamma
        terminal = False
        image, state = env_try.reset()
        while not terminal:
            action = agent.epsilon_greedy(state, epsilon=0)
            rewards, n_image, terminal, n_state, shaping_reward, treasure_reward = env_try.step(action)
            disc_return += gamma * np.dot(rewards, pref_w)
            gamma *= self.gamma
            state = n_state
        return disc_return


if __name__ == '__main__':
    # print(generate_traj(size = 10))
    # size = 30
    # goal_pos = 899
    num_of_step = 1000000

    env = DeepSeaTreasure()
    q_agent = Tabular_Q_Agent(env)
    steps, reward_list = q_agent.imitate_q(
        demo=[(0, 0), (0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)],
        pref_w=np.array([0.84, 0.16]))
    print(reward_list)
    plt.plot(range(steps), reward_list)
    plt.show()
