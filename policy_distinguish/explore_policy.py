import numpy as np
from Algorithm.rl_algorithm.backward_Q_agent import Tabular_Q_Agent
from simulators.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure
from simulators.deep_sea_treasure.preference_space import PreferenceSpace
import matplotlib.pyplot as plt

robustified_rewards = []
agent_list = []
trajs = []
simulator = DeepSeaTreasure(img_repr=True)
corner_weights = np.load("corner_weights.npy")
pref_traj_score = np.load("pref_traj_score.npy", allow_pickle=True).item()
pref_traj_score = dict(pref_traj_score)
print(corner_weights)

expected_utility_list = []
cnt = 0
for i in range(len(corner_weights)):
    if i < len(corner_weights) - 1:
        next_corner_weight = corner_weights[i + 1]
    else:
        next_corner_weight = [1, 0]

    agent = Tabular_Q_Agent(env=simulator)
    agent_list.append(agent)
    pref = tuple(corner_weights[i])
    traj = pref_traj_score[pref][0]
    print(f"demo traj:{traj}\t"
          f"pref:{pref}")
    expected_utilities = agent.imitate_q_(demo=traj,
                                          pref_w=np.array(corner_weights[i]),
                                          agent_list=agent_list)
    # print(f"expected_utilities:{expected_utilities}")
    # num_of_ulist = int(round(corner_weights[i][-1] - next_corner_weight[-1], 2) * 100)
    # print(f"corner_weight:{corner_weights[i]}\t "
    #       f"next_w:{next_corner_weight[-1]}\t "
    #       f"feasible:{num_of_ulist}")

    # for _ in range(num_of_ulist):
        # print(round(corner_weights[i][-1] - next_corner_weight[-1],2))
        # cnt += 1
        # print(f"next_corner_weight:{next_corner_weight}")
    expected_utility_list.append(np.array(expected_utilities))
    # print(f"cnt:{cnt}")
# print(f"counter:{cnt}")
# lists = [np.array([1, 2, 3]), np.array([4, 5]), np.array([]), np.array([6, 7, 8, 9, 10])]

# 找到最大长度
max_length = max(len(lst) for lst in expected_utility_list if len(lst) > 0)

# 创建一个新的NumPy数组，填充为最大长度，使用列表的最后一个元素来填充，如果列表为空，则跳过
extended_lists = np.array([np.pad(lst, (0, max_length - len(lst)), 'constant', constant_values=lst[-1]) if len(lst) > 0 else np.full(max_length, np.nan) for lst in expected_utility_list])

# 打印结果
print(extended_lists)
print(len(extended_lists))
print(np.mean(extended_lists, axis=1))
np.save("expected_utility_list.npy", expected_utility_list)

#     episode_reward, episode_rewards = agent.play_a_episode(pref=np.array(w), agent=agent)
#     print(f"w:{w}|\texpected_utility_list:{expected_utility_list}")
#     robustified_rewards.append(episode_rewards)
#
# robustified_rewards = np.array(robustified_rewards)
#
# plt.scatter(robustified_rewards[:, 0], robustified_rewards[:, 1], color='r', marker='o',
#             label='points found by robustification')
# plt.xlabel('time')
# plt.ylabel('treasure')
# plt.legend()
# plt.show()
