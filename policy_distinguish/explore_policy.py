import numpy as np
from Algorithm.rl_algorithm.tab_q_agent import Tabular_Q_Agent
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
pref_space = PreferenceSpace()
# print(pref_space.iterate())

intervals = []
for i in range(len(corner_weights)):
    if i < len(corner_weights) - 1:
        intervals.append([corner_weights[i][1], corner_weights[i + 1][1]])
    else:
        intervals.append([corner_weights[i][1], 0])

print(intervals)
# print(len(intervals))
interval_pnt = 0
pre_interval_pnt = -np.inf
for pref_w in pref_space.iterate()[:-1]:
    if pref_w[1] == intervals[interval_pnt][1] and not pref_w[1] == 0:
        interval_pnt += 1
    print(f"pref_w:{pref_w}\tcorner_w:{corner_weights[interval_pnt]}")

    agent = Tabular_Q_Agent(env=simulator, gamma=0.99)
    pref = tuple(corner_weights[interval_pnt])
    traj = pref_traj_score[pref][0]
    print(f"demo traj:{traj}")
    if not pre_interval_pnt == interval_pnt:
        expected_utilities = agent.imitate_q_(demo=traj,
                                              pref_w=np.array(corner_weights[interval_pnt]))
        print(f"e_u:{expected_utilities}")
    expected_utility_list.append(np.array(expected_utilities))
    pre_interval_pnt = interval_pnt

# for pref in corner_weights:
#     agent = Tabular_Q_Agent(env=simulator)
#     pref = tuple(pref)
#     traj = pref_traj_score[pref][0]
#     print(f"demo traj:{traj}")
#     expected_rewards_list = agent.imitate_q_(demo=traj,
#                                              pref_w=np.array(pref))
#     expected_utility = np.dot(expected_rewards_list, pref)
#     print(f"expected_utility:{expected_utility}")
#     expected_utility_list.append(np.array(expected_utility))

# 找到最大长度
max_length = max(len(lst) for lst in expected_utility_list if len(lst) > 0)
# max_length = 40000
# 创建一个新的NumPy数组，填充为最大长度，使用列表的最后一个元素来填充，如果列表为空，则跳过
extended_lists = np.array([np.pad(lst, (0, max_length - len(lst)), 'constant', constant_values=lst[-1]) if len(
    lst) > 0 else np.full(max_length, np.nan) for lst in expected_utility_list])

# 打印结果
print(extended_lists)
print(len(extended_lists))
print(np.mean(extended_lists, axis=0))
np.save("expected_utility_list.npy", extended_lists)

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
