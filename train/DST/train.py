import numpy as np
from Algorithm.rl_algorithm.tabular_Q import Tabular_Q_Agent
from simulators.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure
from simulators.deep_sea_treasure.preference_space import PreferenceSpace
import matplotlib.pyplot as plt

exec(open('data_collection.py').read())
print("Data_collection_finish.")
print("Start Corner w recognition...")
exec(open('policy_distinguish.py').read())
print("Corner w recognition finish.")
robustified_rewards = []
agent_list = []
trajs = []
simulator = DeepSeaTreasure(img_repr=True)
corner_weights = np.load("files/corner_weights.npy")
pref_traj_score = np.load("files/pref_traj_score.npy", allow_pickle=True).item()
pref_traj_score = dict(pref_traj_score)
# print(corner_weights)

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

for pref_w in corner_weights:
    pref_w = tuple(pref_w)
    demo = pref_traj_score[pref_w][0]
    print(f"w:{pref_w}\ttraj:{demo}")
# traj_to_10_9 = [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
#                     (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (8, 9), (9, 9), (10, 9)]
#
# traj_to_9_8 = [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
#                    (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8), (8, 8), (9, 8)]
#
# traj_to_7_7 = [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 7),
#                    (3, 7), (4, 7), (5, 7), (6, 7), (7, 7)]
#
# traj_to_7_6 = [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 6), (3, 6),
#                    (4, 6), (5, 6), (6, 6), (7, 6)]
#
# traj_to_4_5 = [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 5), (3, 5), (4, 5)]
# traj_to_4_4 = [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (4, 4)]
# traj_to_4_3 = [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (4, 3)]
# traj_to_3_2 = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (3, 2)]
# traj_to_2_1 = [(0, 0), (0, 1), (1, 1), (2, 1)]
# traj_to_1_0 = [(0, 0), (1, 0)]
# trajs = [traj_to_10_9, traj_to_9_8, traj_to_7_7, traj_to_7_6, traj_to_4_5, traj_to_4_4, traj_to_4_3, traj_to_3_2,
#          traj_to_2_1, traj_to_1_0]
for pref_w in corner_weights:
    agent = Tabular_Q_Agent(env=simulator, gamma=0.99)
    pref_w = tuple(pref_w)
    # if pref_w[1] > 0.7:
    #     traj = trajs[0]
    # elif pref_w[1] > 0.67:
    #     traj = trajs[1]
    # elif pref_w[1] > 0.66:
    #     traj = trajs[2]
    # elif pref_w[1] > 0.58:
    #     traj = trajs[3]
    # elif pref_w[1] > 0.54:
    #     traj = trajs[4]
    # elif pref_w[1] > 0.51:
    #     traj = trajs[5]
    # elif pref_w[1] > 0.47:
    #     traj = trajs[6]
    # elif pref_w[1] > 0.39:
    #     traj = trajs[7]
    # elif pref_w[1] > 0.21:
    #     traj = trajs[8]
    # elif pref_w[1] > 0:
    #     traj = trajs[9]
    # if True:
    demo = pref_traj_score[pref_w][0]
    print(f"w:{pref_w}\ndemo traj:{demo}")
    expected_utilities = agent.jsmoq_discrete(demo=demo,
                                              pref_w=np.array(pref_w))
        # if not expected_utilities:
        #     print(f"empty list @ w:{pref_w}\texpected_utilities:{expected_utilities}")
        # print(f"e_u:{expected_utilities}")
    expected_utility_list.append(np.array(expected_utilities))

# 找到最大长度
max_length = max(len(lst) for lst in expected_utility_list if len(lst) > 0)
max_length = 40000
# 创建一个新的NumPy数组，填充为最大长度，使用列表的最后一个元素来填充，如果列表为空，则跳过
extended_lists = np.array([np.pad(lst, (0, max_length - len(lst)), 'constant', constant_values=lst[-1]) if len(
    lst) > 0 else np.full(max_length, np.nan) for lst in expected_utility_list])

# 打印结果
print(extended_lists)
print(len(extended_lists))
print(np.mean(extended_lists, axis=0))
np.save("../../evaluation/expected_u/DST/expected_utility_list.npy", extended_lists)
