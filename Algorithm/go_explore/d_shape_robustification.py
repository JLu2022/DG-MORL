import keras
import numpy as np
from matplotlib import pyplot as plt

from util.utils import ACTIONS
from simulators.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure
from Algorithm.rl_algorithm.backward_Q_agent import Tabular_Q_Agent
from simulators.deep_sea_treasure.preference_space import PreferenceSpace
from Algorithm.go_explore.explore import traj_cost_calculate

archive = np.load("archives/archive.npy",
                  allow_pickle=True).item()
simulator = DeepSeaTreasure(img_repr=True)
cross = "---------------Policy starts-------------------"
pref_space = PreferenceSpace()
pref_list = pref_space.iterate()
pref_traj_score = {}

for pref in pref_list:
    # print(f"pref:{pref}")
    pref = tuple(pref)
    max_score = -np.inf
    max_traj = None
    for k in archive[pref].keys():
        if archive[pref][k].score > max_score:
            max_score = archive[pref][k].score
            max_traj = archive[pref][k].cell_traj
    if not pref == (0, 1):
        pref_traj_score[pref] = (max_traj, max_score)

# print(pref_traj_score)
other_simulator = DeepSeaTreasure()
# print(len(pref_list))

pref_list = np.array(
    [[0.01, 0.99], [0.65, 0.35]
        , [0.75, 0.25], [0.82, 0.18], [0.84, 0.16], [0.87, 0.13], [0.89, 0.11], [0.91, 0.09],
     [0.93, 0.07], [0.99, 0.01]]
)
pref_list = np.load("../../policy_distinguish/corner_weights.npy")

# print(pref_traj_score)
# pref_list = np.array(pref_list[::-1])
agent_list = []
trajs = []
for pref_index in range(len(pref_list)):
    print(cross)
    agent = Tabular_Q_Agent(env=simulator)
    agent_list.append(agent)
    pref = tuple(pref_list[pref_index])
    traj = pref_traj_score[pref][0]
# expected_utility_lists = []
# for pref_index in range(len(pref_list)):
#     agent = agent_list[pref_index]
    print(f"demo traj:{traj}")
    steps, reward_list, expected_utility_list = agent.imitate_q_(demo=traj, pref_w=np.array(pref_list[pref_index]),
                                                                agent_list=agent_list)
    # expected_utility_lists += expected_utility_list
    # print(f"pref:{pref_list[pref_index]}\treward_list:{reward_list}")
    agent.play_a_episode(pref=np.array(pref_list[pref_index]), agent=agent)
# plt.plot(expected_utility_lists)
# plt.show()
