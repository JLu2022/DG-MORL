import keras
import numpy as np
# from rl_algorithm.tabular_q import Tabular_Q_Agent
from Algorithm.rl_algorithm.PPO.ppo import PPO
from simulators.discrete_grid_world import ImageGridWorld
from util.utils import ACTIONS
from simulators.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure
from simulators.deep_sea_treasure.preference_space import PreferenceSpace
from explore import traj_cost_calculate
from Algorithm.rl_algorithm.D_shaped_DQN import DQNAgent

archive = np.load("C:/Users/19233436/PycharmProjects/MOGOExplore/simulation/deep_sea_treasure/archive/archive.npy",
                  allow_pickle=True).item()
simulator = DeepSeaTreasure(img_repr=True)
cross = "---------------backwards-------------------"
pref_space = PreferenceSpace()
pref_list = pref_space.iterate()
pref_traj_score = {}
# for pref in pref_list:
#     pref = tuple(pref)
#     max_score = -np.inf
#     max_traj = None
#     for k in archive[pref].keys():
#         if archive[pref][k].score > max_score:
#             max_score = archive[pref][k].score
#             max_traj = archive[pref][k].cell_traj
#     if not pref == (0, 1):
#         pref_traj_score[pref] = (max_traj, max_score)
# print(pref_traj_score)
other_simulator = DeepSeaTreasure()
print(len(pref_list))
# print(pref_list)
pref_list = np.array(
    [[0.01, 0.99], [0.65, 0.35]
        , [0.75, 0.25], [0.82, 0.18], [0.84, 0.16], [0.87, 0.13], [0.89, 0.11], [0.91, 0.09],
     [0.93, 0.07], [0.99, 0.01]]
)
print(pref_traj_score)
pref_list = np.array(pref_list[::-1])
agent_list = [None, ]
# for pref_index in range(0, len(pref_list)):
#     agent = DQNAgent(simulator, model_path="../Agent/AgentModel")
#     agent_list.append(agent)
#     # for pref_index in range(1, 2):
#     pref = tuple(pref_list[pref_index])
#     # traj = pref_traj_score[pref][0]
#     # cumulative_rewards = traj_cost_calculate(traj[1:], other_simulator)
#
#     print(cross)
#     print(f"pref:{pref}\n"
#           # f"demonstration traj:{traj}\n"
#           # f"reward_bar:{np.dot(cumulative_rewards, pref)}\t"
#           # f"cumulative_rewards:{cumulative_rewards}"
#           )
#
#     agent.train_model_with_traj(reward_bar=None, pref=np.array(pref), traj=None, save_per=20000)
#     agent.generate_experience(pref=pref)
# result_dict = {}
for pref_index in range(0, len(pref_list)):
    agent = DQNAgent(simulator, model_path="../Agent/AgentModel")
    # agent.model = keras.models.load_model(agent.model_path + str(pref_list[pref_index]))
    pref = tuple(pref_list[pref_index])
    if pref_index == len(pref_list) - 1:
        # agent = DQNAgent(simulator, model_path="../Agent/AgentModel")
        # traj = [3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        print(f"extra training for pref:{pref}")
        agent.train_model_with_traj(episodes=200000, reward_bar=None, pref=np.array(pref), traj=None, save_per=20000)
    agent.generate_experience(pref=pref)
action_list = [3, 1, 3, 1, 3, 1, 3, 3, 3, 1, 1, 1, 3, 3, 1, 3, 1, 1, 1]
agent = DQNAgent(simulator, model_path="../Agent/AgentModel")
agent.model = keras.models.load_model(
    "C:/Users/19233436/PycharmProjects/MOGOExplore/Algorithm/Agent/AgentModel[0.01 0.99]")
state, pos = simulator.reset()
print(f"pos:{pos}")
agent.show_Q(state)
print(f"-------------------------")
for action in action_list:
    rewards, image, terminal, position,shaped_reward = simulator.step(action=action)
    print(f"pos:{position} | rewards:{rewards} | shaped_reward:{shaped_reward}")
    agent.show_Q(image)
    print(f"-------------------------")
