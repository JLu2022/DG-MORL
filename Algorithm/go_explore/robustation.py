import keras
import numpy as np
# from rl_algorithm.tabular_q import Tabular_Q_Agent
from Algorithm.rl_algorithm.PPO.ppo import PPO
from simulators.discrete_grid_world import ImageGridWorld
from util.utils import ACTIONS
from simulators.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure
from simulators.deep_sea_treasure.preference_space import PreferenceSpace
from explore import traj_cost_calculate

terminal_state = (0, 24)

archive = np.load("C:/Users/19233436/PycharmProjects/MOGOExplore/simulation/deep_sea_treasure/archive/archive.npy",
                  allow_pickle=True).item()
simulator = DeepSeaTreasure()
cross = "---------------backwards-------------------"
pref_space = PreferenceSpace()
pref_list = pref_space.iterate()

# agent = PPO(simulator, h_layers=[32, 32, 32], seed=0, steps_per_epoch=500, epochs=200, gamma=1, lam=0.95,
#             clip_ratio=0.2, lr_a=1e-3, lr_c=1e-3, train_a_iters=80, train_c_iters=80,
#             max_ep_len=240, kl_target=0.01, ent_weight=0.005, save_freq=10, save_path='./checkpoints/',
#             mode="discrete")
pref_traj_score = {}
for pref in pref_list:
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
print(len(pref_list))
print(pref_list)
pref_list = np.array(
    [[], [0.1, 0.9], [0.65, 0.35]
        , [0.75, 0.25], [0.82, 0.18], [0.85, 0.15], [0.87, 0.13], [0.89, 0.11], [0.91, 0.09],
     [0.93, 0.07], [0.99, 0.01]]
)
agent_list = [None, ]
for pref_index in range(1, len(pref_list)):
    agent = PPO(simulator, h_layers=[32, 32, 32], seed=0, steps_per_epoch=500, epochs=200, gamma=1, lam=0.95,
                clip_ratio=0.2, lr_a=1e-3, lr_c=1e-3, train_a_iters=80, train_c_iters=80,
                max_ep_len=240, kl_target=0.01, ent_weight=0.01, save_freq=10, save_path='./checkpoints/',
                mode="discrete")
    agent_list.append(agent)
    # for pref_index in range(1, 2):
    pref = tuple(pref_list[pref_index])
    for i in range(2, len(pref_traj_score[pref][0][:]) + 1):
        init_traj = pref_traj_score[pref][0][-i:]
        # print(init_traj)
        cumulative_rewards = traj_cost_calculate(init_traj[1:], other_simulator)
        print(cross)
        print(f"pref:{pref}\n"
              f"demonstration traj:{init_traj}\n"
              f"reward_bar:{np.dot(cumulative_rewards, pref)}\t"
              f"cumulative_rewards:{cumulative_rewards}")

        agent.robustification_train_(reward_bar=np.dot(cumulative_rewards, pref), pref=np.array(pref),
                                     reset_to=init_traj[0])

        agent.actor.save(agent.save_path + 'actor_checkpoint' + str(pref_index) + str(pref_index))
        agent.critic.save(agent.save_path + 'critic_checkpoint' + str(pref_index) + str(pref_index))
result_dict = {}
for pref_index in range(1, len(pref_list)):
    agent = agent_list[pref_index]
    pref = tuple(pref_list[pref_index])
    agent.actor.compile(optimizer='adam', loss='mse')
    agent.critic.compile(optimizer='adam', loss='mse')
    agent.actor = keras.models.load_model(agent.save_path + 'actor_checkpoint' + str(pref_index) + str(pref_index))
    agent.critic = keras.models.load_model(agent.save_path + 'critic_checkpoint' + str(pref_index) + str(pref_index))
    for _ in range(5):
        episode_reward, state_list, pref = agent.generate_experience(reset_to=(0, 0), pref=pref)
        result_dict[tuple(pref)] = state_list

for k in result_dict.keys():
    print(f"pref:{k} -> traj:{result_dict[k]}")
