# import gymnasium as gym
# import mo_gymnasium as mo_gym
# import numpy as np
# import random
# # It follows the original Gymnasium API ...
# env = mo_gym.make('deep-sea-treasure-v0')
#
# obs, info = env.reset()
# # but vector_reward is a numpy array!
# terminated = False
# ACTIONS = ["up", "down", "left", "right"]
# while not terminated:
#     action = random.randint(0,3)
#     next_obs, vector_reward, terminated, truncated, info = env.step(action)
#     print(f"action:{ACTIONS[action]}\n"
#           f"next_obs:{next_obs}\n"
#           f"vector_reward:{vector_reward}\n"
#           f"terminated:{terminated}\n"
#           f"truncated:{truncated}\n"
#           f"info:{info}")
# # Optionally, you can scalarize the reward function with the LinearReward wrapper
# # env = mo_gym.LinearReward(env, weight=np.array([0.8, 0.2, 0.2]))
from Algorithm.go_explore.explore import traj_utility_calculate
from simulators.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure
import numpy as np
simulator = DeepSeaTreasure()
pref = np.array([0.1,0.9])
traj = [[9, 9], [10, 9]]
cumulative_rewards = traj_utility_calculate(traj[1:], simulator)
print(f"cumulative_rewards:{cumulative_rewards},\t{np.dot(cumulative_rewards, pref)}")
