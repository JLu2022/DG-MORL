import numpy as np
from minecart_simulator import Minecart

human_demo = np.load("../../train/minecart/traj/human_traj_utility_dict.npy", allow_pickle=True).item()
print(human_demo[(0.34, 0.36, 0.3)])
env = Minecart(render_mode="human", image_observation=True)
env.render()
demo = (2, 1, 3, 3, 3, 5, 5, 4, 0, 0, 1, 1, 1, 1, 1, 3, 3, 3, 2, 2, 1, 3)
w = np.array((0.38, 0.25, 0.37))
u, _ = env.calculate_utility(demo=demo, pref_w=w)
print(f"u:{u}")
