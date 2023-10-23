import numpy as np
# from simulators.minecart.minecart_simulator import Minecart
human_demo = np.load("traj/human_traj_utility_dict.npy", allow_pickle=True).item()
human_demo = dict(human_demo)
for k, v in human_demo.items():
    print(f"w:{k}\tv:{v[0]}\tmode:{v[1]}\tactions:{v[2]}")
behaviour_modes = ["mean_agent", "ore_1_agent", "ore_2_agent", "quick_ore_1_agent", "quick_ore_2_agent",
                   "balance_agent", "quick_balance_agent"]
evaluation_dict = {}

for behaviour_mode in behaviour_modes:
    evaluation_dict[behaviour_mode] = {"utility": [], "w": []}

for k, v in human_demo.items():
    evaluation_dict[v[1]]["w"].append(k)
    evaluation_dict[v[1]]["utility"].append(v[0])
u = 0
counter = 0
for behaviour_mode, v in evaluation_dict.items():
    if evaluation_dict[behaviour_mode]["utility"]:
        counter += 1
        idx = np.argmax(evaluation_dict[behaviour_mode]["utility"])
        print(f"behaviour_mode:{behaviour_mode}\t"
              f"max u:{evaluation_dict[behaviour_mode]['utility'][idx]}\t"
              f"w:{evaluation_dict[behaviour_mode]['w'][idx]}")
        u += evaluation_dict[behaviour_mode]['utility'][idx]
print(u / counter)
# mine_cart = Minecart()
print(human_demo)