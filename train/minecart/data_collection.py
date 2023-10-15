import copy
import math
import os
import random
from util.archives import DeterministicArchive
from util.dataclass import CellInfo
from util.utils import show_archive
import numpy as np
from simulators.minecart.minecart_simulator import Minecart
from simulators.minecart.preference_space import PreferenceSpace


def calc_sample_prob(archive):
    nums_of_visit = 0
    probs = []

    for key in list(archive.keys()):
        cell = archive[key]
        nums_of_visit += cell.num_of_visit
        probs.append(1 / cell.num_of_visit)
    probs = np.array(probs)
    probs = probs / np.sum(probs)
    return probs


if __name__ == '__main__':
    """
    @Minecart:
    State space - {x_pos, y_pos, cart_speed, sin_angle, cos_angle, ore1/capacity, ore2/capacity}
    Action space - {
    - 0: Mine
    - 1: Left
    - 2: Right
    - 3: Accelerate
    - 4: Brake
    - 5: None
    }
    Reward Space: 
    The reward is a 3D vector:
    - 0: Quantity of the first minerium that was retrieved to the base (sparse)
    - 1: Quantity of the second minerium that was retrieved to the base (sparse)
    - 2: Fuel consumed (dense)
    """
    simulator = Minecart()
    simulator_ = copy.deepcopy(simulator)
    pref_space = PreferenceSpace()
    pref_list = pref_space.iterate()
    # pref_list = [[0.8, 0.1, 0.1]]
    deterministic_archive = DeterministicArchive()
    pref_w_tuples = []
    explore_episodes = 10

    for pref_w in pref_list:
        pref_w = tuple(pref_w)

        pref_w_tuples.append(pref_w)
        # print(f"pref weight vector:{pref_w}")
        init_state = (0., 0., 0., 45., 0., 0.)
        cell_key = init_state
        initial_cell_info = CellInfo(cell_traj=[], num_of_visit=1, score=-np.inf, terminal=False)
        deterministic_archive.update_cell(utility_key=pref_w, cell_key=cell_key, cell_info=initial_cell_info)
    # print(deterministic_archive.archive.items())
    show_archive(deterministic_archive.archive)

    for pref_w in pref_w_tuples:
        print(f"Exploration of pref:{pref_w} starts ---------------------->> ")
        for episode in range(explore_episodes):
            prob = calc_sample_prob(deterministic_archive.archive[pref_w])
            terminated_state = True
            while terminated_state:
                cell_key = random.choices(list(deterministic_archive.archive[pref_w].keys()), prob, k=1)[0]
                simulator.reset_to_state(reset_to=cell_key)
                terminated_state = deterministic_archive.archive[pref_w][cell_key].terminal
            print(f"reset to:{cell_key}")
            terminal = False
            while not terminal:
                trajectory = deterministic_archive.archive[pref_w][cell_key].cell_traj[:]
                action = random.randint(0, 5)
                trajectory.append(action)
                state, reward, terminal, _, _ = simulator.step(action=action)

                s_sin = state[3]
                s_cos = state[4]
                angle = round(math.degrees(math.asin(s_sin)))

                cell_key = (round(state[0], 2), round(state[1], 2), round(state[2], 2), angle, round(state[5], 2),
                            round(state[6], 2))
                if terminal:
                    cell_key = (2., 2., 2., 100., 2., 2.)

                utility, cumulative_rewards = simulator_.calculate_utility(demo=trajectory, pref_w=pref_w)
                cell_info = CellInfo(cell_traj=trajectory[:], num_of_visit=1, score=float(utility),
                                     reward_vec=list(cumulative_rewards), terminal=terminal)

                deterministic_archive.update_cell(utility_key=pref_w, cell_key=cell_key, cell_info=cell_info)
                if terminal:
                    print(
                        f"weight vector:{pref_w}\n"
                        f"cell key:{cell_key}\n"
                        f"traj:{trajectory}\n"
                        f"utility:{utility}\n"
                        f"cumulative rewards:{cumulative_rewards}\n"
                        f"===========================================")
                    if not os.path.exists(f"traj/{pref_w}"):
                        os.makedirs(f"traj/{pref_w}")
                    np.save(f"traj/{pref_w}/prior_policy", trajectory)

        for pref in sorted(deterministic_archive.archive.keys()):
            utility_list = []
            for cell in deterministic_archive.archive[pref].keys():
                utility_list.append(deterministic_archive.archive[pref][cell].score)
            utility_list = np.array(utility_list)
            max_index = np.argmax(utility_list)
            print(f"pref:{pref}\tcell:{list(deterministic_archive.archive[pref].keys())[max_index]}\treward:{utility_list[max_index]}\n")

        np.save("archives/archive.npy", deterministic_archive.archive)
