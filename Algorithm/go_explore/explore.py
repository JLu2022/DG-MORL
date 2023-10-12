import copy
import random
from util.archives import DeterministicArchive
from util.dataclass import CellInfo
import numpy as np
from simulators.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure
from simulators.deep_sea_treasure.preference_space import PreferenceSpace
from decimal import Decimal

cross = "----------------------------------"
GAMMA = 0.99


def traj_utility_calculate(traj, simulator, pref_w):
    gamma = 1
    simulator.reset()
    simulator.calculate_utility(demo=traj, pref_w=pref_w)
    vec_return = np.zeros(2)
    disc_vec_rew = np.zeros(2)
    # print(f"cumulative_rewards:{cumulative_rewards}")
    for pos in traj:
        vec_reward = simulator.calculate_reward(pos)
        disc_vec_rew += vec_reward * gamma
        vec_return += vec_reward * gamma
        gamma *= GAMMA
    utility = np.dot(disc_vec_rew, pref_w)

    return vec_return, utility


def calc_sample_prob(archive):
    nums_of_visit = 0
    probs = []

    for key in list(archive.keys()):
        cell = archive[key]
        nums_of_visit += cell.num_of_visit
        probs.append(1 / (cell.num_of_visit))
    probs = np.array(probs)
    probs = probs / np.sum(probs)
    return probs


if __name__ == '__main__':
    simulator = DeepSeaTreasure()
    simulator_ = copy.deepcopy(simulator)
    pref_space = PreferenceSpace()
    pref_list = pref_space.iterate()
    deterministic_archive = DeterministicArchive()

    for pref in pref_list:
        print(f"pref list:{pref}")
        pref = tuple(pref)
        print(f"pref:{pref}")
        init_pos = (0, 0)
        cell_key = init_pos  # row, col, treasure, step
        initial_cell_info = CellInfo(cell_traj=[init_pos],
                                     num_of_visit=1,
                                     score=-np.inf,
                                     terminal=False)  # cell key: task, time

        deterministic_archive.update_cell(utility_key=pref, cell_key=cell_key, cell_info=initial_cell_info)
    print(deterministic_archive.archive.items())

    for pref_w in pref_list:
        pref_w = tuple(pref_w)
        print(f"Exploration of pref:{pref_w} starts ---------------------->> ")
        for sample_epi in range(5000):
            prob = calc_sample_prob(deterministic_archive.archive[pref_w])
            cell_key = random.choices(list(deterministic_archive.archive[pref_w].keys()), prob, k=1)[0]
            if simulator.calculate_reward(cell_key)[1] >= 0.5:
                terminal = True
            else:
                terminal = False
            simulator.reset_to_state(reset_to=cell_key)

            while not terminal:
                trajectory = deterministic_archive.archive[pref_w][cell_key].cell_traj[:]
                action = random.randint(0, 3)
                rewards, image, terminal, position, _, _ = simulator.step(action=action)

                cell_key = position
                trajectory.append(cell_key)
                utility, cumulative_rewards = simulator.calculate_utility(demo=trajectory, pref_w=pref_w)
                cell_info = CellInfo(cell_traj=trajectory[:], num_of_visit=1, score=float(utility),
                                     reward_vec=list(cumulative_rewards), terminal=terminal)
                deterministic_archive.update_cell(utility_key=pref_w, cell_key=cell_key, cell_info=cell_info)

    for pref in sorted(deterministic_archive.archive.keys()):
        scalar_reward_list = []
        for cell in deterministic_archive.archive[pref].keys():
            scalar_reward_list.append(deterministic_archive.archive[pref][cell].score)
        scalar_reward_list = np.array(scalar_reward_list)
        max_index = np.argmax(scalar_reward_list)
        print(
            f"pref:{pref}\tcell:{list(deterministic_archive.archive[pref].keys())[max_index]}\treward:{scalar_reward_list[max_index]}\n{cross}")

    np.save("archives/DST/archive.npy", deterministic_archive.archive)
