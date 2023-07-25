import copy
import random
from util.archives import DeterministicArchive
from util.dataclass import CellInfo
import numpy as np
from simulators.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure
from simulators.deep_sea_treasure.preference_space import PreferenceSpace
from decimal import Decimal

cross = "----------------------------------"


def traj_cost_calculate(traj, simulator):
    simulator.reset()
    cumulative_rewards = np.zeros(2)
    # print(f"cumulative_rewards:{cumulative_rewards}")
    for pos in traj:
        reward = simulator.calculate_reward(pos)
        cumulative_rewards += reward

    return cumulative_rewards


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
    # print(f"pref_list:{pref_list}")
    deterministic_archive = DeterministicArchive()

    for pref in pref_list:
        print(f"pref list:{pref}")
        pref = tuple(pref)
        print(f"pref:{pref}")
        init_pos = (0, 0)
        cell_key = init_pos  # row, col, treasure, step
        initial_cell_info = CellInfo(cell_traj=[init_pos], num_of_visit=1, score=-np.inf,
                                     terminal=False)  # cell key: task, time
        deterministic_archive.update_cell(utility_key=pref, cell_key=cell_key, cell_info=initial_cell_info)
    print(deterministic_archive.archive.items())

    for pref in pref_list[1:]:
        pref = tuple(pref)
        print(f"Exploration of pref:{pref} starts ---------------------->> ")
        for sample_epi in range(2001):
            prob = calc_sample_prob(deterministic_archive.archive[pref])
            cell_key = random.choices(list(deterministic_archive.archive[pref].keys()), prob, k=1)[0]
            if simulator.calculate_reward(cell_key)[1] >= 0.7:
                terminal = True
            else:
                terminal = False
            simulator.reset_to_state(reset_to=cell_key)

            while not terminal:
                trajectory = deterministic_archive.archive[pref][cell_key].cell_traj[:]
                action = random.randint(0, 3)
                rewards, image, terminal, position, _, _ = simulator.step(action=action)

                cell_key = position
                trajectory.append(cell_key)
                cumulative_rewards = traj_cost_calculate(trajectory[1:], simulator_)
                scalar_reward = np.dot(cumulative_rewards, np.array(pref))
                cell_info = CellInfo(cell_traj=trajectory[:], num_of_visit=1, score=scalar_reward,
                                     reward_vec=cumulative_rewards, terminal=terminal)
                deterministic_archive.update_cell(utility_key=pref, cell_key=cell_key, cell_info=cell_info)

        for cell in deterministic_archive.archive[pref].keys():
            print(f"pref:{pref}\t"
                  f"cell:{cell}"
                  f"{deterministic_archive.archive[pref][cell]}")

    for pref in sorted(deterministic_archive.archive.keys()):
        scalar_reward_list = []
        for cell in deterministic_archive.archive[pref].keys():
            scalar_reward_list.append(deterministic_archive.archive[pref][cell].score)
        scalar_reward_list = np.array(scalar_reward_list)
        max_index = np.argmax(scalar_reward_list)
        print(
            f"pref:{pref}\tcell:{list(deterministic_archive.archive[pref].keys())[max_index]}\treward:{scalar_reward_list[max_index]}\n{cross}")

    np.save("C:/Users/19233436/PycharmProjects/MOGOExplore/simulation/deep_sea_treasure/archive/archive.npy",
            deterministic_archive.archive)
