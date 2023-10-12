import numpy as np
from simulators.deep_sea_treasure.preference_space import PreferenceSpace

ACTIONS = {0: "up", 1: "down", 2: "left", 3: "right"}


def coord_to_pos(size, coords):
    pos = coords[0] * size[0] + coords[1]
    return pos


def pos_to_coord(size, pos):
    return pos // size[0], pos % size[1]


def day_to_five_min(num_days):
    return num_days * 1440 // 5


def find_best_traj(archive):
    pref_space = PreferenceSpace()
    pref_traj_score = {}
    pref_traj_rews = {}
    raw_pref_list = pref_space.iterate()
    for pref in raw_pref_list:
        pref = tuple(pref)
        max_score = -np.inf
        max_traj = None
        for k in archive[pref].keys():
            if archive[pref][k].score > max_score and archive[pref][k].terminal:
                max_rews = archive[pref][k].reward_vec
                max_score = archive[pref][k].score
                max_traj = archive[pref][k].cell_traj
            # if 0.5 < pref[1] < 0.62:
            #     print(f"pref:{pref}\tarchive[pref].keys():{archive[pref][k]}")
        # if not pref == (0, 1):
        pref_traj_score[pref] = (max_traj, max_score)
        pref_traj_rews[pref] = tuple(max_rews)

    preference_list = []
    rew_vec_list = []

    for pref, rews in pref_traj_rews.items():  # treasure, step
        preference_list.append(np.array(pref))
        rew_vec_list.append(np.array([rews[0], rews[1]]))
        # print(f"pref:{pref}|rews:{rews}|utility:{np.dot(rews, pref)}")
    return pref_traj_score, pref_traj_rews, rew_vec_list, preference_list


if __name__ == '__main__':
    # archive = np.load("../Algorithm/go_explore/archives/archive.npy", allow_pickle=True).item()
    # archive = dict(archive)
    # find_best_traj(pref_space=PreferenceSpace(), archive=archive)
    print(coord_to_pos((5, 5), (3, 3)))
    print(pos_to_coord((5, 5), coord_to_pos((5, 5), (3, 3))))
