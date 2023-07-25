import random
import numpy as np


class PreferenceSpace(object):
    def __init__(self, num_objective=2, granularity=100):
        self.num_objective = num_objective
        self.granularity = granularity
        # self.default_pref = default_pref

    def sample(self, default_pref=None):
        pref = []
        upper_bound = self.granularity + 1
        for _ in range(self.num_objective - 1):
            p = random.choice([x for x in range(0, upper_bound)])
            pref.append(p / self.granularity)
            upper_bound = self.granularity - p
        last_p = 1 - sum(pref)
        pref.append(last_p)

        preference = np.array(default_pref) if default_pref is not None else np.array(pref)
        return preference

    def iterate(self):
        preference_list = []
        pref = []
        upper_bound = self.granularity + 1
        for p in range(upper_bound):
            p = p / self.granularity
            pref.append(self.round_to(p))
            last_p = 1 - p
            pref.append(last_p)
            pref = [round(num, 2) for num in pref]
            preference_list.append(pref)
            pref = []

        preference = np.array(preference_list)
        return preference

    def round_to(self, v, dig=2):
        return int(v * 100) / 100.0
        # return v


if __name__ == '__main__':
    preference_space = PreferenceSpace()
    print(preference_space.iterate())
    # for _ in range(100):
    #     print(preference_space.sample())
