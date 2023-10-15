import random
import numpy as np


class PreferenceSpace(object):
    def __init__(self, granularity=100):
        self.granularity = granularity

    def iterate(self):
        preference_list = []
        pref = []
        upper_bound = self.granularity + 1
        for p in range(1, upper_bound):
            for q in range(1, self.granularity - p):
                p_ = p / self.granularity
                q_ = q / self.granularity
                last_p = 1 - p_ - q_

                pref.append(self.round_to(p_))
                pref.append(self.round_to(q_))
                pref.append(last_p)

                pref = [round(num, 2) for num in pref]
                preference_list.append(pref)
                pref = []
        preference = np.array(preference_list)
        return preference
    def round_to(self, v):
        return int(v * 100) / 100.0


if __name__ == '__main__':
    preference_space = PreferenceSpace()
    print(preference_space.iterate())
    for p in preference_space.iterate():
        print(f"p:{p}")

    print(len(preference_space.iterate()))
