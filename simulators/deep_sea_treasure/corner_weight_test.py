from typing import List

import numpy as np
from deep_sea_treasure import DeepSeaTreasure
from preference_space import PreferenceSpace
import cdd


def compute_corner_weights(CCS) -> List[np.ndarray]:
    A = np.vstack(CCS)
    A = np.round_(A, decimals=4)
    A = np.concatenate((A, -np.ones(A.shape[0]).reshape(-1, 1)), axis=1)
    A_plus = np.ones(A.shape[1]).reshape(1, -1)
    A_plus[0, -1] = 0
    A = np.concatenate((A, A_plus), axis=0)

    A_plus = -np.ones(A.shape[1]).reshape(1, -1)
    A_plus[0, -1] = 0
    A = np.concatenate((A, A_plus), axis=0)

    for i in range(2):
        A_plus = np.zeros(A.shape[1]).reshape(1, -1)
        A_plus[0, i] = -1
        A = np.concatenate((A, A_plus), axis=0)

    b = np.zeros(len(CCS) + 2 + 2)
    b[len(CCS)] = 1
    b[len(CCS) + 1] = -1
    vertices = compute_poly_vertices(A, b)
    corners = []
    for v in vertices:
        corners.append(v[:-1])
    return corners


def compute_poly_vertices(A, b):
    # Based on https://stackoverflow.com/questions/65343771/solve-linear-inequalities
    b = b.reshape((b.shape[0], 1))
    mat = cdd.Matrix(np.hstack([b, -A]), number_type="float")
    mat.rep_type = cdd.RepType.INEQUALITY
    P = cdd.Polyhedron(mat)
    g = P.get_generators()
    V = np.array(g)
    vertices = []
    for i in range(V.shape[0]):
        if V[i, 0] != 1:
            continue
        if i not in g.lin_set:
            vertices.append(V[i, 1:])
    return vertices


if __name__ == '__main__':
    DST = DeepSeaTreasure()
    pref_space = PreferenceSpace()
    pref_ws = pref_space.iterate()
    action_demo_1 = [1]  # 0.7
    action_demo_2 = [3, 1, 1]  # 8.2
    action_demo_3 = [3, 3, 1, 1, 1]  # 11.5
    action_demo_4 = [3, 3, 3, 1, 1, 1, 1]  # 14.0
    action_demo_5 = [3, 3, 3, 3, 1, 1, 1, 1]  # 15.1
    action_demo_6 = [3, 3, 3, 3, 3, 1, 1, 1, 1]  # 16.1
    action_demo_7 = [3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1]  # 19.6
    action_demo_8 = [3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1]  # 20.3
    action_demo_9 = [3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 22.4
    action_demo_10 = [3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 23.7
    action_demos = [action_demo_1, action_demo_2, action_demo_3, action_demo_4, action_demo_5, action_demo_6,
                    action_demo_7, action_demo_8, action_demo_9, action_demo_10]
    # action_demos = [action_demo_1, action_demo_10]
    CCS = []
    # --  For test the demos -- #
    for action_demo in action_demos:
        value_scalar, value_vec = DST.calculate_utility_from_actions(action_demo=action_demo,
                                                                     pref_w=np.array([1, 0]))
        CCS.append(value_vec)
        print(f"utility:{value_scalar}\t value_vec:{np.round_(value_vec,2)}")
    # for pref_w in pref_ws:
    #     value_scalar, value_vec = DST.calculate_utility_from_actions(action_demo=action_demo_10,
    #                                                                  pref_w=np.array([1, 0]))
    #     print(f"utility:{value_scalar}\t value_vec:{value_vec}")
    # corner_weights = compute_corner_weights(CCS)
    # print(f"corner_weights:{corner_weights}")
