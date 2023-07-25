import warnings as _warnings
import copy
from typing import List, Any
from collections import deque
import sys

try:
    from dataclasses import dataclass, field as datafield


    def copyfield(data):
        return datafield(default_factory=lambda: copy.deepcopy(data))
except ModuleNotFoundError:
    _warnings.warn('dataclasses not found. To get it, use Python 3.7 or pip install dataclasses')


@dataclass
class CellInfo:
    cell_traj: List[Any] = copyfield([])
    pos_traj: List[Any] = copyfield([])
    # trajectory_len: int = float('inf')
    terminal: bool = False
    num_of_visit: int = 0
    score: float = float('inf')
    reward_vec: List[Any]= copyfield([])

@dataclass
class CellInfoDeterministic:
    #: The score of the last accepted trajectory to this cell
    score: int = -float('inf')

    #: Number of trajectories that included this cell
    nb_seen: int = 0

    #: The number of times this cell was chosen as the cell to explore from
    nb_chosen: int = 0

    #: The number of times this cell was chosen since it was last updated
    nb_chosen_since_update: int = 0

    #: The number of times this cell was chosen since it last resulted in discovering a new cell
    nb_chosen_since_to_new: int = 0

    #: The number of times this cell was chosen since it last resulted in updating any cell
    nb_chosen_since_to_update: int = 0

    #: The number of actions that had this cell as the resulting state (i.e. all frames spend in this cell)
    nb_actions: int = 0

    #: The number of times this cell was chosen to explore towards
    nb_chosen_for_exploration: int = 0

    #: The number of times this cell was reached when chosen to explore towards
    nb_reached_for_exploration: int = 0

    #: Length of the trajectory
    trajectory_len: int = float('inf')

    #: Saved restore state. In a purely deterministic environment,
    #: this allows us to fast-forward to the end state instead
    #: of replaying.
    restore: Any = None

    #: Sliding window for calculating our success rate of reaching different cells
    reached: deque = copyfield(deque(maxlen=100))

    #: List of cells that we went through to reach this cell
    cell_traj: List[Any] = copyfield([])
    exact_pos = None
    real_cell = None
    traj_last = None
    real_traj: List[int] = None

    @property
    def nb_reached(self):
        return sum(self.reached)
