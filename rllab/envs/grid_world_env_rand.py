import numpy as np
from .base import Env
from rllab.spaces import Discrete
from rllab.envs.base import Step
from rllab.core.serializable import Serializable

# simple two state world would look like {"chain":['GSH'],"chain":['HSG']}
MAPS = {
    "two-state": [['GSH'],['HSG']],
    "four-state": [['HHH','GSH','HHH'],['HHH','HSG','HHH'],['HGH','HSH','HHH'], ['HHH','HSH','HGH']]
}


class GridWorldEnvRand(Env, Serializable):
    """
    'S' : starting point
    'F': free space
    'W': wall
    'H': hole (terminates episode)
    'G': goal (terminates episode)

    """
    def __init__(self, desc='two-state', map_id=None):
        self._map_id = map_id
        Serializable.quick_init(self, locals())
        if isinstance(desc, str):
            desc = MAPS[desc]
        self.desc_choices = desc
        self.reset()

    def reset(self, reset_args=None):
        map_id = reset_args
        if map_id is not None:
            self._map_id = map_id
        elif self._map_id is None:
            self._map_id = np.random.randint(len(self.desc_choices))

        self.desc = np.array(list(map(list, self.desc_choices[self._map_id])))
        self.n_row, self.n_col = self.desc.shape
        (start_x,), (start_y,) = np.nonzero(self.desc == 'S')
        self.start_state = start_x * self.n_col + start_y
        self.domain_fig = None

        self.state = self.start_state
        return self.state

    def sample_goals(self, num_goals):
        return np.random.randint(4, size=(num_goals,))

    @staticmethod
    def action_from_direction(d):
        """
        Return the action corresponding to the given direction. This is a helper method for debugging and testing
        purposes.
        :return: the action index corresponding to the given direction
        """
        return dict(
            left=0,
            down=1,
            right=2,
            up=3
        )[d]

    def step(self, action):
        """
        action map:
        0: left
        1: down
        2: right
        3: up
        :param action: should be a one-hot vector encoding the action
        :return:
        """
        possible_next_states = self.get_possible_next_states(self.state, action)

        probs = [x[1] for x in possible_next_states]
        next_state_idx = np.random.choice(len(probs), p=probs)
        next_state = possible_next_states[next_state_idx][0]

        next_x = next_state // self.n_col
        next_y = next_state % self.n_col

        next_state_type = self.desc[next_x, next_y]
        if next_state_type == 'H':
            done = True
            reward = 0
        elif next_state_type in ['F', 'S']:
            done = False
            reward = 0
        elif next_state_type == 'G':
            done = True
            reward = 1
        else:
            raise NotImplementedError
        self.state = next_state
        return Step(observation=self.state, reward=reward, done=done)

    def get_possible_next_states(self, state, action):
        """
        Given the state and action, return a list of possible next states and their probabilities. Only next states
        with nonzero probabilities will be returned
        :param state: start state
        :param action: action
        :return: a list of pairs (s', p(s'|s,a))
        """
        # assert self.observation_space.contains(state)
        # assert self.action_space.contains(action)

        x = state // self.n_col
        y = state % self.n_col
        coords = np.array([x, y])

        increments = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        next_coords = np.clip(
            coords + increments[action],
            [0, 0],
            [self.n_row - 1, self.n_col - 1]
        )
        next_state = next_coords[0] * self.n_col + next_coords[1]
        state_type = self.desc[x, y]
        next_state_type = self.desc[next_coords[0], next_coords[1]]
        if next_state_type == 'W' or state_type == 'H' or state_type == 'G':
            return [(state, 1.)]
        else:
            return [(next_state, 1.)]

    @property
    def action_space(self):
        return Discrete(4)

    @property
    def observation_space(self):
        return Discrete(self.n_row * self.n_col)

