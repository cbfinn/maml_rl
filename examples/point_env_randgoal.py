from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import numpy as np


class PointEnvRandGoal(Env):
    def __init__(self, goal=None):  # Can set goal to test adaptation.
        self._goal = goal

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(2,))

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(2,))

    def sample_goals(self, num_goals):
        return np.random.uniform(-0.5, 0.5, size=(num_goals, 2, ))

    def reset(self, reset_args=None):
        goal = reset_args
        if goal is not None:
            self._goal = goal
        elif self._goal is None:
            # Only set a new goal if this env hasn't had one defined before.
            self._goal = np.random.uniform(-0.5, 0.5, size=(2,))
            #goals = [np.array([-0.5,0]), np.array([0.5,0])]
            #goals = np.array([[-0.5,0], [0.5,0],[0.2,0.2],[-0.2,-0.2],[0.5,0.5],[0,0.5],[0,-0.5],[-0.5,-0.5],[0.5,-0.5],[-0.5,0.5]])
            #self._goal = goals[np.random.randint(10)]

        self._state = (0, 0)
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        self._state = self._state + action
        x, y = self._state
        x -= self._goal[0]
        y -= self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        done = abs(x) < 0.01 and abs(y) < 0.01
        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=done, goal=self._goal)

    def render(self):
        print('current state:', self._state)
