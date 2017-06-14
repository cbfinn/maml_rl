from .mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
import numpy as np

from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger


# going to a goal position in 2D
class AntEnvRandGoalOracle(MujocoEnv, Serializable):

    FILE = 'ant.xml'

    def __init__(self, goal= None, *args, **kwargs):
        self._goal_pos = goal
        super(AntEnvRandGoalOracle, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def get_current_obs(self):
        obs = np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            self.get_body_xmat("torso").flat,
            self.get_body_com("torso"),
        ]).reshape(-1)
        return np.r_[obs, self._goal_pos]

    def sample_goals(self, num_goals):
        return np.random.uniform(-3.0, 3.0, (num_goals, 2, ))

    @overrides
    def reset(self, init_state=None, reset_args=None, **kwargs):
        goal_pos = reset_args
        if goal_pos is not None:
            self._goal_pos = goal_pos
        elif self._goal_pos is None:
            #self._goal_pos = np.random.uniform(0.1, 0.8)
            self._goal_pos = np.random.uniform(-3.0, 3.0, (2,))
        self.reset_mujoco(init_state)
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        obs = self.get_current_obs()
        return obs


    def step(self, action):
        self.forward_dynamics(action)
        com = self.get_body_com("torso")
        # ref_x = x + self._init_torso_x
        goal_reward = -np.sum(np.abs(com[:2] - self._goal_pos)) + 4.0 # make it happy, not suicidal
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
        survive_reward = 0.05
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self._state
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self.get_current_obs()
        return Step(ob, float(reward), done)

    @overrides
    def log_diagnostics(self, paths, prefix=''):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular(prefix+'AverageForwardProgress', np.mean(progs))
        logger.record_tabular(prefix+'MaxForwardProgress', np.max(progs))
        logger.record_tabular(prefix+'MinForwardProgress', np.min(progs))
        logger.record_tabular(prefix+'StdForwardProgress', np.std(progs))

