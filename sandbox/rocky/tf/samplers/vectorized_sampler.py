import pickle

import tensorflow as tf
from rllab.sampler.base import BaseSampler
from sandbox.rocky.tf.envs.parallel_vec_env_executor import ParallelVecEnvExecutor
from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor
from rllab.misc import tensor_utils
import numpy as np
from rllab.sampler.stateful_pool import ProgBarCounter
import rllab.misc.logger as logger
import itertools


class VectorizedSampler(BaseSampler):

    def __init__(self, algo, n_envs=None):
        super(VectorizedSampler, self).__init__(algo)
        self.n_envs = n_envs

    def start_worker(self):
        n_envs = self.n_envs
        if n_envs is None:
            n_envs = int(self.algo.batch_size / self.algo.max_path_length)
            n_envs = max(1, min(n_envs, 100))

        if getattr(self.algo.env, 'vectorized', False):
            self.vec_env = self.algo.env.vec_env_executor(n_envs=n_envs, max_path_length=self.algo.max_path_length)
        else:
            envs = [pickle.loads(pickle.dumps(self.algo.env)) for _ in range(n_envs)]
            self.vec_env = VecEnvExecutor(
                envs=envs,
                #env=pickle.loads(pickle.dumps(self.algo.env)),
                #n = n_envs,
                max_path_length=self.algo.max_path_length
            )
        self.env_spec = self.algo.env.spec

    def shutdown_worker(self):
        self.vec_env.terminate()


    def obtain_samples(self, itr, reset_args=None, return_dict=False, log_prefix=''):
        # reset_args: arguments to pass to the environments to reset
        # return_dict: whether or not to return a dictionary or list form of paths

        logger.log("Obtaining samples for iteration %d..." % itr)

        #paths = []
        paths = {}
        for i in range(self.vec_env.num_envs):
            paths[i] = []

        # if the reset args are not list/numpy, we set the same args for each env
        if reset_args is not None and (type(reset_args) != list and type(reset_args)!=np.ndarray):
            reset_args = [reset_args]*self.vec_env.num_envs

        n_samples = 0
        obses = self.vec_env.reset(reset_args)
        dones = np.asarray([True] * self.vec_env.num_envs)
        running_paths = [None] * self.vec_env.num_envs

        pbar = ProgBarCounter(self.algo.batch_size)
        policy_time = 0
        env_time = 0
        process_time = 0

        policy = self.algo.policy
        import time


        while n_samples < self.algo.batch_size:
            t = time.time()
            policy.reset(dones)
            actions, agent_infos = policy.get_actions(obses)

            policy_time += time.time() - t
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions, reset_args)
            env_time += time.time() - t

            t = time.time()

            agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            env_infos = tensor_utils.split_tensor_dict_list(env_infos)
            if env_infos is None:
                env_infos = [dict() for _ in range(self.vec_env.num_envs)]
            if agent_infos is None:
                agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos, agent_infos,
                                                                                    dones):
                if running_paths[idx] is None:
                    running_paths[idx] = dict(
                        observations=[],
                        actions=[],
                        rewards=[],
                        env_infos=[],
                        agent_infos=[],
                    )
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)
                if done:
                    paths[idx].append(dict(
                        observations=self.env_spec.observation_space.flatten_n(running_paths[idx]["observations"]),
                        actions=self.env_spec.action_space.flatten_n(running_paths[idx]["actions"]),
                        rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                        env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))
                    n_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = None
            process_time += time.time() - t
            pbar.inc(len(obses))
            obses = next_obses

        pbar.stop()

        logger.record_tabular(log_prefix+"PolicyExecTime", policy_time)
        logger.record_tabular(log_prefix+"EnvExecTime", env_time)
        logger.record_tabular(log_prefix+"ProcessExecTime", process_time)

        if not return_dict:
            flatten_list = lambda l: [item for sublist in l for item in sublist]
            paths = flatten_list(paths.values())
            #path_keys = flatten_list([[key]*len(paths[key]) for key in paths.keys()])

        return paths

"""
    def new_obtain_samples(self, itr, reset_args=None, return_dict=False):
        # reset_args: arguments to pass to the environments to reset
        # return_dict: whether or not to return a dictionary or list form of paths

        logger.log("Obtaining samples for iteration %d..." % itr)

        #paths = []
        paths = {}
        for i in range(self.algo.meta_batch_size):
            paths[i] = []

        n_samples = 0
        pbar = ProgBarCounter(self.algo.batch_size)
        policy_time = 0
        env_time = 0
        process_time = 0
        policy = self.algo.policy
        import time

        num_tasks = self.algo.meta_batch_size
        task_batch = self.vec_env.num_envs / num_tasks
        # inds 0 through task_batch are task 0, the next task_batch are task 1, etc.
        task_idxs = np.reshape(np.tile(np.arange(num_tasks), [task_batch,1]).T, [-1])
        obses = self.vec_env.reset([reset_args[idx] for idx in task_idxs])
        dones = np.asarray([True] * self.vec_env.num_envs)
        running_paths = [None] * self.vec_env.num_envs

        while n_samples <= self.algo.batch_size:

            t = time.time()
            policy.reset(dones)
            actions, agent_infos = policy.get_actions(obses)

            policy_time += time.time() - t
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            env_time += time.time() - t

            t = time.time()

            agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            env_infos = tensor_utils.split_tensor_dict_list(env_infos)

            if env_infos is None:
                env_infos = [dict() for _ in range(self.vec_env.num_envs)]
            if agent_infos is None:
                agent_infos = [dict() for _ in range(self.vec_env.num_envs)]

            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos, agent_infos,
                                                                                    dones):
                if running_paths[idx] is None:
                    running_paths[idx] = dict(
                        observations=[],
                        actions=[],
                        rewards=[],
                        env_infos=[],
                        agent_infos=[],
                        )
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)
                if done:
                    paths[task_idxs[idx]].append(dict(
                        observations=self.env_spec.observation_space.flatten_n(running_paths[idx]["observations"]),
                        actions=self.env_spec.action_space.flatten_n(running_paths[idx]["actions"]),
                        rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                        env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))
                    n_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = None
            process_time += time.time() - t
            pbar.inc(len(obses))
            obses = next_obses

        pbar.stop()

        logger.record_tabular("PolicyExecTime", policy_time)
        logger.record_tabular("EnvExecTime", env_time)
        logger.record_tabular("ProcessExecTime", process_time)

        if not return_dict:
            flatten_list = lambda l: [item for sublist in l for item in sublist]
            paths = flatten_list(paths.values())
            #path_keys = flatten_list([[key]*len(paths[key]) for key in paths.keys()])

        return paths
"""

