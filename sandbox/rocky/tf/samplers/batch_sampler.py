from rllab.misc import logger
from rllab.sampler.base import BaseSampler
from rllab.sampler import parallel_sampler
from rllab.sampler.stateful_pool import singleton_pool
from rllab.misc.tensor_utils import flatten_tensors
import tensorflow as tf
import numpy as np
import itertools


def worker_init_tf(G):
    G.sess = tf.Session()
    G.sess.__enter__()


def worker_init_tf_vars(G):
    G.sess.run(tf.initialize_all_variables())


class BatchSampler(BaseSampler):
    def __init__(self, algo, n_envs=1):
        super(BatchSampler, self).__init__(algo)
        self.n_envs = n_envs

    def start_worker(self):
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf)
        parallel_sampler.populate_task(self.algo.env, self.algo.policy)
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf_vars)

    def shutdown_worker(self):
        parallel_sampler.terminate_task(scope=self.algo.scope)

    def obtain_samples(self, itr, reset_args=None, return_dict=False, log_prefix=''):
        cur_policy_params = self.algo.policy.get_param_values()
        if hasattr(self.algo.env,"get_param_values"):
            try:
                cur_env_params = self.algo.env.get_param_values()
            except:
                cur_env_params = None
        else:
            cur_env_params = None
        import time
        start = time.time()
        # first, a naive implementation.
        paths = {}
        if type(reset_args) != list and type(reset_args)!=np.ndarray:
            reset_args = [reset_args]*self.n_envs
        if self.algo.policy.all_param_vals:
            cur_policy_params = [flatten_tensors(x.values()) for x in self.algo.policy.all_param_vals]
        else:
            cur_policy_params = [cur_policy_params]*self.n_envs
        for i in range(self.n_envs):
            # set cur_policy_params = self.algo.policy.all_param_vals[i]? don't forget to reset env appropriately.
            paths[i] = parallel_sampler.sample_paths(
                policy_params=cur_policy_params[i],
                env_params=cur_env_params,
                max_samples=self.algo.batch_size / self.n_envs,
                max_path_length=self.algo.max_path_length,
                scope=self.algo.scope,
                reset_arg=reset_args[i],
            )
        total_time = time.time() - start
        logger.record_tabular(log_prefix+"TotalExecTime", total_time)

        if not return_dict:
            flatten_list = lambda l: [item for sublist in l for item in sublist]
            paths = flatten_list(paths.values())

        #import pdb; pdb.set_trace()  # TODO - post process paths?
        #if self.algo.whole_paths:
        return paths
        #else:
        #    paths_truncated = parallel_sampler.truncate_paths(paths, self.algo.batch_size)
        #    return paths_truncated
