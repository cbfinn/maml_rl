from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.grid_world_env_rand import GridWorldEnvRand
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from sandbox.rocky.tf.policies.maml_minimal_categorical_mlp_policy import MAMLCategoricalMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

import tensorflow as tf

fast_learning_rates = [0.1]
baselines = ['linear']
fast_batch_size = 20
meta_batch_size = 60
max_path_length = 10
num_grad_updates = 1
meta_step_size = 0.01

use_maml = True

for fast_learning_rate in fast_learning_rates:
    for bas in baselines:
        stub(globals())

        env = TfEnv(normalize(GridWorldEnvRand('four-state')))
        policy = MAMLCategoricalMLPPolicy(
            name="policy",
            env_spec=env.spec,
            grad_step_size=fast_learning_rate,
            hidden_nonlinearity=tf.nn.relu,
            hidden_sizes=(100,100),
        )
        if bas == 'zero':
            baseline = ZeroBaseline(env_spec=env.spec)
        elif 'linear' in bas:
            baseline = LinearFeatureBaseline(env_spec=env.spec)
        else:
            baseline = GaussianMLPBaseline(env_spec=env.spec)
        algo = MAMLTRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=fast_batch_size, # number of trajs for grad update
            max_path_length=max_path_length,
            meta_batch_size=meta_batch_size,
            num_grad_updates=num_grad_updates,
            n_itr=800,
            use_maml=use_maml,
            step_size=meta_step_size,
            plot=False,
        )
        run_experiment_lite(
            algo.train(),
            n_parallel=4,
            snapshot_mode="last",
            seed=1,
            exp_prefix='trpo_maml_4state',
            exp_name='trpo_maml'+str(int(use_maml))+'_fbs'+str(fast_batch_size)+'_mbs'+str(meta_batch_size)+'_flr_' + str(fast_learning_rate) + 'metalr_' + str(meta_step_size) +'_step1'+str(num_grad_updates),
            plot=False,
        )
