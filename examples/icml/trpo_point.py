
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.policies.minimal_gauss_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from examples.point_env_randgoal import PointEnvRandGoal
from examples.point_env_randgoal_oracle import PointEnvRandGoalOracle
#from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

import tensorflow as tf

#env = normalize(PointEnvRandGoal())
env = normalize(PointEnvRandGoalOracle())
#env = normalize(HalfCheetahEnv())
#env = normalize(Walker2DEnv())
env = TfEnv(env)
policy = GaussianMLPPolicy(
    name='policy',
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    #hidden_sizes=(32, 32)
    #hidden_nonlinearity=tf.nn.relu,
    hidden_sizes=(100, 100)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=500,  # was 4k
    max_path_length=5,
    n_itr=100,
    discount=0.99,
    step_size=0.01,
    #plot=True,
)
#algo.train()

run_experiment_lite(
    algo.train(),
    # Number of parallel workers for sampling
    n_parallel=4,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    exp_prefix='vpg_sensitive_point',
    exp_name='oracleenv',
    #plot=True,
)
