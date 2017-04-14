use_tf = True

if use_tf:
    from sandbox.rocky.tf.algos.trpo import TRPO
    # from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from sandbox.rocky.tf.policies.minimal_gauss_mlp_policy import GaussianMLPPolicy
    from sandbox.rocky.tf.envs.base import TfEnv
else:
    from rllab.algos.trpo import TRPO
    from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.swimmer_randgoal_oracle_env import SwimmerRandGoalOracleEnv
from rllab.envs.mujoco.swimmer_randgoal_env import SwimmerRandGoalEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.half_cheetah_env_oracle import HalfCheetahEnvOracle
from rllab.envs.mujoco.half_cheetah_env_direc_oracle import HalfCheetahEnvDirecOracle
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())


#env = normalize(SwimmerEnv())
#env = normalize(SwimmerRandGoalOracleEnv())
#env = normalize(SwimmerRandGoalEnv())

max_path_length = 500
#env = normalize(HalfCheetahEnv())
env = normalize(HalfCheetahEnvDirecOracle())

#env = normalize(Walker2DEnv())
if use_tf:
    env = TfEnv(env)
    policy = GaussianMLPPolicy(
        name='policy',
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        #hidden_sizes=(32, 32)
        hidden_sizes=(100, 100)
    )
else:
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(100, 100)
    )

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=max_path_length*10,  # was 4k
    max_path_length=max_path_length,
    n_itr=500,
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
    exp_prefix='deleteme',
    #exp_prefix='trpo_sensitive_cheetah' + str(max_path_length),
    exp_name='oracledirec_env',
    #plot=True,
)
