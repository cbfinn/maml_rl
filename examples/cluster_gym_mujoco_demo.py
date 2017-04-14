from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
#from rllab.envs.gym_env import GymEnv
#from rllab.envs.mujoco.swimmer_randgoal_env import SwimmerRandGoalEnv
from rllab.envs.mujoco.swimmer_randgoal_oracle_env import SwimmerRandGoalOracleEnv
import sys

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant


class VG(VariantGenerator):

    @variant
    def step_size(self):
        return [0.005,0.01,0.02] #, 0.05, 0.1]

    @variant
    def seed(self):
        return [2,3] #, 11, 21, 31, 41]

variants = VG().variants()

for v in variants:

    env = TfEnv(normalize(SwimmerRandGoalOracleEnv()))
    #env = TfEnv(normalize(GymEnv('HalfCheetah-v1', record_video=False, record_log=False)))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(100, 100),
        name="policy"
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=10000,
        max_path_length=500,
        n_itr=500,
        discount=0.99,
        step_size=v["step_size"],
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="trpo_swimmer_baselines",
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=v["seed"],
        # mode="local",
        mode="ec2",
        variant=v,
        # plot=True,
        # terminate_machine=False,
    )
