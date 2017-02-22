
from sandbox.rocky.tf.algos.sensitive_trpo import SensitiveTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.swimmer_randgoal_env import SwimmerRandGoalEnv
from rllab.envs.mujoco.swimmer_randgoal_oracle_env import SwimmerRandGoalOracleEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
#from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.sens_minimal_gauss_mlp_policy import SensitiveGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

import tensorflow as tf

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant


class VG(VariantGenerator):

    @variant
    def fast_lr(self):
        return [0.001] #[0.01, 0.05, 0.1]  # would like to try 0.01, 0.05, 0.1, 0.2

    @variant
    def meta_step_size(self):
        return [0.02] #[0.01, 0.02] #, 0.05, 0.1]

    @variant
    def fast_batch_size(self):
        return [10, 20]  # 10, 20, 40

    @variant
    def meta_batch_size(self):
        return [20, 40] # try 20, 40

    @variant
    def seed(self):
        return [1]

# should also code up alternative KL thing

variants = VG().variants()

#fast_learning_rates = [0.1]  # 0.5 works for [0.1, 0.2], too high for 2 step
#fast_batch_size = 10  # 10 works for [0.1, 0.2], 20 doesn't improve much for [0,0.2]
#meta_batch_size = 20  # 10 also works, but much less stable, 20 is fairly stable, 40 is more stable
#meta_step_size = 0.01  # trpo constraint step size
max_path_length = 500
num_grad_updates = 1
use_sensitive = True
oracle = False  # Use vpg_swimmer.py for oracle

for v in variants:

    env = TfEnv(normalize(SwimmerRandGoalEnv()))
    policy = SensitiveGaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        grad_step_size=v['fast_lr'],
        hidden_nonlinearity=tf.nn.relu,
        hidden_sizes=(100,100),
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = SensitiveTRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=v['fast_batch_size'], # number of trajs for grad update
        max_path_length=max_path_length,
        meta_batch_size=v['meta_batch_size'],
        num_grad_updates=num_grad_updates,
        n_itr=800,
        use_sensitive=use_sensitive,
        step_size=v['meta_step_size'],
        #optimizer_args={'tf_optimizer_args':{'learning_rate': learning_rate}},
        plot=False,
    )

    #algo = TRPO(
    #    env=env,
    #    policy=policy,
    #    baseline=baseline,
    #    batch_size=4000,
    #    max_path_length=100,
    #    n_itr=40,
    #    discount=0.99,
    #    step_size=v["step_size"],
    #    # Uncomment both lines (this and the plot parameter below) to enable plotting
    #    # plot=True,
    #)

    run_experiment_lite(
        algo.train(),
        exp_prefix="trpo_sensitive_swimmer",
        exp_name='sens'+str(int(use_sensitive))+'_fbs'+str(v['fast_batch_size'])+'_mbs'+str(v['meta_batch_size'])+'_flr_' + str(v['fast_lr']) +'_step1'+str(num_grad_updates) + '_orac' + str(int(oracle)),
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
