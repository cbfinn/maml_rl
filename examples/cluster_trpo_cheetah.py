
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.mujoco.half_cheetah_env_rand import HalfCheetahEnvRand
from rllab.envs.mujoco.half_cheetah_env_rand_direc import HalfCheetahEnvRandDirec
from rllab.envs.mujoco.half_cheetah_env_direc_oracle import HalfCheetahEnvDirecOracle
from rllab.envs.mujoco.half_cheetah_env_rand import HalfCheetahEnvRand
from rllab.envs.mujoco.half_cheetah_env_oracle import HalfCheetahEnvOracle
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.sens_minimal_gauss_mlp_policy import SensitiveGaussianMLPPolicy
from sandbox.rocky.tf.policies.minimal_gauss_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv


import tensorflow as tf

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant


class VG(VariantGenerator):

    @variant
    def seed(self):
        return [1]

    @variant
    def oracle(self):
        # oracle or baseline
        return [True]

    @variant
    def direc(self):
        return [False]


# should also code up alternative KL thing

variants = VG().variants()

max_path_length = 200
num_grad_updates = 1
use_sensitive = True

for v in variants:
    direc = v['direc']
    oracle = v['oracle']

    if direc:
        if oracle:
            env = TfEnv(normalize(HalfCheetahEnvDirecOracle()))
        else:
            env = TfEnv(normalize(HalfCheetahEnvRandDirec()))
    else:
        if oracle:
            env = TfEnv(normalize(HalfCheetahEnvOracle()))
        else:
            env = TfEnv(normalize(HalfCheetahEnvRand()))
    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_nonlinearity=tf.nn.relu,
        hidden_sizes=(100,100),
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=max_path_length*100, # number of trajs for grad update
        max_path_length=max_path_length,
        n_itr=1000,
        use_sensitive=use_sensitive,
        step_size=0.01,
        plot=False,
    )

    if oracle:
        exp_name = 'oracleenv'
    else:
        exp_name = 'randenv'
    if direc:
        exp_prefix = 'trpo_sensitive_cheetahdirec' + str(max_path_length)
    else:
        exp_prefix = 'bugfix_trpo_sensitive_cheetah' + str(max_path_length)

    run_experiment_lite(
        algo.train(),
        exp_prefix=exp_prefix,
        exp_name=exp_name,
        #exp_name='sens'+str(int(use_sensitive))+'_fbs'+str(v['fast_batch_size'])+'_mbs'+str(v['meta_batch_size'])+'_flr_' + str(v['fast_lr'])  + '_mlr' + str(v['meta_step_size']),
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        #snapshot_mode="last",
        snapshot_mode="gap",
        snapshot_gap=25,
        sync_s3_pkl=True,
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=v["seed"],
        #mode="local",
        mode="ec2",
        variant=v,
        # plot=True,
        # terminate_machine=False,
    )
