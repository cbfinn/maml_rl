
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

oracle = False
direc = False


class VG(VariantGenerator):

    #@variant
    #def fast_lr(self):
    #    return [0.01, 0.1, 1.0] #[0.01, 0.05, 0.1]  # would like to try 0.01, 0.05, 0.1, 0.2

    #@variant
    #def meta_step_size(self):
    #    return [0.01, 0.02] #[0.01, 0.02] #, 0.05, 0.1]

    #@variant
    #def fast_batch_size(self):
    #    return [20,40,80]  # 10, 20, 40

    #@variant
    #def meta_batch_size(self):
    #    return [20, 40] # try 20, 40

    @variant
    def seed(self):
        return [1]

# should also code up alternative KL thing

variants = VG().variants()

#fast_learning_rates = [0.1]  # 0.5 works for [0.1, 0.2], too high for 2 step
#fast_batch_size = 10  # 10 works for [0.1, 0.2], 20 doesn't improve much for [0,0.2]
#meta_batch_size = 20  # 10 also works, but much less stable, 20 is fairly stable, 40 is more stable
#meta_step_size = 0.01  # trpo constraint step size
max_path_length = 200
num_grad_updates = 1
use_sensitive = True

for v in variants:

    #env = TfEnv(normalize(HalfCheetahEnvRand()))
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
        n_itr=500,
        use_sensitive=use_sensitive,
        step_size=0.01,
        #optimizer_args={'tf_optimizer_args':{'learning_rate': learning_rate}},
        plot=False,
    )

    if oracle:
        exp_name = 'oracleenv'
    else:
        exp_name = 'randenv'
    if direc:
        exp_prefix = 'trpo_sensitive_cheetahdirec' + str(max_path_length)
    else:
        exp_prefix = 'trpo_sensitive_cheetah' + str(max_path_length)

    run_experiment_lite(
        algo.train(),
        exp_prefix=exp_prefix,
        exp_name=exp_name,
        #exp_name='sens'+str(int(use_sensitive))+'_fbs'+str(v['fast_batch_size'])+'_mbs'+str(v['meta_batch_size'])+'_flr_' + str(v['fast_lr'])  + '_mlr' + str(v['meta_step_size']),
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=v["seed"],
        mode="local",
        #mode="ec2",
        variant=v,
        # plot=True,
        # terminate_machine=False,
    )
