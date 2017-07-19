
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.mujoco.ant_env_rand import AntEnvRand
from rllab.envs.mujoco.ant_env_rand_direc import AntEnvRandDirec
from rllab.envs.mujoco.ant_env_direc_oracle import AntEnvDirecOracle
from rllab.envs.mujoco.ant_env_rand_goal import AntEnvRandGoal
from rllab.envs.mujoco.ant_env_rand_goal_oracle import AntEnvRandGoalOracle
from rllab.envs.mujoco.ant_env_rand import AntEnvRand
from rllab.envs.mujoco.ant_env_oracle import AntEnvOracle
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy
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
    def task_var(self):  # fwd/bwd task or goal vel task
        # 0 for fwd/bwd, 1 for goal vel (kind of), 2 for goal pose
        return [2]

# should also code up alternative KL thing

variants = VG().variants()

max_path_length = 200
num_grad_updates = 1
use_maml = True

for v in variants:
    task_var = v['task_var']
    oracle = v['oracle']

    if task_var == 0:
        task_var = 'direc'
        exp_prefix = 'bugfix_trpo_maml_antdirec' + str(max_path_length)
        if oracle:
            env = TfEnv(normalize(AntEnvDirecOracle()))
        else:
            env = TfEnv(normalize(AntEnvRandDirec()))
    elif task_var == 1:
        task_var = 'vel'
        exp_prefix = 'posticml_trpo_maml_ant' + str(max_path_length)
        if oracle:
            env = TfEnv(normalize(AntEnvOracle()))
        else:
            env = TfEnv(normalize(AntEnvRand()))
    elif task_var == 2:
        task_var = 'pos'
        exp_prefix = 'posticml_trpo_maml_antpos_' + str(max_path_length)
        if oracle:
            env = TfEnv(normalize(AntEnvRandGoalOracle()))
        else:
            env = TfEnv(normalize(AntEnvRandGoal()))
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
        n_itr=2000,
        use_maml=use_maml,
        step_size=0.01,
        plot=False,
    )

    if oracle:
        exp_name = 'oracleenv100traj'
    else:
        exp_name = 'randenv100traj'


    run_experiment_lite(
        algo.train(),
        exp_prefix=exp_prefix,
        exp_name=exp_name,
        # Number of parallel workers for sampling
        n_parallel=8,
        # Only keep the snapshot parameters for the last iteration
        #snapshot_mode="last",
        snapshot_mode="gap",
        snapshot_gap=25,
        sync_s3_pkl=True,
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=v["seed"],
        mode="local",
        #mode="ec2",
        #variant=v,
        # plot=True,
        # terminate_machine=False,
    )
