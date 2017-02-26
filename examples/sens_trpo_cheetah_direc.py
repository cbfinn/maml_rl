#from rllab.algos.vpg import VPG
from sandbox.rocky.tf.algos.sensitive_vpg import SensitiveVPG
from sandbox.rocky.tf.algos.sensitive_trpo import SensitiveTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.swimmer_randgoal_env import SwimmerRandGoalEnv
from rllab.envs.mujoco.swimmer_randgoal_oracle_env import SwimmerRandGoalOracleEnv
from rllab.envs.mujoco.half_cheetah_env_rand_direc import HalfCheetahEnvRandDirec
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
#from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.sens_minimal_gauss_mlp_policy import SensitiveGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

import tensorflow as tf

#[[0.2,0.8]] #[0.5]  # tried 0.001 and it seemed to low, 0.1 did something reasonable (loss was impr..), # 0.5 does well sometimes, but also goes to nans

# 1e-3 for sensitive, 1e-2 for oracle, non-sensitive
learning_rates = [1e-3]  # 1e-3 works well for 1 step, trying lower for 2 step, trying 1e-2 for large batch
fast_learning_rates = [0.001]  # 0.5 works for [0.1, 0.2], too high for 2 step
baselines = ['linear']
fast_batch_size = 20  # 10 works for [0.1, 0.2], 20 doesn't improve much for [0,0.2]
meta_batch_size = 20  # 10 also works, but much less stable, 20 is fairly stable, 40 is more stable
max_path_length = 500
num_grad_updates = 1
use_sensitive = True

for fast_learning_rate in fast_learning_rates:
    for learning_rate in learning_rates:
        for bas in baselines:
            stub(globals())

            env = TfEnv(normalize(HalfCheetahEnvRandDirec()))
            policy = SensitiveGaussianMLPPolicy(
                name="policy",
                env_spec=env.spec,
                grad_step_size=fast_learning_rate,
                hidden_nonlinearity=tf.nn.relu,
                hidden_sizes=(100,100),
            )
            if bas == 'zero':
                baseline = ZeroBaseline(env_spec=env.spec)
            elif bas == 'linear':
                baseline = LinearFeatureBaseline(env_spec=env.spec)
            else:
                baseline = GaussianMLPBaseline(env_spec=env.spec)
            algo = SensitiveTRPO(
            #algo = SensitiveVPG(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=fast_batch_size, # number of trajs for grad update
                max_path_length=max_path_length,
                meta_batch_size=meta_batch_size,
                num_grad_updates=num_grad_updates,
                n_itr=400,
                use_sensitive=use_sensitive,
                #optimizer_args={'tf_optimizer_args':{'learning_rate': learning_rate}},
                plot=False,
            )
            run_experiment_lite(
                algo.train(),
                n_parallel=0,
                snapshot_mode="last",
                seed=1,
                #exp_prefix='deleteme',
                #exp_name='deleteme'
                #exp_prefix='sensitive1dT5_2017_01_19',
                #exp_prefix='bugfix_sensitive0d_8tasks_T'+str(max_path_length)+'_2017_02_05',
                exp_prefix='trpo_sensitive_cheetahdirec' + str(max_path_length),
                exp_name='sens'+str(int(use_sensitive))+'_fbs'+str(fast_batch_size)+'_mbs'+str(meta_batch_size)+'_flr_' + str(fast_learning_rate) + '_lr_' + str(learning_rate) + '_step1'+str(num_grad_updates),
                plot=False,
            )
