#from rllab.algos.vpg import VPG
from sandbox.rocky.tf.algos.sensitive_vpg import SensitiveVPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from examples.point_env import PointEnv
from examples.point_env_randgoal import PointEnvRandGoal
from examples.point_env_randgoal_oracle import PointEnvRandGoalOracle
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
#from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.sens_minimal_gauss_mlp_policy import SensitiveGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

import tensorflow as tf

learning_rates = [1e-3] # 1e-3 for sensitive, 1e-2 for oracle, non-sensitive
fast_learning_rates = [0.5] #[[0.2,0.8]] #[0.5]
baselines = ['linear']
fast_batch_size = 100
meta_batch_size = 10
max_path_length = 5

for fast_learning_rate in fast_learning_rates:
    for learning_rate in learning_rates:
        for bas in baselines:
            stub(globals())

            #env = TfEnv(normalize(PointEnv()))
            env = TfEnv(normalize(PointEnvRandGoal()))
            #env = TfEnv(normalize(PointEnvRandGoalOracle()))
            policy = SensitiveGaussianMLPPolicy(
                name="policy",
                env_spec=env.spec,
                grad_step_size=fast_learning_rate,
                hidden_nonlinearity=tf.nn.relu,
            )
            if bas == 'zero':
                baseline = ZeroBaseline(env_spec=env.spec)
            elif bas == 'linear':
                baseline = LinearFeatureBaseline(env_spec=env.spec)
            else:
                baseline = GaussianMLPBaseline(env_spec=env.spec)
            algo = SensitiveVPG(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=fast_batch_size, # number of trajs for grad update
                max_path_length=max_path_length,
                meta_batch_size=meta_batch_size,
                n_itr=200,
                use_sensitive=True,
                optimizer_args={'tf_optimizer_args':{'learning_rate': learning_rate}},
                plot=False,
            )
            run_experiment_lite(
                algo.train(),
                n_parallel=1,
                snapshot_mode="last",
                seed=1,
                #exp_prefix='deleteme',
                #exp_name='deleteme'
                #exp_prefix='sensitive1dT5_2017_01_19',
                #exp_prefix='bugfix_sensitive0d_8tasks_T'+str(max_path_length)+'_2017_02_05',
                exp_prefix='bugfix_sensitive2d_T'+str(max_path_length)+'_2017_02_07',
                exp_name='sensitive_fbs'+str(fast_batch_size)+'_mbs'+str(meta_batch_size)+'_flr_' + str(fast_learning_rate) + '_lr_' + str(learning_rate) + 'baseline_' + bas +'_length'+str(),
                plot=False,
            )
