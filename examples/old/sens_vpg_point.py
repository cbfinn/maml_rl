#from rllab.algos.vpg import VPG
from sandbox.rocky.tf.algos.sensitive_vpg import SensitiveVPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from examples.point_env import PointEnv
from examples.point_env_randgoal import PointEnvRandGoal
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
#from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.sens_minimal_gauss_mlp_policy import SensitiveGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

import tensorflow as tf

stub(globals())

#env = TfEnv(normalize(PointEnv()))
env = TfEnv(normalize(PointEnvRandGoal()))
policy = SensitiveGaussianMLPPolicy(
    name="policy",
    env_spec=env.spec,
    grad_step_size=1.0,
    hidden_nonlinearity=tf.nn.relu,
)
baseline = LinearFeatureBaseline(env_spec=env.spec)
#baseline = ZeroBaseline(env_spec=env.spec)
algo = SensitiveVPG(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=20, # use 100 trajs for grad update
    max_path_length=5,
    meta_batch_size=100,
    n_itr=100,
    use_sensitive=False,
    optimizer_args={'learning_rate': 1e-3}
    #plot=True,
)
run_experiment_lite(
    algo.train(),
    n_parallel=1,
    snapshot_mode="last",
    seed=1,
    exp_prefix='sensitive1dT5_2017_01_18',
    exp_name='nosensitive_linbaseline',
    #plot=True,
)
