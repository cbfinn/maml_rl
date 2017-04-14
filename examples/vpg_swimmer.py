#from rllab.algos.vpg import VPG
from sandbox.rocky.tf.algos.vpg import VPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.swimmer_randgoal_oracle_env import SwimmerRandGoalOracleEnv
from rllab.envs.mujoco.swimmer_randgoal_env import SwimmerRandGoalEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
#from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
#from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.minimal_gauss_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

stub(globals())
oracle = False
random = True

if oracle:
    env = TfEnv(normalize(SwimmerRandGoalOracleEnv()))
    batch_size = 200
elif random:
    env = TfEnv(normalize(SwimmerRandGoalEnv()))
    batch_size = 200
else:
    env = TfEnv(normalize(SwimmerEnv()))
    batch_size = 20
policy = GaussianMLPPolicy(
    name="policy",
    env_spec=env.spec,
    hidden_sizes=(100,100),
)
baseline = LinearFeatureBaseline(env_spec=env.spec)
#baseline = ZeroBaseline(env_spec=env.spec)
algo = VPG(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=500*batch_size,
    max_path_length=500,
    n_itr=500,
    #plot=True,
    optimizer_args={'tf_optimizer_args':{'learning_rate': 1e-3}},
)
run_experiment_lite(
    algo.train(),
    n_parallel=1,  # try increasing this to make it faster??? (Maybe need to modify code for this)
    snapshot_mode="last",
    seed=1,
    exp_prefix='vpgswimmer',
    #exp_name='basic',
    exp_name='randomenv',
    #plot=True,
)
