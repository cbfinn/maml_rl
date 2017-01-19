#from rllab.algos.vpg import VPG
from sandbox.rocky.tf.algos.vpg import VPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from examples.point_env import PointEnv
from examples.point_env_randgoal import PointEnvRandGoal
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
#from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
#from sandbox.rocky.tf.policies.minimal_gauss_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

stub(globals())

#env = TfEnv(normalize(PointEnv()))
env = TfEnv(normalize(PointEnvRandGoal()))
policy = GaussianMLPPolicy(
    name="policy",
    env_spec=env.spec,
)
#baseline = LinearFeatureBaseline(env_spec=env.spec)
baseline = ZeroBaseline(env_spec=env.spec)
algo = VPG(
    env=env,
    policy=policy,
    baseline=baseline,
    #batch_size=20,
    max_path_length=5,
    n_itr=100,
    #plot=True,
)
run_experiment_lite(
    algo.train(),
    n_parallel=1,
    snapshot_mode="last",
    seed=1,
    exp_prefix='deleteme',
    exp_name='deleteme',
    #plot=True,
)
