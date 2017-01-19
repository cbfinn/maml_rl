from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from examples.point_env import PointEnv
from examples.point_env_randgoal import PointEnvRandGoal
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

env = normalize(PointEnv())
#env = normalize(PointEnvRandGoal())
policy = GaussianMLPPolicy(
    env_spec=env.spec,
)
baseline = LinearFeatureBaseline(env_spec=env.spec)
algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    #max_path_length=5,
)
algo.train()
