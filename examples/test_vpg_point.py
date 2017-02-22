
from sandbox.rocky.tf.algos.vpg import VPG
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.algos.sensitive_vpg import SensitiveVPG
# from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.minimal_gauss_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from examples.point_env_randgoal import PointEnvRandGoal
from examples.point_env_randgoal_oracle import PointEnvRandGoalOracle
#from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

import joblib
import numpy as np
import tensorflow as tf

sens = False

#initial_params_file = 'data/local/trpo-sensitive-point/bignet_sens1_fbs20_mbs100_flr_1.0baseline_linear_step11_oracle0/params.pkl'
#initial_params_file = 'data/local/trpo-sensitive-point/bignet_sens1_fbs20_mbs100_flr_1.0baseline_linear_step11_oracle0/params.pkl'
#initial_params_file = 'data/local/trpo-sensitive-point/bignet_sens0_fbs20_mbs100_flr_1.0metalr_0.01_step11_oracle0/params.pkl'
#initial_params_file = 'data/local/trpo-sensitive-point0d/bignet_sens1_fbs20_mbs100_flr_1.0metalr_0.01_step11_oracle0/params.pkl'

# 10 goal
#initial_params_file = 'data/local/trpo-sensitive-point10goal/sens1_fbs20_mbs100_flr_1.0metalr_0.01_step11_oracle0/params.pkl'
initial_params_file = 'data/local/trpo-sensitive-point10goal/randenv/params.pkl'

# 2d range of goals
#initial_params_file = 'data/local/vpg-sensitive-point/sens1_fbs20_mbs100_flr_1.0metalr_0.01_step11/params.pkl'
#initial_params_file = 'data/local/vpg-sensitive-point/randenv/params.pkl'

# horizon of 100
#initial_params_file1 = 'data/local/vpg-sensitive-point100/sens1_fbs20_mbs20_flr_1.0metalr_0.01_step11/params.pkl'
initial_params_file1 = 'data/local/vpg-sensitive-point100/trposens1_fbs20_mbs20_flr_0.5metalr_0.01_step11/params.pkl'
initial_params_file2 = 'data/local/vpg-sensitive-point100/vpgrandenv/params.pkl'
initial_params_file3 = 'data/local/vpg-sensitive-point100/sens0_fbs20_mbs20_flr_1.0metalr_0.01_step11/params.pkl'
initial_params_files = [initial_params_file1, initial_params_file2, initial_params_file3]

# the 10 goals
#goals = [[-0.5,0], [0.5,0],[0.2,0.2],[-0.2,-0.2],[0.5,0.5],[0,0.5],[0,-0.5],[-0.5,-0.5],[0.5,-0.5],[-0.5,0.5]]

test_num_goals = 10
np.random.seed(1)
goals = np.random.uniform(-0.5, 0.5, size=(test_num_goals, 2, ))
print goals

#step_sizes = [0.5, 1.0, 1.0]


initial_params_files = [initial_params_file1]
step_sizes = [0.5]


#initial_params_files = [initial_params_file1, initial_params_file2, ]

all_avg_returns = []
for step_i, initial_params_file in zip(range(len(step_sizes)), initial_params_files):
    avg_returns = []
    for goal in goals:
        goal = list(goal)

        if sens:
            env = normalize(PointEnvRandGoal(goal=goal))
            n_itr = 1
            #env = normalize(PointEnvRandGoal())
        else:
            #env = normalize(PointEnvRandGoal(goal=[0.5,-0.5]))
            env = normalize(PointEnvRandGoal(goal=goal))
            n_itr = 4
        env = TfEnv(env)
        policy = GaussianMLPPolicy(  # random policy
            name='policy',
            env_spec=env.spec,
            # The neural network policy should have two hidden layers, each with 32 hidden units.
            #hidden_sizes=(32, 32)
            #hidden_nonlinearity=tf.nn.relu,
            hidden_sizes=(100, 100),
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)
        if sens:
            algo = SensitiveVPG(
                env=env,
                policy=None,
                load_policy=initial_params_file,
                baseline=baseline,
                batch_size=20, #100,  # was 4k
                meta_batch_size=1,  # only used for sens
                max_path_length=5,
                n_itr=n_itr,
                #tf_optimizer_cls=tf.train.GradientDescentOptimizer,
                #tf_optimizer_args={'learning_rate': 1.0}
                #plot=True,
            )
        else:
            algo = VPG(
                env=env,
                policy=None,
                load_policy=initial_params_file,
                baseline=baseline,
                batch_size=4000,  # 2x
                max_path_length=100,
                n_itr=n_itr,
                #step_size=10.0,
                optimizer_args={'init_learning_rate': step_sizes[step_i], 'tf_optimizer_args': {'learning_rate': 0.5*step_sizes[step_i]}, 'tf_optimizer_cls': tf.train.GradientDescentOptimizer}
            )


        run_experiment_lite(
            algo.train(),
            # Number of parallel workers for sampling
            n_parallel=4,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            seed=4,
            exp_prefix='trpopoint2d_test',
            exp_name='test',
            #plot=True,
        )
        # get return from the experiment
        import csv
        with open('data/local/trpopoint2d-test/test/progress.csv', 'r') as f:
            reader = csv.reader(f, delimiter=',')
            i = 0
            row = None
            returns = []
            for row in reader:
                i+=1
                if i ==1:
                    assert row[-1] == 'AverageReturn'
                else:
                    returns.append(float(row[-1]))
            avg_returns.append(returns)
    all_avg_returns.append(avg_returns)

for i in range(len(initial_params_files)):
    returns = []
    for itr in range(n_itr):
        returns.append(np.mean([ret[itr] for ret in all_avg_returns[i]]))
    print initial_params_files[i], returns #np.mean(all_avg_returns[i]), np.std(all_avg_returns[i])




