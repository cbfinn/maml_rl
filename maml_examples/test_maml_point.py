from maml_examples.point_env_randgoal import PointEnvRandGoal
from maml_examples.point_env_randgoal_oracle import PointEnvRandGoalOracle
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.algos.vpg import VPG
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.policies.minimal_gauss_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

import csv
import joblib
import numpy as np
import pickle
import tensorflow as tf

stub(globals())

# horizon of 100
initial_params_file1 = 'data/local/vpg-maml-point100/trpomaml1_fbs20_mbs20_flr_0.5metalr_0.01_step11/params.pkl'
initial_params_file2 = 'data/local/vpg-maml-point100/vpgrandenv/params.pkl'
initial_params_file3 = 'data/local/vpg-maml-point100/maml0_fbs20_mbs20_flr_1.0metalr_0.01_step11/params.pkl'
initial_params_file4 = 'data/local/vpg-maml-point100/oracleenv2/params.pkl'

test_num_goals = 40
np.random.seed(1)
goals = np.random.uniform(-0.5, 0.5, size=(test_num_goals, 2, ))
print(goals)

goals = [goals[6]]


# ICML values
step_sizes = [0.5, 0.5, 0.5,0.0, 0.5]
initial_params_files = [initial_params_file1, initial_params_file3, None,initial_params_file4]
gen_name = 'icml_point_results_'
names = ['maml','maml0','random','oracle']

exp_names = [gen_name + name for name in names]

all_avg_returns = []
for step_i, initial_params_file in zip(range(len(step_sizes)), initial_params_files):
    avg_returns = []
    for goal in goals:
        goal = list(goal)


        if initial_params_file is not None and 'oracle' in initial_params_file:
            env = normalize(PointEnvRandGoalOracle(goal=goal))
            n_itr = 1
        else:
            env = normalize(PointEnvRandGoal(goal=goal))
            n_itr = 5
        env = TfEnv(env)
        policy = GaussianMLPPolicy(  # random policy
            name='policy',
            env_spec=env.spec,
            hidden_sizes=(100, 100),
        )


        if initial_params_file is not None:
            policy = None

        baseline = LinearFeatureBaseline(env_spec=env.spec)
        algo = VPG(
            env=env,
            policy=policy,
            load_policy=initial_params_file,
            baseline=baseline,
            batch_size=4000,  # 2x
            max_path_length=100,
            n_itr=n_itr,
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
        import pdb; pdb.set_trace()
        # get return from the experiment
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
    std_returns = []
    task_avg_returns = []
    for itr in range(len(all_avg_returns[i][0])):
        returns.append(np.mean([ret[itr] for ret in all_avg_returns[i]]))
        std_returns.append(np.std([ret[itr] for ret in all_avg_returns[i]]))

        task_avg_returns.append([ret[itr] for ret in all_avg_returns[i]])

    results = {'task_avg_returns': task_avg_returns}
    with open(exp_names[i] + '.pkl', 'w') as f:
        pickle.dump(results, f)

import pdb; pdb.set_trace()

