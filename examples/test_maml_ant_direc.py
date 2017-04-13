
from sandbox.rocky.tf.algos.vpg import VPG
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.algos.sensitive_vpg import SensitiveVPG
# from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.minimal_gauss_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.mujoco.ant_env_rand_direc import AntEnvRandDirec
from rllab.envs.mujoco.ant_env_direc_oracle import AntEnvDirecOracle
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

import joblib
import numpy as np
import tensorflow as tf

sens = False



# TODO - adjust files
file1 = 'data/s3/bugfix-trpo-sensitive-antdirec200/_sens1_fbs40_mbs20_flr_0.1_mlr0.01/itr_350.pkl'  # this is still training
file2 = 'data/s3/bugfix-trpo-sensitive-antdirec200/randenv100traj/params.pkl'
file3 = 'data/s3/bugfix-trpo-sensitive-antdirec200/oracleenv100traj/itr_975.pkl'
# the 10 goals
#goals = [[-0.5,0], [0.5,0],[0.2,0.2],[-0.2,-0.2],[0.5,0.5],[0,0.5],[0,-0.5],[-0.5,-0.5],[0.5,-0.5],[-0.5,0.5]]

test_num_goals = 40
np.random.seed(2)
goals = np.random.uniform(0.1, 0.8, size=(test_num_goals, ))
print(goals)

gen_name = 'icml_antdirec_results_'
names = ['maml','pretrain','random', 'oracle']

step_sizes = [0.1, 0.2, 1.0, 0.0]
initial_params_files = [file1, file2, None, file3]

names = ['maml']
step_sizes=[0.1]
initial_params_files=[file1]

# reun middle2
#names = ['pretrain','random']
#step_sizes = [0.1, 1.0]  # maybe choose 0.05 or 0.2 for pretrain
#initial_params_files = [file2, None]

#names = [ 'oracle']
#step_sizes = [ 0.0]
#initial_params_files = [ file3]

exp_names = [gen_name + name for name in names]

all_avg_returns = []
for step_i, initial_params_file in zip(range(len(step_sizes)), initial_params_files):
    avg_returns = []
    for goal in goals:


        if initial_params_file is not None and 'oracle' in initial_params_file:
            env = normalize(AntEnvDirecOracle())
            n_itr = 1
        elif sens:
            env = normalize(AntEnvRandDirec())
            n_itr = 1
        else:
            env = normalize(AntEnvRandDirec())
            n_itr = 4
        env = TfEnv(env)
        policy = GaussianMLPPolicy(  # random policy
            name='policy',
            env_spec=env.spec,
            hidden_nonlinearity=tf.nn.relu,
            hidden_sizes=(100, 100),
        )


        if initial_params_file is not None:
            policy = None

        baseline = LinearFeatureBaseline(env_spec=env.spec)
        if sens:
            algo = SensitiveVPG(
                env=env,
                policy=policy,
                load_policy=initial_params_file,
                baseline=baseline,
                batch_size=20, #100,  # was 4k
                meta_batch_size=1,  # only used for sens
                max_path_length=200,
                n_itr=n_itr,
                #tf_optimizer_cls=tf.train.GradientDescentOptimizer,
                #tf_optimizer_args={'learning_rate': 1.0}
                plot=True,
            )
        else:
            algo = VPG(
                env=env,
                policy=policy,
                load_policy=initial_params_file,
                baseline=baseline,
                batch_size=8000,
                max_path_length=200,
                n_itr=n_itr,
                #step_size=10.0,
                reset_arg=goal,
                optimizer_args={'init_learning_rate': step_sizes[step_i], 'tf_optimizer_args': {'learning_rate': 0.5*step_sizes[step_i]}, 'tf_optimizer_cls': tf.train.GradientDescentOptimizer}
            )


        run_experiment_lite(
            algo.train(),
            # Number of parallel workers for sampling
            n_parallel=4,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="all",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            seed=1,
            exp_prefix='antdirec_test1',
            exp_name='test',
            plot=True,
        )

        # get return from the experiment
        import csv
        with open('data/local/antdirec-test1/test/progress.csv', 'r') as f:
            reader = csv.reader(f, delimiter=',')
            i = 0
            row = None
            returns = []
            for row in reader:
                i+=1
                if i ==1:
                    ret_idx = row.index('AverageReturn')
                else:
                    returns.append(float(row[ret_idx]))
            avg_returns.append(returns)
    all_avg_returns.append(avg_returns)

    import pickle

    task_avg_returns = []
    for itr in range(len(all_avg_returns[step_i][0])):
        task_avg_returns.append([ret[itr] for ret in all_avg_returns[step_i]])

    results = {'task_avg_returns': task_avg_returns}
    with open(exp_names[step_i] + '.pkl', 'wb') as f:
        pickle.dump(results, f)


for i in range(len(initial_params_files)):
    returns = []
    std_returns = []
    returns.append(np.mean([ret[itr] for ret in all_avg_returns[i]]))
    std_returns.append(np.std([ret[itr] for ret in all_avg_returns[i]]))
    print(initial_params_files[i])
    print(returns) #np.mean(all_avg_returns[i]), np.std(all_avg_returns[i])
    print(std_returns) #np.mean(all_avg_returns[i]), np.std(all_avg_returns[i])


