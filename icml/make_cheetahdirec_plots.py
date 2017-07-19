import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#names = ['maml','maml0','random','oracle']

prefix = 'icml_cheetahdirec_results_'
oracle_pkl = prefix+'oracle.pkl'

maml_pkl = prefix+'maml.pkl'
pretrain_pkl = prefix+'pretrain.pkl'
random_pkl = prefix+'random.pkl'

key = 'task_avg_returns'

n_itr = 4

with open(oracle_pkl, 'rb') as f:
    oracle_data = np.array(pickle.load(f)[key])[0]


oracle_data = np.reshape(oracle_data, [-1, 1])
oracle_data = np.tile(oracle_data[:,0:1], [1,n_itr])

fig = plt.figure()
plt.clf()

with open(maml_pkl, 'rb') as maml_f:
    maml_data = np.array(pickle.load(maml_f)[key]).T[:,:n_itr]

with open(pretrain_pkl, 'rb') as f:
    pretrain_data = np.array(pickle.load(f)[key]).T[:,:n_itr]

with open(random_pkl, 'rb') as f:
    random_data = np.array(pickle.load(f)[key]).T[:,:n_itr]


sns.tsplot(time=range(n_itr), data=maml_data[:,:n_itr], color='g', linestyle='-', marker='o', condition='MAML (ours)', legend=False)
sns.tsplot(time=range(n_itr), data=pretrain_data[:,:n_itr], color='b', linestyle='--', marker='s', condition='pretrained', legend=False)
sns.tsplot(time=range(n_itr), data=random_data[:,:n_itr], color='k', linestyle=':', marker='^', condition='random', legend=False)
sns.tsplot(time=range(n_itr), data=oracle_data[:,:n_itr], color='r', linestyle='-.', marker='v', condition='oracle', legend=False)
ax = fig.gca()
#ax.set(yscale='symlog')

#plt.ylim([-100,-2.0])

plt.xlabel('number of gradient steps', fontsize=26)
plt.ylabel('average return', fontsize=26)
#lgd=plt.legend(['MAML (ours)', 'pretrained', 'random', 'oracle'], loc=0, bbox_to_anchor=(1, 0.5), fontsize=20)
plt.title('half-cheetah, forward/backward', fontsize=26)
#plt.ylim([-0.04, 3.5])
plt.tight_layout()

ax = plt.gca()
plt.setp(ax.get_xticklabels(), fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=18)
plt.xticks(np.arange(0,4,1.0))
plt.savefig('cheetahdirec_results.png', bbox_inches='tight')
