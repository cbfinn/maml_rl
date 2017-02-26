import matplotlib.pyplot as plt
import numpy as np
import pickle

prefixes = ['maml', 'pretrain']

n_itr = 4
goal =  [-0.29554775,  0.37811744]


plt.clf()
plt.hold(True)
itr_line_styles = [':', '-.', '--', '-']
maml_colors = ['dodgerblue', None, None, 'darkblue']
pretrain_colors = ['limegreen',None, None, 'darkgreen']

plt.figure(figsize=(9.0,4.5))
ind = 0
#for itr in range(n_itr):
for itr in [0,3]:
    with open('maml_paths_itr'+str(itr)+'.pkl', 'rb') as f:
        paths = pickle.load(f)
    points = paths[ind]['observations']
    plt.plot(points[:,0], points[:,1], itr_line_styles[itr], color=maml_colors[itr], linewidth=2)
plt.plot(goal[0], goal[1], 'r*', markersize=28, markeredgewidth=0)
plt.title('MAML', fontsize=25)
plt.legend(['pre-update',  '3 steps', 'goal position'], fontsize=23, loc='upper right') #, 'pretrain preupdate', 'pretrain 3 steps'])
plt.xlim([-0.5, 0.3])
plt.ylim([-0.2, 0.6])
plt.tight_layout()
ax = plt.gca()
plt.setp(ax.get_xticklabels(), fontsize=14)
plt.setp(ax.get_yticklabels(), fontsize=14)
plt.savefig('maml_paths_viz.png')

plt.clf()
#for itr in n_itr:
for itr in [0,3]:
    with open('pretrain_paths_itr'+str(itr)+'.pkl', 'rb') as f:
        paths = pickle.load(f)
    points = paths[ind]['observations']
    plt.plot(points[:,0], points[:,1], itr_line_styles[itr], color=pretrain_colors[itr], linewidth=2)
plt.plot(goal[0], goal[1], 'r*', markersize=28, markeredgewidth=0)
plt.title('pretrained', fontsize=25)
plt.legend(['pre-update',  '3 steps', 'goal position'], fontsize=23, loc='lower left') #, 'pretrain preupdate', 'pretrain 3 steps'])

plt.xlim([-0.5, 0.3])
plt.ylim([-0.2, 0.6])
plt.tight_layout()
ax = plt.gca()
plt.setp(ax.get_xticklabels(), fontsize=14)
plt.setp(ax.get_yticklabels(), fontsize=14)
plt.savefig('pretrain_paths_viz.png')
