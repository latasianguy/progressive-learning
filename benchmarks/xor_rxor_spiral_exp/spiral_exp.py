#%%
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import seaborn as sns 

import numpy as np
import pickle

from sklearn.model_selection import StratifiedKFold
from math import log2, ceil 

import sys
sys.path.append("../../src/")
from lifelong_dnn import LifeLongDNN
from joblib import Parallel, delayed

#%%
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
def get_colors(colors, inds):
    c = [colors[i] for i in inds]
    return c

def generate_2d_rotation(theta=0, acorn=None):
    if acorn is not None:
        np.random.seed(acorn)
    
    R = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    
    return R


def generate_spirals(N, D=2, K=5, noise = 0.5, acorn = None, density=0.3):

    #N number of poinst per class
    #D number of features, 
    #K number of classes
    X = []
    Y = []
    if acorn is not None:
        np.random.seed(acorn)
    
    if K == 2:
        turns = 2
    elif K==3:
        turns = 2.5
    elif K==5:
        turns = 3.5
    elif K==7:
        turns = 4.5
    else:
        print ("sorry, can't currently surpport %s classes " %K)
        return
    
    mvt = np.random.multinomial(N, 1/K * np.ones(K))
    
    if K == 2:
        r = np.random.uniform(0,1,size=int(N/K))
        r = np.sort(r)
        t = np.linspace(0,  np.pi* 4 * turns/K, int(N/K)) + noise * np.random.normal(0, density, int(N/K))
        dx = r * np.cos(t)
        dy = r* np.sin(t)

        X.append(np.vstack([dx, dy]).T )
        X.append(np.vstack([-dx, -dy]).T)
        Y += [0] * int(N/K) 
        Y += [1] * int(N/K)
    else:    
        for j in range(1, K+1):
            r = np.linspace(0.01, 1, int(mvt[j-1]))
            t = np.linspace((j-1) * np.pi *4 *turns/K,  j* np.pi * 4* turns/K, int(mvt[j-1])) + noise * np.random.normal(0, density, int(mvt[j-1]))
            dx = r * np.cos(t)
            dy = r* np.sin(t)

            dd = np.vstack([dx, dy]).T        
            X.append(dd)
            #label
            Y += [j-1] * int(mvt[j-1])
    return np.vstack(X), np.array(Y).astype(int)

#%%
def experiment(n_spiral3, n_spiral5, n_test, reps, n_trees, max_depth, acorn=None):
    #print(1)
    if n_spiral3==0 and n_rxor==0:
        raise ValueError('Wake up and provide samples to train!!!')
    
    if acorn != None:
        np.random.seed(acorn)
    
    errors = np.zeros((reps,4),dtype=float)
    
    for i in range(reps):
        l2f = LifeLongDNN()
        uf = LifeLongDNN()
        #source data
        spiral3, label_spiral3 = generate_spirals(n_spiral3, 2, 3, noise = 2.5)
        test_spiral3, test_label_spiral3 = generate_spirals(n_test, 2, 3, noise = 2.5)
    
        #target data
        spiral5, label_spiral5 = generate_spirals(n_spiral5, 2, 5, noise = 2.5)
        test_spiral5, test_label_spiral5 = generate_spirals(n_test, 2, 5, noise = 2.5)
    
        if n_spiral3 == 0:
            l2f.new_forest(spiral5, label_spiral5, n_estimators=n_trees,max_depth=max_depth)
            
            errors[i,0] = 0.5
            errors[i,1] = 0.5
            
            uf_task2=l2f.predict(test_spiral5, representation=0, decider=0)
            l2f_task2=l2f.predict(test_spiral5, representation='all', decider=0)
            
            errors[i,2] = 1 - np.sum(uf_task2 == test_label_spiral5)/n_test
            errors[i,3] = 1 - np.sum(l2f_task2 == test_label_spiral5)/n_test
        elif n_spiral5 == 0:
            l2f.new_forest(spiral3, label_spiral3, n_estimators=n_trees,max_depth=max_depth)
            
            uf_task1=l2f.predict(test_spiral3, representation=0, decider=0)
            l2f_task1=l2f.predict(test_spiral3, representation='all', decider=0)
            
            errors[i,0] = 1 - np.sum(uf_task1 == test_label_spiral3)/n_test
            errors[i,1] = 1 - np.sum(l2f_task1 == test_label_spiral3)/n_test
            errors[i,2] = 0.5
            errors[i,3] = 0.5
        else:
            l2f.new_forest(spiral3, label_spiral3, n_estimators=n_trees,max_depth=max_depth)
            l2f.new_forest(spiral5, label_spiral5, n_estimators=n_trees,max_depth=max_depth)
            
            uf.new_forest(spiral3, label_spiral3, n_estimators=2*n_trees,max_depth=max_depth)
            uf.new_forest(spiral5, label_spiral5, n_estimators=2*n_trees,max_depth=max_depth)

            uf_task1=uf.predict(test_spiral3, representation=0, decider=0)
            l2f_task1=l2f.predict(test_spiral3, representation='all', decider=0)
            uf_task2=uf.predict(test_spiral5, representation=1, decider=1)
            l2f_task2=l2f.predict(test_spiral5, representation='all', decider=1)
            
            errors[i,0] = 1 - np.sum(uf_task1 == test_label_spiral3)/n_test
            errors[i,1] = 1 - np.sum(l2f_task1 == test_label_spiral3)/n_test
            errors[i,2] = 1 - np.sum(uf_task2 == test_label_spiral5)/n_test
            errors[i,3] = 1 - np.sum(l2f_task2 == test_label_spiral5)/n_test

    return np.mean(errors,axis=0)

#%%
mc_rep = 1000
n_test = 1000
n_trees = 10
n_spiral3 = (100*np.arange(0.5, 7.25, step=0.25)).astype(int)
n_spiral5 = (100*np.arange(0.5, 7.50, step=0.25)).astype(int)

mean_error = np.zeros((4, len(n_spiral3)+len(n_spiral5)))
std_error = np.zeros((4, len(n_spiral3)+len(n_spiral5)))

mean_te = np.zeros((2, len(n_spiral3)+len(n_spiral5)))
std_te = np.zeros((2, len(n_spiral3)+len(n_spiral5)))

for i,n1 in enumerate(n_spiral3):
    print('starting to compute %s spiral 3\n'%n1)
    error = np.array(
        Parallel(n_jobs=40,verbose=1)(
        delayed(experiment)(n1,0,n_test,1,n_trees=n_trees,max_depth=ceil(log2(750))) for _ in range(mc_rep)
    )
    )
    mean_error[:,i] = np.mean(error,axis=0)
    std_error[:,i] = np.std(error,ddof=1,axis=0)
    mean_te[0,i] = np.mean(error[:,0]/error[:,1])
    mean_te[1,i] = np.mean(error[:,2]/error[:,3])
    std_te[0,i] = np.std(error[:,0]/error[:,1],ddof=1)
    std_te[1,i] = np.std(error[:,2]/error[:,3],ddof=1)
    
    if n1==n_spiral3[-1]:
        for j,n2 in enumerate(n_spiral5):
            print('starting to compute %s spiral 5\n'%n2)
            
            error = np.array(
                Parallel(n_jobs=40,verbose=1)(
                delayed(experiment)(n1,n2,n_test,1,n_trees=n_trees,max_depth=ceil(log2(750))) for _ in range(mc_rep)
            )
            )
            mean_error[:,i+j+1] = np.mean(error,axis=0)
            std_error[:,i+j+1] = np.std(error,ddof=1,axis=0)
            mean_te[0,i+j+1] = np.mean(error[:,0]/error[:,1])
            mean_te[1,i+j+1] = np.mean(error[:,2]/error[:,3])
            std_te[0,i+j+1] = np.std(error[:,0]/error[:,1],ddof=1)
            std_te[1,i+j+1] = np.std(error[:,2]/error[:,3],ddof=1)
            
with open('result/mean_spiral.pickle','wb') as f:
    pickle.dump(mean_error,f)
    
with open('result/std_spiral.pickle','wb') as f:
    pickle.dump(std_error,f)
    
with open('result/mean_te_spiral.pickle','wb') as f:
    pickle.dump(mean_te,f)
    
with open('result/std_te_spiral.pickle','wb') as f:
    pickle.dump(std_te,f)

#%%
#mc_rep = 50
mean_error = unpickle('result/mean_spiral.pickle')
std_error = unpickle('result/std_spiral.pickle')

n_xor = (100*np.arange(0.5, 7.25, step=0.25)).astype(int)
n_rxor = (100*np.arange(0.5, 7.50, step=0.25)).astype(int)

n1s = n_xor
n2s = n_rxor

ns = np.concatenate((n1s, n2s + n1s[-1]))
ls=['-', '--']
algorithms = ['Uncertainty Forest', 'Lifelong Forest']


TASK1='3 spirals'
TASK2='5 spirals'

fontsize=30
labelsize=27.5

colors = sns.color_palette("Set1", n_colors = 2)

fig1 = plt.figure(figsize=(8,8))
ax1 = fig1.add_subplot(1,1,1)
# for i, algo in enumerate(algorithms):
ax1.plot(ns, mean_error[0], label=algorithms[0], c=colors[1], ls=ls[np.sum(0 > 1).astype(int)], lw=3)
#ax1.fill_between(ns, 
#        mean_error[0] + 1.96*std_error[0], 
#        mean_error[0] - 1.96*std_error[0], 
#        where=mean_error[0] + 1.96*std_error[0] >= mean_error[0] - 1.96*std_error[0], 
#        facecolor=colors[1], 
#        alpha=0.15,
#        interpolate=True)

ax1.plot(ns, mean_error[1], label=algorithms[1], c=colors[0], ls=ls[np.sum(1 > 1).astype(int)], lw=3)
#ax1.fill_between(ns, 
#        mean_error[1] + 1.96*std_error[1, ], 
#        mean_error[1] - 1.96*std_error[1, ], 
#        where=mean_error[1] + 1.96*std_error[1] >= mean_error[1] - 1.96*std_error[1], 
#        facecolor=colors[0], 
#        alpha=0.15,
#        interpolate=True)

ax1.set_ylabel('Generalization Error (%s)'%(TASK1), fontname="Arial", fontsize=fontsize)
ax1.legend(loc='upper right', fontsize=20, frameon=False)
ax1.set_ylim(0.24, 0.57)
ax1.set_xlabel('Total Sample Size', fontname="Arial", fontsize=fontsize)
ax1.tick_params(labelsize=labelsize)
ax1.set_yticks([0.3, 0.4,.5])
ax1.set_xticks([250,750,1500])
ax1.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")

right_side = ax1.spines["right"]
right_side.set_visible(False)
top_side = ax1.spines["top"]
top_side.set_visible(False)

ax1.text(200, np.mean(ax1.get_ylim()), "%s"%(TASK1), fontsize=25)
ax1.text(900, np.mean(ax1.get_ylim()), "%s"%(TASK2), fontsize=25)

plt.tight_layout()

plt.savefig('./result/figs/generalization_error_3spiral.pdf',dpi=500)

#%%
mean_error = unpickle('result/mean_spiral.pickle')
std_error = unpickle('result/std_spiral.pickle')

algorithms = ['Uncertainty Forest', 'Lifelong Forest']

TASK1='3 spirals'
TASK2='5 spirals'

fig1 = plt.figure(figsize=(8,8))
ax1 = fig1.add_subplot(1,1,1)
# for i, algo in enumerate(algorithms):
ax1.plot(ns[len(n1s):], mean_error[2, len(n1s):], label=algorithms[0], c=colors[1], ls=ls[1], lw=3)
#ax1.fill_between(ns[len(n1s):], 
#        mean_error[2, len(n1s):] + 1.96*std_error[2, len(n1s):], 
#        mean_error[2, len(n1s):] - 1.96*std_error[2, len(n1s):], 
#        where=mean_error[2, len(n1s):] + 1.96*std_error[2, len(n1s):] >= mean_error[2, len(n1s):] - 1.96*std_error[2, len(n1s):], 
#        facecolor=colors[1], 
#        alpha=0.15,
#        interpolate=True)

ax1.plot(ns[len(n1s):], mean_error[3, len(n1s):], label=algorithms[1], c=colors[0], ls=ls[1], lw=3)
#ax1.fill_between(ns[len(n1s):], 
#        mean_error[3, len(n1s):] + 1.96*std_error[3, len(n1s):], 
#        mean_error[3, len(n1s):] - 1.96*std_error[3, len(n1s):], 
#        where=mean_error[3, len(n1s):] + 1.96*std_error[3, len(n1s):] >= mean_error[3, len(n1s):] - 1.96*std_error[3, len(n1s):], 
#        facecolor=colors[0], 
#        alpha=0.15,
#        interpolate=True)

ax1.set_ylabel('Generalization Error (%s)'%(TASK2), fontsize=fontsize)
ax1.legend(loc='upper right', fontsize=20, frameon=False)
#         ax1.set_ylim(-0.01, 0.22)
ax1.set_xlabel('Total Sample Size', fontsize=fontsize)
ax1.tick_params(labelsize=labelsize)
# ax1.set_yticks([0.15, 0.25, 0.35])
ax1.set_yticks([0.5,.6,.7])
ax1.set_xticks([250,750,1500])
ax1.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")

ax1.set_ylim(0.49, 0.72)

ax1.set_xlim(-10)
right_side = ax1.spines["right"]
right_side.set_visible(False)
top_side = ax1.spines["top"]
top_side.set_visible(False)

# ax1.set_ylim(0.14, 0.36)
ax1.text(200, np.mean(ax1.get_ylim()), "%s"%(TASK1), fontsize=25)
ax1.text(900, np.mean(ax1.get_ylim()), "%s"%(TASK2), fontsize=25)

plt.tight_layout()

plt.savefig('result/figs/generalization_error_5spiral.pdf',dpi=500)

#%%
mean_error = unpickle('result/mean_te_spiral.pickle')
std_error = unpickle('result/std_te_spiral.pickle')

algorithms = ['Forward Transfer', 'Backward Transfer']

TASK1='3 spirals'
TASK2='5 spirals'

fig1 = plt.figure(figsize=(8,8))
ax1 = fig1.add_subplot(1,1,1)

ax1.plot(ns, mean_error[0], label=algorithms[0], c=colors[0], ls=ls[0], lw=3)
#ax1.fill_between(ns, 
#        mean_error[0] + 1.96*std_error[0], 
#        mean_error[0] - 1.96*std_error[0], 
#        where=mean_error[1] + 1.96*std_error[0] >= mean_error[0] - 1.96*std_error[0], 
#        facecolor=colors[0], 
#        alpha=0.15,
#        interpolate=True)

ax1.plot(ns[len(n1s):], mean_error[1, len(n1s):], label=algorithms[1], c=colors[0], ls=ls[1], lw=3)
#ax1.fill_between(ns[len(n1s):], 
#        mean_error[1, len(n1s):] + 1.96*std_error[1, len(n1s):], 
#        mean_error[1, len(n1s):] - 1.96*std_error[1, len(n1s):], 
#        where=mean_error[1, len(n1s):] + 1.96*std_error[1, len(n1s):] >= mean_error[1, len(n1s):] - 1.96*std_error[1, len(n1s):], 
#        facecolor=colors[0], 
#        alpha=0.15,
#        interpolate=True)

ax1.set_ylabel('Transfer Efficiency', fontsize=fontsize)
ax1.legend(loc='upper right', fontsize=20, frameon=False)
ax1.set_ylim(0.92, 1.09)
ax1.set_xlabel('Total Sample Size', fontsize=fontsize)
ax1.tick_params(labelsize=labelsize)
ax1.set_yticks([0.95,1, 1.05])
ax1.set_xticks([250,750,1500])
ax1.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")
right_side = ax1.spines["right"]
right_side.set_visible(False)
top_side = ax1.spines["top"]
top_side.set_visible(False)
ax1.hlines(1, 50,1500, colors='gray', linestyles='dashed',linewidth=1.5)

ax1.text(200, np.mean(ax1.get_ylim())+.01, "%s"%(TASK1), fontsize=25)
ax1.text(900, np.mean(ax1.get_ylim())+.01, "%s"%(TASK2), fontsize=25)

plt.tight_layout()

plt.savefig('./result/figs/TE_spiral.pdf',dpi=500)

#%%
colors = sns.color_palette('Dark2', n_colors=3)

X, Y = generate_spirals(750, 2, 3, noise = 2.5)
Z, W = generate_spirals(750, 2, 5, noise = 2.5)

fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.scatter(X[:, 0], X[:, 1], c=get_colors(colors, Y), s=50)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title('3 spirals', fontsize=30)

plt.tight_layout()
ax.axis('off')
plt.savefig('./result/figs/spiral3.pdf')

#%%
colors = sns.color_palette('Dark2', n_colors=5)
fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.scatter(Z[:, 0], Z[:, 1], c=get_colors(colors, W), s=50)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title('5 spirals', fontsize=30)
ax.axis('off')
plt.tight_layout()
plt.savefig('./result/figs/spiral5.pdf')

# %%
