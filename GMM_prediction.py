#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import BayesianGaussianMixture
from itertools import cycle
from gmr import GMM, kmeansplusplus_initialization, covariance_initialization
from gmr.utils import check_random_state
import os
from statistics import variance
import glob

#my_file = open("linear/train/train.txt", "r")
data = []
# _dir = os.path.dirname(__file__)
# _dir = _dir.split("/")[:-1]
# _dir = "/".join(_dir)
# _path = os.path.join(_dir, 'linear_char')
# all_files = os.listdir(_path)
# all_files = [os.path.join(_path, _path2) for _path2 in all_files]
all_files = glob.glob("linear_char/train.txt")
train_trajectory = []
trajectory1 = []
print(all_files)
for path in all_files:
    with open(path, "r") as f:
    #with open("vertical/train/train.txt", "r") as f:
    #with open("real/train/train.txt", "r") as f:
        for line in f:
            line = line.strip().split('\t')
            if(line[0] == "new"):
                if len(trajectory1) == 20:
                    train_trajectory.append(trajectory1)
                trajectory1 = []
                line = [float(i) for i in line[2:]]
                trajectory1.append(line)
            else:
                line = [float(i) for i in line[2:]]
                trajectory1.append(line)

print(train_trajectory)



X_train= np.array(train_trajectory)

random_state = check_random_state(0)
n_components = 20
initial_means = kmeansplusplus_initialization(X_train, n_components, random_state)
initial_covs = covariance_initialization(X_train, n_components)
bgmm = BayesianGaussianMixture(n_components=n_components, max_iter=100).fit(X_train)
gmm = GMM(
    n_components=n_components,
    priors=bgmm.weights_,
    means=bgmm.means_,
    covariances=bgmm.covariances_,
    random_state=random_state)

#with open("linear/vis/vis.txt", "r") as f:
#with open("vertical/vis/vis.txt", "r") as f:
data = []
all_files = glob.glob("linear_char/testdisrupt*.txt")
print(all_files)
for path in all_files:
    with open(path, "r") as f:
    #with open("vertical/train/train.txt", "r") as f:
    #with open("real/train/train.txt", "r") as f:
        for line in f:
            line = line.strip().split('\t')
            line = [float(i) for i in line[1:]]
            data.append(line)
#with open("real/vis/vis.txt", "r") as f:
    # for line in f:
    #     line = line.strip().split('\t')
    #     line = [float(i) for i in line]
    #     data.append(line)
trajectory1 = []
for j in range(100):
    trajectory = []
    for i in range(20):
        trajectory.append(data[j * 20 + i][2])
        trajectory.append(data[j * 20 + i][3])
        trajectory.append(data[j * 20 + i][4])
    trajectory1.append(trajectory)
trajectory1 = np.array(trajectory1)
dx = []
dy = []
dz = []
p = []
for i in range(100):
    sample_observed=trajectory1[i][0:30]
    
    true_traj= trajectory1[i][30:60]
    
    conditional_gmm = gmm.condition(list(range(30)), sample_observed)
    samples_prediction = conditional_gmm.sample(1)
    
    
    gt = trajectory1[i][30:60]
    obs_trajp = sample_observed
    pred_traj_gtp = gt
    pred_traj_fake = samples_prediction
    
    obs_trajp = sample_observed.T.reshape(10,3)
    #print(obs_trajp)
    pred_traj_gtp = gt.T.reshape(10,3)
    #print(pred_traj_gtp)
    pred_traj_fake = samples_prediction.T.reshape(10,3)
    #print(pred_traj_fake)
    
    diff = pred_traj_fake - pred_traj_gtp
    ade = []
    for i in range(10):
        dx.append(diff[i][0])
        dy.append(diff[i][1])
        dz.append(diff[i][2])
        s = diff[i][0]**2 + diff[i][1]**2 + diff[i][2]**2
        ade.append(s**0.5)
    p.append(ade)

    
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(projection='3d')
    # ax.set_title('GMR predict linear landing')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    #for i in range(10):
    x = obs_trajp.T[0]
    y = obs_trajp.T[1]
    z = obs_trajp.T[2] 
    # ax.scatter(x,y, z, s = 10, label= "observed trajectory", c = 'red')
    # ax.scatter(pred_traj_gtp.T[0],pred_traj_gtp.T[1], pred_traj_gtp.T[2], s = 10, label= "real trajectory", c = 'orange')
    # ax.scatter(pred_traj_fake.T[0],pred_traj_fake.T[1], pred_traj_fake.T[2], s = 10, label= "predict trajectory", c = 'blue')
    #         #print(pred_traj_fake)
    # ax.plot(x,y, z, c = 'red')
    # ax.plot(pred_traj_gtp.T[0],pred_traj_gtp.T[1], pred_traj_gtp.T[2],  c = 'orange')
    # ax.plot(pred_traj_fake.T[0],pred_traj_fake.T[1], pred_traj_fake.T[2],  c = 'blue')
    # ax.legend()
for i in range(10):
    l = []
    for k in range(100):
        l.append(p[k][i])
    print(str(round(np.mean(l),2)) + '$\pm$' + str(round(variance(l)**0.5,2)))



# error=samples_prediction-true_traj

