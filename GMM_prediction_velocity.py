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
import time

data = []
#all_files = glob.glob("linear_char/train.txt")
#all_files = glob.glob("vertical_char/train.txt")
all_files = glob.glob("trajectory_real/realdata2_train.txt")
#all_files = glob.glob("trajectory_real/realdata2_train_smooth.txt")
train_trajectory = []
trajectory1 = []
print(all_files)
for path in all_files:
    with open(path, "r") as f:
    #with open("vertical/train/train.txt", "r") as f:
    #with open("real/train/train.txt", "r") as f:
        count = 0;
        for line in f:
            count+= 1
            if count > 20000:
                break
            line = line.strip().split('\t')
            if(line[0] == "new"):
                # else:
                #     for i in range(20-len(trajectory1)):
                #         trajectory1.append([0, 0, 0])
                #     train_trajectory.append(trajectory1)
                trajectory1 = []
                line = [float(i) for i in line[2:]]
                trajectory1.append(line)
            elif(line[0] == "future"):
                line = [float(i) for i in line[2:]]
                last = trajectory1[-1]
                velocity = []
                for i in range(len(line)):
                    velocity.append(line[i] - last[i])
                trajectory1.append(velocity)
                train_trajectory.append(np.array(trajectory1))
                trajectory1.pop(0)
                trajectory1.pop()
                trajectory1.append(line)
            else:
                line = [float(i) for i in line[2:]]
                trajectory1.append(line)


X_train= np.array(train_trajectory)
old_shape = X_train.shape
X_train = X_train.reshape(old_shape[0],old_shape[1] * old_shape[2])
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

data = []
#all_files = glob.glob("linear_char/val/testgt*.txt")
#all_files = glob.glob("linear_char/val/testdisrupt*.txt")
#all_files = glob.glob("linear_char/val/testmusk*.txt")
#all_files = glob.glob("linear_char/val/testpointmusk*.txt")
#all_files = glob.glob("vertical_char/val/testgt*.txt")
#all_files = glob.glob("vertical_char/val/testdisrupt*.txt")
#all_files = glob.glob("vertical_char/val/testmusk*.txt")
#all_files = glob.glob("vertical_char/val/testpointmusk*.txt")
all_files = glob.glob("trajectory_real/val/testgt*.txt")
#all_files = glob.glob("trajectory_real/val/testdisrupt*.txt")
#all_files = glob.glob("trajectory_real/val/testmusk*.txt")
#all_files = glob.glob("trajectory_real/val/testpointmusk*.txt")
#all_files = glob.glob("trajectory_real/val_smooth/testgt*.txt")
#all_files = glob.glob("trajectory_real/val_smooth/testdisrupt*.txt")
#all_files = glob.glob("trajectory_real/val_smooth/testmusk*.txt")
#all_files = glob.glob("trajectory_real/val_smooth/testpointmusk*.txt")
t0 = time.time()
test_trajectory = []
trajectory1 = []
for path in all_files:
    with open(path, "r") as f:
    #with open("vertical/train/train.txt", "r") as f:
    #with open("real/train/train.txt", "r") as f:
        for line in f:
            line = line.strip().split('\t')
            if(line[0] == "new"):
                if len(trajectory1) == 20:
                    test_trajectory.append(trajectory1)
                # else:
                #     for i in range(20-len(trajectory1)):
                #         trajectory1.append([0, 0, 0])
                #     test_trajectory.append(trajectory1)
                trajectory1 = []
                line = [float(i) for i in line[2:]]
                trajectory1.append(line)
            else:
                line = [float(i) for i in line[2:]]
                trajectory1.append(line)

dx = []
dy = []
dz = []
ade = []
fde = []
X_test= np.array(test_trajectory[1:])
old_shape = X_test.shape
X_test = X_test.reshape(old_shape[0],old_shape[1] * old_shape[2])
random_state = check_random_state(0)
miss = 0
for i in range(len(X_test)):
    sample_observed=X_test[i][0:30]
    
    true_traj= X_test[i][30:60]
    pred_traj_fake = []
    for j in range(10):
        conditional_gmm = gmm.condition(list(range(30)), sample_observed)
        samples_prediction = conditional_gmm.sample(1)
        samples_prediction = samples_prediction.T
        newpoint = []
        newpoint.append(sample_observed[27] + samples_prediction[0])
        newpoint.append(sample_observed[28] + samples_prediction[1])
        newpoint.append(sample_observed[29] + samples_prediction[2])
        newpoint = np.array(newpoint).flatten()
        pred_traj_fake.append(newpoint)
        sample_observed = sample_observed[3:]
        sample_observed = np.concatenate((sample_observed, newpoint))
    gt = X_test[i][30:60]
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
    for j in range(10):
        dx.append(diff[j][0])
        dy.append(diff[j][1])
        dz.append(diff[j][2])
        if abs(diff[j][0]) > 0.05 or abs(diff[j][1]) > 0.05 or abs(diff[j][2]) > 0.05:
            miss += 1
        s = diff[j][0]**2 + diff[j][1]**2 + diff[j][2]**2
        #print(str(j) + ':' + str(np.sqrt(s)))
        ade.append(np.sqrt(s))
        if j == 9:
            fde.append(np.sqrt(s))

    
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(projection='3d')
    # ax.set_title('GMR predict linear landing')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    #for i in range(10):
    # x = obs_trajp.T[0]
    # y = obs_trajp.T[1]
    # z = obs_trajp.T[2]
    # ax.scatter(x,y, z, s = 10, label= "observed trajectory", c = 'red')
    # ax.scatter(pred_traj_gtp.T[0],pred_traj_gtp.T[1], pred_traj_gtp.T[2], s = 10, label= "real trajectory", c = 'orange')
    # ax.scatter(pred_traj_fake.T[0],pred_traj_fake.T[1], pred_traj_fake.T[2], s = 10, label= "predict trajectory", c = 'blue')
    #         #print(pred_traj_fake)
    # ax.plot(x,y, z, c = 'red')
    # ax.plot(pred_traj_gtp.T[0],pred_traj_gtp.T[1], pred_traj_gtp.T[2],  c = 'orange')
    # ax.plot(pred_traj_fake.T[0],pred_traj_fake.T[1], pred_traj_fake.T[2],  c = 'blue')
    # ax.legend()


# error=samples_prediction-true_traj
print(ade)
print(fde)
print('number of test' + str(len(X_test)))
print(str(round(np.mean(ade),2)) + '$\pm$' + str(round(np.var(ade)**0.5,2)))
print(str(round(np.mean(fde),2)) + '$\pm$' + str(round(np.var(fde)**0.5,2)))
print('generating time:')
gtime = time.time() - t0
print(gtime)
print('AGT:')
print(gtime/(len(X_test)*10))
print('miss rate:')
print(miss/(len(X_test)*10))