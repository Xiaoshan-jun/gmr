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

#------------------------------------------------------TO DO-------------------------------
#------------------------------------------------------training file and component --------
#DATASET1 #train, keep same for each test file
# all_files = glob.glob("dataset/trajectory_linear/train.txt") 
#all_files2 = glob.glob("dataset/trajectory_linear/val/testgt*.txt") 
#all_files2 = glob.glob("dataset/trajectory_linear/val/testdisrupt*.txt")
# all_files2 = glob.glob("dataset/trajectory_linear/val/testmusk*.txt")
# all_files2 = glob.glob("dataset/trajectory_linear/val/testpointmusk*.txt")
# xt = 25
# yt = 25
# zt = 9
#DATASET2
#all_files = glob.glob("dataset/trajectory_vertical/train.txt")
#all_files2 = glob.glob("dataset/trajectory_vertical/val/testgt*.txt")
#all_files2 = glob.glob("dataset/trajectory_vertical/val/testdisrupt*.txt")
#all_files2 = glob.glob("dataset/trajectory_vertical/val/testmusk*.txt")
#all_files2 = glob.glob("dataset/trajectory_vertical/val/testpointmusk*.txt")
# xt = 25
# yt = 25
# zt = 9
#DATASET3
#all_files = glob.glob("dataset/trajectory_real_smooth/train.txt")
# all_files2 = glob.glob("dataset/trajectory_real_smooth/val/testgt*.txt")
#all_files2 = glob.glob("dataset/trajectory_real_smooth/val/testdisrupt*.txt")
#all_files2 = glob.glob("dataset/trajectory_real_smooth/val/testmusk*.txt")
#all_files2 = glob.glob("dataset/trajectory_real_smooth/val/testpointmusk*.txt")
xt = 0.15
yt = 0.15
zt = 0.15
#DATASET4
# all_files = glob.glob("dataset/cmu/train.txt")
# all_files2 = glob.glob("dataset/cmu/val/testgt*.txt")
#DATASET5
day = 4
filename = 'dataset/cmu_original/7day' + str(day) + 'train.txt'
all_files = glob.glob(filename)
filename = 'dataset/cmu_original/7days' + str(day) + '/val/testgt*.txt'
all_files2 = glob.glob(filename)
past = 11
prediction = 12
train = 1


n_components = 300
filename = '7daywights' + str(day) + str(n_components) + '.npy'
saveweights = filename
filename = '7daymeans' + str(day) + str(n_components) + '.npy'
savemeans = filename
filename = '7daycovariances' + str(day) + str(n_components) + '.npy'
savecovariances = filename
loadweights = 'wights1.npy'
loadmeans = 'means1.npy'
loadcovariances = 'covariances1.npy'
#------------------------------------------------------TO DO-------------------------------
#---------------------------------------------------do not change----------------------------
if train:
    t0 = time.time()
    data = []
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
                    if len(trajectory1) == past + prediction:
                        train_trajectory.append(trajectory1)
                    # else:
                    #     for i in range(20-len(trajectory1)):
                    #         trajectory1.append([0, 0, 0])
                    #     test_trajectory.append(trajectory1)
                    trajectory1 = []
                    line1 = [float(i) for i in line[1:4]]
                    line2 = [float(i) for i in line[4:]]
                    line2.append(0)
                    trajectory1.append(line1)
                    trajectory1.append(line2)
                else:
                    line = [float(i) for i in line[1:]]
                    trajectory1.append(line)
    
    print('Loading time:')
    gtime = time.time() - t0
    print(gtime)
    t0 = time.time()
    X_train= np.array(train_trajectory)
    old_shape = X_train.shape
    X_train = X_train.reshape(old_shape[0],old_shape[1] * old_shape[2])
    random_state = check_random_state(0)
    
    initial_means = kmeansplusplus_initialization(X_train, n_components, random_state)
    initial_covs = covariance_initialization(X_train, n_components)
    bgmm = BayesianGaussianMixture(n_components=n_components, max_iter=100).fit(X_train)
    np.save(saveweights, bgmm.weights_)
    np.save(savemeans, bgmm.means_)
    np.save(savecovariances, bgmm.covariances_)
    print('number of train data: ' +  str(len(X_train)))
    gmm = GMM(
        n_components=n_components,
        priors=bgmm.weights_,
        means=bgmm.means_,
        covariances=bgmm.covariances_,
        random_state=random_state)
    print('train time:')
    gtime = time.time() - t0
    print(gtime)
else:
    weights = np.load(loadweights)
    means = np.load(loadmeans)
    covariances = np.load(loadcovariances)
    random_state = check_random_state(0)
    gmm = GMM(
        n_components=n_components,
        priors=weights,
        means=means,
        covariances=covariances,
        random_state=random_state)

data = []
t0 = time.time()
#all_files = glob.glob("trajectory_real/val_smooth/testpointmusk*.txt")
test_trajectory = []
trajectory1 = []
#print(all_files2)
for path in all_files2:
    with open(path, "r") as f:
        for line in f:
            line = line.strip().split('\t')
            if(line[0] == "new"):
                if len(trajectory1) == past + prediction:
                    test_trajectory.append(trajectory1)
                # else:
                #     for i in range(20-len(trajectory1)):
                #         trajectory1.append([0, 0, 0])
                #     test_trajectory.append(trajectory1)
                trajectory1 = []
                line1 = [float(i) for i in line[1:4]]
                line2 = [float(i) for i in line[4:]]
                line2.append(0)
                trajectory1.append(line1)
                trajectory1.append(line2)
            else:
                line = [float(i) for i in line[1:]]
                trajectory1.append(line)
test_trajectory.append(trajectory1)
dx = []
dy = []

dz = []
ade = []
fde = []
X_test= np.array(test_trajectory)
print('number of test' + str(len(X_test)))
old_shape = X_test.shape
X_test = X_test.reshape(old_shape[0],old_shape[1] * old_shape[2])
random_state = check_random_state(0)
miss = 0
for i in range(len(X_test)):
    sample_observed=X_test[i][0:3*past]
    
    true_traj= X_test[i][3*past:3*past + 3*prediction]
    
    conditional_gmm = gmm.condition(list(range(3*past)), sample_observed)
    samples_prediction = conditional_gmm.sample(5)
    
    gt = X_test[i][3*past:3*past + 3*prediction]
    obs_trajp = sample_observed
    pred_traj_gtp = gt
    ade1 = 1000
    fde1 = 1000
    for x in range(5):
        adet = []
        pred_traj_fake = samples_prediction[x]
        
        obs_trajp = sample_observed.T.reshape(past,3)
        #print(obs_trajp)
        pred_traj_gtp = gt.T.reshape(prediction,3)
        #print(pred_traj_gtp)
        pred_traj_fake = pred_traj_fake.T.reshape(prediction,3)
        #print(pred_traj_fake)
        
        diff = pred_traj_fake - pred_traj_gtp
        loss = diff**2
        loss = np.sqrt(np.sum(loss,1))
        if np.mean(loss) < ade1:
            ade1 = np.mean(loss)
        diff = (pred_traj_fake[-1,:] - pred_traj_gtp[-1,:])**2
        if np.sqrt(np.sum(diff)) < fde1:
            fde1 = np.sqrt(np.sum(diff))
    ade.append(ade1)
    fde.append(fde1)

    
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



print('ade: '+  str(round(np.mean(ade),2)) + '$\pm$' + str(round(np.var(ade)**0.5,2)))
print('fde: '+ str(round(np.mean(fde),2)) + '$\pm$' + str(round(np.var(fde)**0.5,2)))
print('generating time:')
gtime = time.time() - t0
print(gtime)
# print('AGT:')
# print(gtime/(len(X_test)*10))
print('miss rate:')
print(miss/(len(X_test)*prediction))
#-------------------------------------------do not change----------------------------