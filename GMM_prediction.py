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
all_files = glob.glob("dataset/trajectory_linear/train.txt") 
all_files2 = glob.glob("dataset/trajectory_linear/val/testgt*.txt") 
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
# all_files = glob.glob("dataset/trajectory_linear_raylr/train.txt")
#all_files2 = glob.glob("dataset/trajectory_linear_raylr/val/testgt*.txt")
#all_files2 = glob.glob("dataset/trajectory_linear_raylr/val/testdisrupt*.txt")
#all_files2 = glob.glob("dataset/trajectory_linear_raylr/val/testmusk*.txt")
# all_files2 = glob.glob("dataset/trajectory_linear_raylr/val/testpointmusk*.txt")
# xt = 25
# yt = 25
# zt = 9
#DATASET5
# all_files = glob.glob("dataset/trajectory_vertical_raylr/train.txt")
#all_files2 = glob.glob("dataset/trajectory_vertical_raylr/val/testgt*.txt")
# all_files2 = glob.glob("dataset/trajectory_vertical_raylr/val/testdisrupt*.txt")
# all_files2 = glob.glob("dataset/trajectory_vertical_raylr/val/testmusk*.txt")
# all_files2 = glob.glob("dataset/trajectory_vertical_raylr/val/testpointmusk*.txt")
# xt = 25
# yt = 25
# zt = 9
#DATASET6
# all_files = glob.glob("dataset/trajectory_real_smooth_raylr/train.txt")
# all_files2 = glob.glob("dataset/trajectory_real_smooth_raylr/val/testgt*.txt")
# all_files2 = glob.glob("dataset/trajectory_real_smooth_raylr/val/testdisrupt*.txt")
# all_files2 = glob.glob("dataset/trajectory_real_smooth_raylr/val/testmusk*.txt")
# all_files2 = glob.glob("dataset/trajectory_real_smooth_raylr/val/testpointmusk*.txt")
# xt = 0.15
# yt = 0.15
# zt = 0.15
#n_components = 8, 20, 30, 40, 50, 60, 70, 80, 90, 100
n_components = 8
#------------------------------------------------------TO DO-------------------------------
#---------------------------------------------------do not change----------------------------
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
                if len(trajectory1) == 20:
                    train_trajectory.append(trajectory1)
                # else:
                #     for i in range(20-len(trajectory1)):
                #         trajectory1.append([0, 0, 0])
                #     test_trajectory.append(trajectory1)
                trajectory1 = []
                line = [float(i) for i in line[1:]]
                trajectory1.append(line)
            else:
                line = [float(i) for i in line[1:]]
                trajectory1.append(line)


X_train= np.array(train_trajectory)
old_shape = X_train.shape
X_train = X_train.reshape(old_shape[0],old_shape[1] * old_shape[2])
random_state = check_random_state(0)

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

t0 = time.time()
#all_files = glob.glob("trajectory_real/val_smooth/testpointmusk*.txt")
test_trajectory = []
trajectory1 = []
#print(all_files2)
for path in all_files2:
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
                line = [float(i) for i in line[1:]]
                trajectory1.append(line)
            else:
                line = [float(i) for i in line[1:]]
                trajectory1.append(line)
test_trajectory.append(trajectory1)
X_train
dx = []
dy = []

dz = []
ade = []
fde = []
X_test= np.array(test_trajectory)
old_shape = X_test.shape
X_test = X_test.reshape(old_shape[0],old_shape[1] * old_shape[2])
random_state = check_random_state(0)
miss = 0
for i in range(1):
    sample_observed=X_test[0][0:30]
    
    true_traj= X_test[0][30:60]
    
    conditional_gmm = gmm.condition(list(range(30)), sample_observed)
    t0 = time.time()
    samples_prediction = conditional_gmm.sample(1000)
    print('generating time:')
    gtime = time.time() - t0
    print(gtime)
    F1T0 = samples_prediction[:,0:3]
    F1T1 = samples_prediction[:,3:6]
    F1T2 = samples_prediction[:,6:9]
    F1T3 = samples_prediction[:,9:12]
    F1T4 = samples_prediction[:,12:15]
    F1T5 = samples_prediction[:,15:18]
    F1T6 = samples_prediction[:,18:21]
    F1T7 = samples_prediction[:,21:24]
    F1T8 = samples_prediction[:,24:27]
    F1T9 = samples_prediction[:,27:30]
    np.save('F1T0.npy', F1T0)
    np.save('F1T1.npy', F1T1)
    np.save('F1T2.npy', F1T2)
    np.save('F1T3.npy', F1T3)
    np.save('F1T4.npy', F1T4)
    np.save('F1T5.npy', F1T5)
    np.save('F1T6.npy', F1T6)
    np.save('F1T7.npy', F1T7)
    np.save('F1T8.npy', F1T8)
    np.save('F1T9.npy', F1T9)
    gt = X_test[i][30:60]
    obs_trajp = sample_observed
    pred_traj_gtp = gt
    pred_traj_fake = samples_prediction
for i in range(1):
    sample_observed=X_test[20][0:30]
    
    true_traj= X_test[20][30:60]
    
    conditional_gmm = gmm.condition(list(range(30)), sample_observed)
    t0 = time.time()
    samples_prediction = conditional_gmm.sample(1000)
    print('generating time:')
    gtime = time.time() - t0
    print(gtime)
    F2T0 = samples_prediction[:,0:3]
    F2T1 = samples_prediction[:,3:6]
    F2T2 = samples_prediction[:,6:9]
    F2T3 = samples_prediction[:,9:12]
    F2T4 = samples_prediction[:,12:15]
    F2T5 = samples_prediction[:,15:18]
    F2T6 = samples_prediction[:,18:21]
    F2T7 = samples_prediction[:,21:24]
    F2T8 = samples_prediction[:,24:27]
    F2T9 = samples_prediction[:,27:30]
    np.save('F2T0.npy', F2T0)
    np.save('F2T1.npy', F2T1)
    np.save('F2T2.npy', F2T2)
    np.save('F2T3.npy', F2T3)
    np.save('F2T4.npy', F2T4)
    np.save('F2T5.npy', F2T5)
    np.save('F2T6.npy', F2T6)
    np.save('F2T7.npy', F2T7)
    np.save('F2T8.npy', F2T8)
    np.save('F2T9.npy', F2T9)
    gt = X_test[i][30:60]
    obs_trajp = sample_observed
    pred_traj_gtp = gt
    pred_traj_fake = samples_prediction
    

    
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

print('number of test' + str(len(X_test)))
print('ade: '+  str(round(np.mean(ade),2)) + '$\pm$' + str(round(np.var(ade)**0.5,2)))
print('fde: '+ str(round(np.mean(fde),2)) + '$\pm$' + str(round(np.var(fde)**0.5,2)))
# print('generating time:')
# gtime = time.time() - t0
# print(gtime)
# print('AGT:')
# print(gtime/(len(X_test)*10))
print('miss rate:')
print(miss/(len(X_test)*10))
#-------------------------------------------do not change----------------------------