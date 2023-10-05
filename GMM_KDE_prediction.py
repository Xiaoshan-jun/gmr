#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
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

def myKDE(data, grid_points):
    # grid_custom = np.linspace(-100, 100, 64)
    # kde_my = FFTKDE(bw=0.2).fit(data)
    # kde_my.evaluate(grid_custom)

    scale_h = 0.15
    scale_v = 0.15  #0.3
    scale_a = 0.15
    Xmin = data[:, 0].min() - (data[:, 0].max() - data[:, 0].min()) * scale_h
    Xmax = data[:, 0].max() + (data[:, 0].max() - data[:, 0].min()) * scale_h
    Ymin = data[:, 1].min() - (data[:, 1].max() - data[:, 1].min()) * scale_v
    Ymax = data[:, 1].max() + (data[:, 1].max() - data[:, 1].min()) * scale_v
    Zmin = data[:, 2].min() - (data[:, 2].max() - data[:, 2].min()) * scale_a
    Zmax = data[:, 2].max() + (data[:, 2].max() - data[:, 2].min()) * scale_a

    datarange = (Xmin, Xmax, Ymin, Ymax, Zmin, Zmax)

    grid_points_x = np.linspace(Xmin, Xmax, num = grid_points)
    grid_points_y = np.linspace(Ymin, Ymax, num = grid_points)
    grid_points_z = np.linspace(Zmin, Zmax, num = grid_points)

    X, Y, Z = np.mgrid[Xmin:Xmax:grid_points*1j, Ymin:Ymax:grid_points*1j, Zmin:Zmax:grid_points*1j]  # capital letter Z = traditional kde
    positions = np.vstack([ X.ravel(), Y.ravel(), Z.ravel() ])
    kernel = stats.gaussian_kde(data.T, bw_method=0.1)
    den = np.reshape(kernel(positions).T, X.shape)  #  density   scipy green
    # print("3: obtained traditional kde")

    # # start_findmidval = time.time()
    # mat_scipy, critical_kdeval_scipy = bisectionSearch(den, level, error_level)   # int matrix, scalar
    # # end_findmidval = time.time()
    # grids_in_cregion = int(np.around(np.sum(mat_scipy)))  # number of grids in confidence region, total grids = grid_points^2

    return datarange, X, Y, Z, den

#------------------------------------------------------TO DO-------------------------------
#------------------------------------------------------training file and component --------
#DATASET1 #train, keep same for each test file
all_files = glob.glob("dataset/trajectory_linear/train.txt")  #training
agent1_files = glob.glob("dataset/trajectory_linear/agent1.txt")  #agent1
agent2_files = glob.glob("dataset/trajectory_linear/agent2.txt")  #agent2
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
# all_files = glob.glob("dataset/trajectory_real_smooth/train.txt")
# agent1_files = glob.glob("dataset/tajectory_real_smooth/agent1.txt")  #agent1
# agent2_files = glob.glob("dataset/tajectory_real_smooth/agent2.txt")  #agent2
# all_files2 = glob.glob("dataset/trajectory_real_smooth/val/testgt*.txt")
#all_files2 = glob.glob("dataset/trajectory_real_smooth/val/testdisrupt*.txt")
#all_files2 = glob.glob("dataset/trajectory_real_smooth/val/testmusk*.txt")
#all_files2 = glob.glob("dataset/trajectory_real_smooth/val/testpointmusk*.txt")
# xt = 0.15
# yt = 0.15
# zt = 0.15
#DATASET5
day = 1
filename = 'dataset/cmu_original/7day' + str(day) + 'train.txt'
all_files = glob.glob(filename)
filename = 'dataset/cmu_original/7days' + str(day) + '/val/testgt*.txt'
all_files2 = glob.glob(filename)
past = 23
prediction = 120
train = 1
sep_dis_min = 1.5 #change it

n_components = 150
filename = '7daywights' + str(day) + str(n_components) + '.npy'
saveweights = filename
filename = '7daymeans' + str(day) + str(n_components) + '.npy'
savemeans = filename
filename = '7daycovariances' + str(day) + str(n_components) + '.npy'
savecovariances = filename
loadweights = '7daywights' + str(day) + str(n_components) + '.npy'
loadmeans = '7daymeans' + str(day) + str(n_components) + '.npy'
loadcovariances = '7daycovariances' + str(day) + str(n_components) + '.npy'

#------------------------------------------------------TO DO-------------------------------
#---------------------------------------------------do not change----------------------------
#-----------------------------------------training section--------------------------------
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
                    #line2 = [float(i) for i in line[4:]]
                    #line2.append(0)
                    trajectory1.append(line1)
                    #trajectory1.append(line2)
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
#------------------------------trajectory prediction section--------------------------------------------------
data = []
t0 = time.time()
#all_files = glob.glob("trajectory_real/val_smooth/testpointmusk*.txt")
test_trajectory = []
trajectory1 = []
#print(all_files2)
for i in range(1):
    path1 = all_files2[i]  #F1
    with open(path1, "r") as f:
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
    agent1= np.array(test_trajectory)
    old_shape = agent1.shape
    agent1 = agent1.reshape(old_shape[0],old_shape[1] * old_shape[2])
    random_state = check_random_state(0)
    sample_observed=agent1[0][0:3*past]
    true_traj1 = agent1[0][3*past:3*past + 3*prediction]
    true_traj1 = true_traj1.T.reshape(prediction,3)
    conditional_gmm = gmm.condition(list(range(3*past)), sample_observed)
    F1 = conditional_gmm.sample(1000)
    MSE = []
    for j in range(i+1, len(all_files2)): #F2
        test_trajectory = []
        trajectory1 = []
        path2 = all_files2[j]
        with open(path2, "r") as f:
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
        agent2= np.array(test_trajectory)
        old_shape = agent2.shape
        agent2 = agent2.reshape(old_shape[0],old_shape[1] * old_shape[2])
        random_state = check_random_state(0)
        sample_observed=agent2[0][0:3*past] 
        true_traj2 = agent2[0][3*past:3*past + 3*prediction]
        true_traj2 = true_traj2.T.reshape(prediction,3)
        
#-------------------------------------------------------calculate true collision---------------------------------------------
        crt = []
        for time in range(240):
            diff = (true_traj2[time,:] - true_traj1[time,:])**2
            if np.sqrt(np.sum(diff)) < sep_dis_min:
                crt.append(1)
            else:
                crt.append(0)
        conditional_gmm = gmm.condition(list(range(3*past)), sample_observed)
        F2 = conditional_gmm.sample(1000)

#------------------------------KDE section--------------------------------------------------
        cr = []
        for time in range(240):
            f1t0 = F1[:, time*3:time*3 + 3]  # 100000by3
            f2t0 = F2[:, time*3:time*3 + 3]  # 1000by3
    
            dr = f1t0 - f2t0
        
        
            n_ds = f1t0.shape[0]
        
            
        
        
            grid_points = 20
        
        
        
            # kde
            datarange, X1, Y1, Z1, den = myKDE(dr, grid_points)  # kde for random vector dr
        
            xgrid, ygrid, zgrid = np.mgrid[datarange[0]:datarange[1]:grid_points * 1j, datarange[2]:datarange[3]:grid_points * 1j, datarange[4]:datarange[5]:grid_points * 1j]
            # xgrid = xgrid.flatten()
            # ygrid = ygrid.flatten()
            grid_points_x = np.linspace(datarange[0], datarange[1], num=grid_points)
            grid_points_y = np.linspace(datarange[2], datarange[3], num=grid_points)
            grid_points_z = np.linspace(datarange[4], datarange[5], num=grid_points)
        
        
        
        
        
        
        
            # collision prob kde
            mat_bin = np.zeros((grid_points, grid_points, grid_points))
            for i in range(0,grid_points):
                for j in range(0,grid_points):
                    for k in range(0,grid_points):
                        if (grid_points_x[i] ** 2 + grid_points_y[j]** 2 + grid_points_z[k]** 2) < sep_dis_min**2:
                            mat_bin[i][j] = 1
        
            mat_select = np.multiply(den, mat_bin)
            prob_kde = np.sum(mat_select) * 1.0 / np.sum(den)
            print("prob of instant collision calc by KDE", time, ":" ,prob_kde)    # 0.14
            cr.append(prob_kde)
        
        
        
        
        
            # Monte carlo calculate probability of collision      instant probability
            cnt = 0
            for i in range(n_ds):
                if (dr[i,0]**2 + dr[i,1]**2 + dr[i,2]**2) <= sep_dis_min**2:
                    cnt = cnt + 1
        
            prob_mc = (cnt*1.0)/n_ds
            print("prob of instant collision calc by MC", time , ":",prob_mc)  # 0.13
        indices = np.arange(len(cr))
        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the values against indices
        ax.plot(indices, cr, marker='o', linestyle='-', color='b', label='KDE')
        ax.plot(indices, crt, marker='o', linestyle='-', color='r', label='True')
        # Set labels and title
        ax.set_xlabel('time')
        ax.set_ylabel('collision rate')
        ax.set_title('collision rate in each time step ' + str(day))

        # Add grid and legend
        ax.grid()
        ax.legend()
        loss = crt - cr
        loss = diff**2
        mse = np.mean(loss)
        MSE.append(mse)
        # Show the plot
        plt.show()
#----------------------------



#-------------------------------------------do not change----------------------------