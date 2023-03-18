# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 14:21:31 2022

@author: dekom
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from itertools import cycle
from gmr import GMM, kmeansplusplus_initialization, covariance_initialization
from gmr.utils import check_random_state
my_file = open("train.txt", "r")
data = []
with open("train.txt", "r") as f:
    for line in f:
        line = line.strip().split('\t')
        line = [float(i) for i in line]
        data.append(line)
trajectory1 = []
for j in range(4936):
    trajectory = []
    for i in range(20):
        trajectory.append(data[j * 20 + i][2])
        trajectory.append(data[j * 20 + i][3])
        trajectory.append(data[j * 20 + i][4])
    trajectory1.append(trajectory)
trajectory1 = np.array(trajectory1)
X_train= trajectory1[0:4000]
sample_observed= trajectory1[4500][0:30]
gt = trajectory1[4500][30:60]