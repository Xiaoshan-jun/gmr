import numpy as np
import matplotlib.pyplot as plt
import random
import math
from math import floor

from sympy import *
# plot_implicit, symbols, Eq, And, Or

from scipy import stats



# from data_generation23 import get_data






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







def main():     # instant collision probability
    sep_dis_min = 10 #change it

    for time in range(240):
        name1 = 'F1T' + str(time) + '.npy'
        name2 = 'F2T' + str(time) + '.npy'
        f1t0 = np.load(name1)  # 100000by3
        f2t0 = np.load(name2)  # 1000by3

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
    
    
    
    
    
    
        # Monte carlo calculate probability of collision      instant probability
        cnt = 0
        for i in range(n_ds):
            if (dr[i,0]**2 + dr[i,1]**2 + dr[i,2]**2) <= sep_dis_min**2:
                cnt = cnt + 1
    
        prob_mc = (cnt*1.0)/n_ds
        print("prob of instant collision calc by MC", time , ":",prob_mc)  # 0.13




if __name__ == '__main__':
    main()
