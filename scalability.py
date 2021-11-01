#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# System integration
import os
import sys
import abc

# Standard python libraries
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp

def plot_scalability(Ntrains, results, title, savefig=False):

    colors=mcp.gen_color(cmap="viridis",n=9)

    outputs = ['Blast 1', 'Blast 2', 'Blast 4', 'Blast 10', 
                'Therm 2', 'Therm 3', 'Therm 4', 'Therm 6']

    fig = plt.figure()
    plt.loglog(Ntrains[:], (results[:,0]), color=colors[1], label=outputs[0])
    plt.loglog(Ntrains[:], (results[:,1]), color=colors[3], label=outputs[1])
    plt.loglog(Ntrains[:], (results[:,2]), color=colors[5], label=outputs[2])
    plt.loglog(Ntrains[:], (results[:,3]), color=colors[7], label=outputs[3])
    plt.ylabel(title)
    plt.xlabel('Number of training points')
    plt.legend()
    plt.grid(True, which='both')
    plt.draw()
    if savefig:
        plt.show(block=False)
        fig.savefig('./results/scalability/Blast_' + title + ".pdf", bbox_inches="tight")
    else:
        plt.show()
    plt.close()


    fig = plt.figure()
    # for k, color in enumerate(colors):
    plt.loglog(Ntrains[:], (results[:,4]), color=colors[1], label=outputs[4])
    plt.loglog(Ntrains[:], (results[:,5]), color=colors[3], label=outputs[5])
    plt.loglog(Ntrains[:], (results[:,6]), color=colors[5], label=outputs[6])
    plt.loglog(Ntrains[:], (results[:,7]), color=colors[7], label=outputs[7])
        # plt.loglog(Ntrains[:], (results[:,4+k]), color=color, label=outputs[4+k])
    # plt.title(title + ' on the test set')
    plt.ylabel(title)
    plt.xlabel('Number of training points')
    plt.legend()
    plt.grid(True, which='both')
    plt.draw()
    if savefig:
        plt.show(block=False)
        fig.savefig('./results/scalability/Therm_' + title + ".pdf", bbox_inches="tight")
    else:
        plt.show()
    plt.close()


    return

Ntrain_list = [1000, 10000, 100000, 1000000]

data_1e3 = pd.read_csv('./results/Ntrain_1e+03/Summary_of_results.csv')
data_1e4 = pd.read_csv('./results/Ntrain_1e+04/Summary_of_results.csv')
data_1e5 = pd.read_csv('./results/Ntrain_1e+05/Summary_of_results.csv')
data_1e6 = pd.read_csv('./results/Ntrain_1e+06/Summary_of_results.csv')

accuracies = np.concatenate(
    (data_1e3.iloc[:,2:3].values, 
    data_1e4.iloc[:,2:3].values, 
    data_1e5.iloc[:,2:3].values, 
    data_1e6.iloc[:,2:3].values), axis=1).T
MAEs = np.concatenate(
    (data_1e3.iloc[:,6:7].values, 
    data_1e4.iloc[:,6:7].values, 
    data_1e5.iloc[:,6:7].values, 
    data_1e6.iloc[:,6:7].values), axis=1).T
MREs = np.concatenate(
    (data_1e3.iloc[:,7:8].values, 
    data_1e4.iloc[:,7:8].values, 
    data_1e5.iloc[:,7:8].values, 
    data_1e6.iloc[:,7:8].values), axis=1).T
MEDREs = np.concatenate(
    (data_1e3.iloc[:,8:9].values, 
    data_1e4.iloc[:,8:9].values, 
    data_1e5.iloc[:,8:9].values, 
    data_1e6.iloc[:,8:9].values), axis=1).T

if os.path.exists('./results/scalability') == False:
    os.mkdir('./results/scalability')

# plot_scalability(Ntrain_list, 1 - 0.01 * accuracies, 'Missclassification rate', savefig=False)
plot_scalability(Ntrain_list, 0.001 * MAEs, 'Mean absolute error [km]', savefig=True)
# plot_scalability(Ntrain_list, 0.01 * MREs, 'MRE', savefig=False)
# plot_scalability(Ntrain_list, 0.01 * MEDREs, 'MedRE', savefig=False)