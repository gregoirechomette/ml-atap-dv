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

def plot_scalability(Ntrains, results, title, savefig=False):
    
    colors_linspace = np.linspace(0, 1, 4)
    colors = [plt.cm.seismic(x) for x in colors_linspace]

    outputs = ['BlastRad1', 'BlastRad2', 'BlastRad4', 'BlastRad10', 
                'ThermRad2', 'ThermRad3', 'ThermRad4', 'ThermRad6']

    fig = plt.figure()
    for k, color in enumerate(colors):
        plt.loglog(Ntrains[:], (results[:,k]), color=color, label=outputs[k])
    plt.title(title + ' on the test set')
    plt.ylabel(title)
    plt.xlabel('Number of training points')
    plt.legend()
    plt.grid(True, which='both')
    plt.draw()
    if savefig:
        plt.show(block=False)
        fig.savefig('./Blast_' + title + ".png", bbox_inches="tight")
    else:
        plt.show()
    plt.close()


    fig = plt.figure()
    for k, color in enumerate(colors):
        plt.loglog(Ntrains[:], (results[:,4+k]), color=color, label=outputs[4+k])
    plt.title(title + ' on the test set')
    plt.ylabel(title)
    plt.xlabel('Number of training points')
    plt.legend()
    plt.grid(True, which='both')
    plt.draw()
    if savefig:
        plt.show(block=False)
        fig.savefig('./Therm_' + title + ".png", bbox_inches="tight")
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

# plot_scalability(Ntrain_list, 1 - 0.01 * accuracies, 'Classification_error', savefig=False)
plot_scalability(Ntrain_list, MAEs, 'MAE', savefig=False)
plot_scalability(Ntrain_list, 0.01 * MREs, 'MRE', savefig=False)
plot_scalability(Ntrain_list, 0.01 * MEDREs, 'MedRE', savefig=False)

# print(accuracies)
