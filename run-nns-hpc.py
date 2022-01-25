#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# System integration
from distutils.command.config import config
import os
import sys
import abc

# Standard python libraries
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ML libraries
import tensorflow as tf
import horovod.tensorflow.keras as hvd
# import tensorflow.keras.backend as K
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras import Model, Input, regularizers, layers, models
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import preprocessing

from dataset import Dataset
from nn_modules import FCNN, FCNN_classification, FCNN_with_variance

from utils import save_classification_results, save_regression_results, save_all_results
from utils import save_NN_info, remove_zero_class
from utils import plot_abs_error, plot_rel_error, plot_rel_distribution
from utils import plot_sorted_predictions, plot_regression_results, plot_pred_and_re, plot_pred_and_re_populations, plot_classification, plot_probability_threshold


""" 
C'est la dedans que ca se passe:
https://github.com/shashankprasanna/distributed-tensorflow-horovod-sagemaker/tree/master/code
"""


"""  ================ Function used for the full regression task without variance ================ """

def regressor(dataset, outputs, NNtype, x_train, y_train, x_test, y_test, hvd,
                show_conv=False, plot_predictions=True, show_error_bars=False, savefig=False, savemodel=False):
    
    # Instantiate the neural network
    NN_regressor = FCNN(learningrate, regularizer, batchsize, epochs, 
                                        patience, verbosity, outputfolder)
    
    # Create the neural network
    model = NN_regressor.make_nn_for_hpc(regularizer, learningrate, dataset.x.shape[1], hvd)

    # Train the neural network
    [y_predict_train, y_predict_test] = \
        NN_regressor.train_nn_for_hpc(model, x_train, y_train, x_test, y_test, hvd, show_conv=show_conv)

    # Compute metrics
    mean_se, mean_ae, mean_re, med_re = NN_regressor.compute_metrics(
            y_predict_test, y_test, dataset.scaler_y, show_metrics=True)

    if savemodel or savefig:
        if os.path.exists('./' + outputfolder) == False:
            os.mkdir('./' + outputfolder)
        else:
            if os.path.exists('./' + outputfolder + '/' + outputs[0]) == False:
                os.mkdir('./' + outputfolder + '/' + outputs[0])

    # Save the plots
    if plot_predictions:
        plot_pred_and_re_populations(y_test, y_predict_test, dataset.scaler_y,
        outputs[0], outputfolder + '/' + outputs[0], savefig)

    # Save the model architecture, the weights, and the results
    if savemodel:
        save_NN_info(dataset, NN_regressor, model, outputfolder, 'Regressor_summary.txt')
        model.save('./' + outputfolder + '/' + outputs[0] + '/Regression_model')
        save_regression_results(mean_se, mean_ae, mean_re, med_re, 
                                outputfolder + '/' + outputs[0], 'Regression_results.txt')
        dataset.save_rescaling_params(outputfolder + '/' + outputs[0], 'Scaling_parameters.csv')

    return mean_se, mean_ae, mean_re, med_re



""" ================ Definition of the hyperparameters ================ """
# Choices concerning the data
inputs = ['latitude', 'longitude']
outputs = ['population']
Ntrain = 1500000

# Design of the network and the hyper-parameters
learningrate = 0.001
regularizer = 0.0
batchsize = 256
epochs = 150
patience = 5
verbosity = 1
NNtype = 'shallow'

# Inputs/outputs
inputfile = './data-population/cpp-scripts/population-count-coarse.csv'
outputfolder = 'results-population/Ntrain_' + "{:.0e}".format(Ntrain)



''' ================ Run the model once to classify and predict ================ '''

# Initialize horovod and get the size of the cluster
hvd.init()
size = hvd.size()

# Pin the GPU to local process (one GPU per process)
conf = tf.compat.v1.ConfigProto()
conf.gpu_options.allow_growth = True
conf.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.compat.v1.Session(config=conf))



# Obtain the data
dataset = Dataset(inputs, outputs, Ntrain, inputfile)
dataset.prepare_data()

# Call the regressor
mean_se, mean_ae, mean_re, med_re = regressor(
    dataset, outputs, NNtype, dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test, hvd,
    show_conv=False, plot_predictions=True, show_error_bars=True, savefig=False, savemodel=False)
