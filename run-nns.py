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

# ML libraries
import tensorflow as tf
from tensorflow.keras import Model, Input, regularizers, layers, models
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import preprocessing

from dataset import Dataset
from nn_modules import FCNN_classification, FCNN_with_variance

from utils import save_classification_results, save_regression_results, save_all_results
from utils import save_NN_info, remove_zero_class
from utils import plot_abs_error, plot_rel_error, plot_rel_distribution
from utils import plot_sorted_predictions, plot_regression_results


"""  ================ Function used for the full classification task ================ """

def classifier(dataset, outputs, NNtype, show_conv=False, show_metrics=False, savemodel=False):
    
    # Instantiate the neural network
    NN_classifier = FCNN_classification(learningrate, regularizer, batchsize, epochs, 
                                            patience, verbosity, outputfolder)

    # Create the neural network
    model = NN_classifier.make_nn(regularizer, learningrate, dataset.x.shape[1], NNtype=NNtype)

    # Train the neural network
    [y_predict_train, y_predict_test] = NN_classifier.train_nn(
        model, dataset.x_train, dataset.y_class_train, dataset.x_test, dataset.y_class_test, show_conv=show_conv)

    # Compute metrics
    acc, fp, fn = NN_classifier.compute_metrics(y_predict_test, dataset.y_class_test, show_metrics=show_metrics)

    if savemodel:
        if os.path.exists('./' + outputfolder) == False:
            os.mkdir('./' + outputfolder)
        else:
            if os.path.exists('./' + outputfolder + '/' + outputs[0]) == False:
                os.mkdir('./' + outputfolder + '/' + outputs[0])

        # Save the model architecture, the weights, and the results
        save_NN_info(dataset, NN_classifier, model, outputfolder, 'Classifier_summary.txt')
        model.save('./' + outputfolder + '/' + outputs[0] + '/Classification_model')
        save_classification_results(acc, fp, fn, outputfolder + '/' + outputs[0], 'Classification_results.txt')

    return acc, fp, fn, y_predict_train, y_predict_test



"""  ================ Function used for the full regression task ================ """

def regressor(dataset, outputs, NNtype, x_train, y_train, x_test, y_test, 
                show_conv=False, plot_predictions=True, show_error_bars=False, savefig=False, savemodel=False):
    
    # Instantiate the neural network
    NN_regressor = FCNN_with_variance(learningrate, regularizer, batchsize, epochs, 
                                        patience, verbosity, outputfolder)
    
    # Create the neural network
    model = NN_regressor.make_nn(regularizer, learningrate, dataset.x.shape[1], NNtype=NNtype)

    # Train the neural network
    [y_predict_train, y_predict_test] = \
        NN_regressor.train_nn(model, x_train, y_train, x_test, y_test, show_conv=show_conv)

    # Compute metrics
    mean_se, mean_ae, mean_re, med_re = NN_regressor.compute_metrics(
            y_predict_test, y_test, dataset.scaler_y, show_metrics=False)

    if savemodel or savefig:
        if os.path.exists('./' + outputfolder) == False:
            os.mkdir('./' + outputfolder)
        else:
            if os.path.exists('./' + outputfolder + '/' + outputs[0]) == False:
                os.mkdir('./' + outputfolder + '/' + outputs[0])

    # Save the plots
    if plot_predictions:
        plot_regression_results(
        dataset, y_train, y_predict_train, y_test, y_predict_test, 
        outputs, outputfolder+ '/' + outputs[0], show_error_bars, savefig)

    # Save the model architecture, the weights, and the results
    if savemodel:
        save_NN_info(dataset, NN_regressor, model, outputfolder, 'Regressor_summary.txt')
        model.save('./' + outputfolder + '/' + outputs[0] + '/Regression_model')
        save_regression_results(mean_se, mean_ae, mean_re, med_re, 
                                outputfolder + '/' + outputs[0], 'Regression_results.txt')

    return mean_se, mean_ae, mean_re, med_re



""" ================ Definition of the hyperparameters ================ """
# Choices concerning the data
inputs = ['Diameter', 'Density', 'Strength', 'Velocity', 
          'Angle', 'Azimuth', 'Alpha', 'LumEff', 'Ablation']
          
outputs = ['ThermRad2']
Ntrain = 5000     

# Design of the network and the hyper-parameters
learningrate = 0.001
regularizer = 0.0
batchsize = 256
epochs = 50
patience = 5
verbosity = 1
NNtype = 'shallow'

# Inputs/outputs
inputfile = 'data-1e4.csv'
outputfolder = 'results/Ntrain_' + "{:.0e}".format(Ntrain)



''' ================ Run the model once to classify and predict ================ '''

# Obtain the data
dataset = Dataset(inputs, outputs, Ntrain, inputfile)
dataset.prepare_data()

# Call the classifier
accuracy, fpos, fneg, y_classified_train, y_classified_test = classifier(
    dataset, outputs, NNtype, show_conv=False, show_metrics=True, savemodel=False)

# Update the dataset
remove_zero_radius = True
if remove_zero_radius:
    x_train, y_train, x_test, y_test = remove_zero_class(
        dataset, inputs, y_classified_train, y_classified_test)
else:
    x_train, y_train, x_test, y_test = dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test

# Call the regressor
mean_se, mean_ae, mean_re, med_re = regressor(
    dataset, outputs, NNtype, x_train, y_train, x_test, y_test, 
    show_conv=False, plot_predictions=True, show_error_bars=True, savefig=False, savemodel=False)




''' =========== Make a synthesis of the performance of the models =========== '''

output_list = [['BlastRad1'], ['BlastRad2'], ['BlastRad4'], ['BlastRad10'],
               ['ThermRad2'], ['ThermRad3'], ['ThermRad4'], ['ThermRad6']]
results = np.zeros((len(output_list), 7 ))

for n in range(len(output_list)):
      
    # Obtain the data
    dataset = Dataset(inputs, output_list[n], Ntrain, inputfile)
    dataset.prepare_data()

    # Call the classifier
    results[n,0], results[n,1], results[n,2], y_classified_train, y_classified_test = classifier(
        dataset, output_list[n], NNtype, show_conv=False, show_metrics=True, savemodel=True)

    # Update the dataset
    remove_zero_radius = True
    if remove_zero_radius:
        x_train, y_train, x_test, y_test = remove_zero_class(dataset, inputs, y_classified_train, y_classified_test)
    else:
        x_train, y_train, x_test, y_test = dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test

    # Call the regressor
    results[n,3], results[n,4], results[n,5], results[n,6] = regressor(
        dataset, output_list[n], NNtype, x_train, y_train, x_test, y_test, 
        show_conv=False, plot_predictions=True, show_error_bars=True, savefig=True, savemodel=True)
    
save_all_results(output_list, results, outputfolder, 'Summary_of_results.csv')