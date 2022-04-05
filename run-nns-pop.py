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
import plotly.express as px
import matplotlib.pyplot as plt

# ML libraries
import tensorflow as tf
from tensorflow.keras import Model, Input, regularizers, layers, models
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import preprocessing

from dataset import Dataset
from nn_modules import FCNN, FCNN_classification, FCNN_with_variance

from utils import plot_pred_and_re_zoom, save_classification_results, save_regression_results, save_all_results
from utils import save_NN_info, remove_zero_class
from utils import plot_abs_error, plot_rel_error, plot_rel_distribution
from utils import plot_sorted_predictions, plot_regression_results, plot_pred_and_re, plot_classification, plot_probability_threshold, plot_pred_and_re_zoom


"""  ================ Function used for the full classification task ================ """

def classifier(dataset, outputs, NNtype, show_conv=False, show_metrics=False, plot_classif=False, savefig=False, savemodel=False):
    
    # Instantiate the neural network
    NN_classifier = FCNN_classification(learningrate, regularizer, batchsize, epochs, 
                                            patience, verbosity, outputfolder)

    # Create the neural network
    model = NN_classifier.make_nn(regularizer, learningrate, dataset.x.shape[1], NNtype=NNtype)

    # Train the neural network
    [y_predict_train, y_predict_test] = NN_classifier.train_nn(
        model, dataset.x_train, dataset.y_class_train, dataset.x_test, dataset.y_class_test, show_conv=show_conv)

    if savemodel or savefig:
        if os.path.exists('./' + outputfolder) == False:
            os.mkdir('./' + outputfolder)

    if plot_classif:
        
        # Retrieve the latitudes and longitudes
        latitudes = dataset.scaler_x.inverse_transform(dataset.x_train)[:,0]
        longitudes = dataset.scaler_x.inverse_transform(dataset.x_train)[:,1]

        my_array = np.array([latitudes, longitudes, dataset.y_class_train.flatten(),  y_predict_train.flatten()]).T
        df = pd.DataFrame(my_array, columns = ['latitudes', 'longitudes', 'population_true', 'Probability'])

        fig = px.density_mapbox(df, lat='latitudes', lon='longitudes', z='Probability', 
                                radius=10, center=dict(lat=37, lon=-119), zoom=2,
                                opacity=0.5, title= 'Probility of human life',mapbox_style="stamen-terrain")
        if savefig:   
            fig.write_html('./' + outputfolder + '/human-life-probabilities.html')
        fig.show()

    # Compute metrics
    acc, fp, fn = NN_classifier.compute_metrics(y_predict_test, dataset.y_class_test, show_metrics=show_metrics)

    if savemodel:
        # Save the model architecture, the weights, and the results
        save_NN_info(dataset, NN_classifier, model, outputfolder, 'Classifier_summary.txt')
        model.save('./' + outputfolder + '/Classification_model')
        save_classification_results(acc, fp, fn, outputfolder, 'Classification_results.txt')

    return acc, fp, fn, y_predict_train, y_predict_test



"""  ================ Function used for the full regression task without variance ================ """

def regressor_wo_variance(dataset, outputs, NNtype, x_train, y_train, x_test, y_test, 
                show_conv=False, plot_predictions=True, show_error_bars=False, savefig=False, savemodel=False):
    
    # Instantiate the neural network
    NN_regressor = FCNN(learningrate, regularizer, batchsize, epochs, 
                                        patience, verbosity, outputfolder)
    
    # Create the neural network
    model = NN_regressor.make_nn(regularizer, learningrate, dataset.x.shape[1], NNtype=NNtype)

    # Train the neural network
    [y_predict_train, y_predict_test] = \
        NN_regressor.train_nn(model, x_train, y_train, x_test, y_test, show_conv=show_conv)

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
        plot_pred_and_re_zoom(y_test, y_predict_test, dataset.scaler_y,
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

# World 
minLat = -85
maxLat = 85
minLon = -180
maxLon = 180

# # USA 
# minLat = 20
# maxLat = 50
# minLon = -130
# maxLon = -60

# Take 75% of the available data in that region
# Ntrain = int(0.75 * (maxLat - minLat) * (maxLon - minLon) / (0.042) / (0.042))
Ntrain = 1000000

# Design of the network and the hyper-parameters
learningrate = 0.001
regularizer = 0.0
batchsize = 256
epochs = 10
patience = 5
verbosity = 1
NNtype = 'shallow'

# Inputs/outputs
inputfile = './data-population/cpp-scripts/population-count-coarse.csv'
outputfolder = 'results/Ntrain_pop_world_' + "{:.0e}".format(Ntrain)


""" ================ Population classification model ================ """

# Obtain the data
dataset = Dataset(inputs, outputs, Ntrain, inputfile)
dataset.prepare_data_population(minLat, maxLat, minLon, maxLon)

# Call the classifier
accuracy, fpos, fneg, y_classified_train, y_classified_test = classifier(
    dataset, outputs, NNtype, show_conv=False, plot_classif=True, show_metrics=True, savefig=True, savemodel=True)