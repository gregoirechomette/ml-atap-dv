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
from nn_modules import FCNN, FCNN_classification, FCNN_with_variance

from utils import save_classification_results, save_regression_results, save_all_results
from utils import save_NN_info, remove_zero_class
from utils import plot_abs_error, plot_rel_error, plot_rel_distribution
from utils import plot_sorted_predictions, plot_regression_results



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



''' =========== Implementation of the inverse design problem =========== '''

def inverse_problem(model_folder, inputs_values, target_value, epochs, learningrate, regularizer, show_conv):

    # Load the model
    model = models.load_model(model_folder + 'Regression_model')

    # Rescaling coefficients
    x_scalings = pd.read_csv(model_folder + 'Scaling_parameters.csv')[inputs].to_numpy()
    y_scalings = pd.read_csv(model_folder + 'Scaling_parameters.csv')[outputs].to_numpy()

    # Create the input, rescale it and convert it to a tensorflow format
    x_input = np.reshape(np.array(inputs_values),(1,9))
    x_input = np.divide(np.subtract(x_input, x_scalings[0:1,:]), x_scalings[1:2,:])
    x_input = tf.Variable(x_input, dtype='float32')

    # Create the output, rescale it and convert it to a tensorflow format
    y = np.reshape(np.array([target_value]),(1,1))
    y = np.divide(np.subtract(y, y_scalings[0:1,:]), y_scalings[1:2,:])
    y = tf.constant(y, shape=(1,), dtype='float32')

    # Instantiate the neural network
    NN_regressor = FCNN(learningrate, regularizer, batchsize, epochs, 
                                    patience, verbosity, outputfolder)
        
    # Create the neural network
    new_model = NN_regressor.make_nn_for_inverse(
        regularizer, learningrate, x_scalings.shape[1], y, model, NNtype=NNtype)

    # Optimization loop
    for i in range(epochs):
        with tf.GradientTape() as tape:
            preds = new_model(x_input)
        grads = tape.gradient(preds, x_input)

        # Update the input
        x_input = tf.Variable(tf.math.subtract(x_input, tf.math.scalar_mul(learningrate, grads)))

        # Evaluate the result at this epoch
        y_out = model.predict(x_input.numpy())[0,0] * y_scalings[1,0] + y_scalings[0,0]

        if show_conv:
            print("Iteration # ", i, ", the output is: ", y_out)

        # Break if less than 3% relative error
        if (np.absolute(y_out - target_value) < 0.05 * target_value):
            if show_conv:
                print("Stopped after ", i ," iterations")
                
            break

    return np.multiply(x_input, x_scalings[1:2,:]) + x_scalings[0:1,:], y_out


model_folder = './results/Ntrain_2e+03/ThermRad2/'
inputs_values = [100, 2000, 1e6, 0.2, 1e4, 45, 0.0, 3e-3, 1e-8]
target_value = 15000


x,y = inverse_problem(
    model_folder, inputs_values, target_value, epochs=500, learningrate=0.5, regularizer=0, show_conv=False)

print("Result x= ", x)
print("Result y= ", y)