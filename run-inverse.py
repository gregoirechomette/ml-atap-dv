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
from scipy.stats import loguniform, cosine

from dataset import Dataset
from nn_modules import FCNN, FCNN_classification, FCNN_with_variance

from utils import save_classification_results, save_regression_results, save_all_results
from utils import save_NN_info, remove_zero_class
from utils import plot_abs_error, plot_rel_error, plot_rel_distribution
from utils import plot_sorted_predictions, plot_regression_results
from utils import plot_inputs_distributions



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


''' =========== Sampling input coefficients from distributions of Mathias et al. (2017) =========== '''
def inputs_sampling():

    # Diameter (double check the lambda coefficients!)
    diameter = 1.326e6 *  np.power(10,-0.2 * np.random.uniform(20,30)) / np.sqrt(1e-8 + np.absolute(
        0.44 * np.random.normal(loc=0.034, scale=0.014) +
        1.21 * np.random.normal(loc=0.151, scale=0.122)
        ))

    # Density
    density = 1000 * (1 - np.random.normal(loc=0.34, scale=0.18)) * (
        0.349 * np.random.normal(loc=3.38, scale=0.19) +
        0.389 * np.random.normal(loc=3.30, scale=0.12) +
        0.093 * np.random.normal(loc=3.19, scale=0.14) +
        0.043 * np.random.normal(loc=2.27, scale=0.13) +
        0.025 * np.random.normal(loc=6.75, scale=1.84) +
        0.024 * np.random.normal(loc=7.15, scale=0.57) +
        0.034 * np.random.normal(loc=2.84, scale=0.13) +
        0.011 * np.random.normal(loc=3.12, scale=0.19) +
        0.034 * np.random.normal(loc=2.86, scale=0.11)
    )
    # Strength
    strength = loguniform.rvs(1e5, 1e7)
    # Alpha
    alpha = np.random.uniform(0.1,0.3)
    # Velocity
    velocity = 5e3 * np.random.gamma(shape=9.0, scale=0.5) # Approximation, not correct
    # Angle 
    angle = 45 + cosine.rvs(scale=45/np.pi)
    # Azimuth
    azimuth = np.random.uniform(0,360)
    # LumEff
    lumeff = loguniform.rvs(3e-4, 3e-2)
    # Ablation coefficient
    ablation = loguniform.rvs(3.5e-10, 7e-8)

    return [diameter, density, strength, alpha, velocity, angle, azimuth, lumeff, ablation]


''' =========== Sampling of multiple input coefficients and ploting of distributions ==========='''
def sample_and_plot_distributions(N_samples):

    # Initiate the table with first sampling
    input_samples = np.reshape(inputs_sampling(), (1,9))
    for i in range(N_samples):
        input_samples = np.concatenate((input_samples, np.reshape(inputs_sampling(), (1,9))), axis=0)

    plot_inputs_distributions(input_samples)

    return


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



''' ================ Run the model once to classify and predict ================ '''

model_folder = './results/Ntrain_2e+03/ThermRad2/'
inputs_values = [100, 2000, 1e6, 0.2, 1e4, 45, 0.0, 3e-3, 1e-8]
target_value = 15000


x,y = inverse_problem(
    model_folder, inputs_values, target_value, epochs=500, learningrate=0.5, regularizer=0, show_conv=False)

print("Result x= ", x)
print("Result y= ", y)

sample_and_plot_distributions(10000)