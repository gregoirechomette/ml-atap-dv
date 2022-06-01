#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# System integration
import os
import sys
import abc

from pandas.io.parsers import read_csv

# Standard python libraries
import h5py
import subprocess
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


''' =========== Update of input with consideration of bounds =========== '''
def update_inputs(x_input, x_scalings, grads, learningrate):

    # Order is: D, rho, S, alpha, V, angle, azimuth, eta, sigma
    lower_bounds = [0, 8e2, 1e5, 0.1, 5e3, 0, 0, 3e-4, 3.5e-10]
    upper_bounds = [2e3, 1e4, 1e7, 0.3, 5e4, 90, 360, 3e-2, 7e-8]

    # Convert input to numpy format
    x_input = x_input.numpy()
    gradients = grads.numpy()

    # After gradient descent
    x_modified = np.multiply(x_input - learningrate * gradients, x_scalings[1:2,:]) + x_scalings[0:1,:]

    # Loop over all gradients and check that solutions are in acceptable range
    for i in range(x_input.shape[1]):
        if((lower_bounds[i] < x_modified[0,i]) and (x_modified[0,i] < upper_bounds[i])):
            x_input [0,i] -= learningrate * gradients[0,i]
    
    x_input = tf.Variable(x_input)

    return x_input


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
        x_input = update_inputs(x_input, x_scalings, grads, learningrate)

        # Evaluate the result at this epoch
        y_out = model.predict(x_input.numpy())[0,0] * y_scalings[1,0] + y_scalings[0,0]

        if show_conv:
            print("Iteration # ", i, ", the output is: ", y_out)

        # Break if less than 3% relative error
        if (np.absolute(y_out - target_value) < 0.03 * target_value):
            if show_conv:
                print("Stopped after ", i ," iterations")
                
            break

        if i == epochs - 1:
            y_out = 0
            if show_conv:
                print("Did not converge")

    return np.multiply(x_input, x_scalings[1:2,:]) + x_scalings[0:1,:], y_out


''' =========== Implementation of the inverse design problem multiple times =========== '''

def multiple_inverse_problems(N_inverse, model_folder, target_value):

    for n in range(N_inverse):

        # Call the inverse function
        x,y = inverse_problem(
            model_folder, 
            inputs_sampling(), 
            target_value, 
            epochs=500, 
            learningrate=0.5, 
            regularizer=0, 
            show_conv=False)

        # Print results
        print("Iteration # ", n, " for inverse problem, output = ", y)

        # Concatenate
        if n == 0:
            input_inverse = np.reshape(x,(1,9))
            output_inverse = np.reshape(y,(1,1))
        else:
            input_inverse = np.concatenate((input_inverse, np.reshape(x,(1,9))), axis=0)
            output_inverse = np.concatenate((output_inverse, np.reshape(y,(1,1))), axis=0)

    res_df = pd.DataFrame(input_inverse, columns=inputs)
    res_df = res_df.assign(Output=pd.Series(output_inverse.flatten()).values)

    columns_names = ['Diameter', 'Density', 'Strength', 'Alpha', 'Velocity', 
          'Angle', 'Azimuth', 'LumEff', 'Ablation', outputs[0]]

    res_df.columns = (columns_names)

    return res_df


''' =========== Validate solution w/ PAIR =========== '''

def validate_solution(N_inverse, result_np, pair_folder):
    
    # Create input file
    file_name= 'scenarios.in.txt'
    os.system('rm ' + file_name)
    f = open(file_name, 'a+')
    for scenario in range(N_inverse):
        f.write('NAS' + '\t' + '1' + '\t' + str(result_np[scenario, 0]) 
                + '\t' + '0' + '\t' + '0' + '\t' + 'S' + '\t' 
                + str(result_np[scenario, 1]) + '\t' + str(result_np[scenario, 2]) + '\t' 
                + str(result_np[scenario, 3]) + '\t' + str(result_np[scenario, 4]) + '\t' 
                + str(result_np[scenario, 5]) + '\t' + str(result_np[scenario, 6]) + '\t' 
                + '37.421' + '\t' + '-122.065' + '\t' 
                + str(result_np[scenario, 7]) + '\t' + str(result_np[scenario, 8]) + '\t' 
                + '0' + '\n')
    f.close()

    # Run PAIR
    os.system('cp ' + file_name + ' ' + pair_folder)
    os.chdir(pair_folder)
    os.system('rm output-pair.csv')
    with open('output-pair.csv', 'a+') as output_f:
        p = subprocess.Popen('./risk',
        stdout = output_f,
        stderr = output_f)
    p.wait()
    
    return pd.read_csv('output-pair.csv', usecols=[' ThermRad2 ']).to_numpy()



''' ================ Run the inverse problem model ================ '''
# Target value
target_value = 1500
# Number of inverse problems to solve
N_inverse = 3

# Model and PAIR folder paths
model_folder = './results/Ntrain_2e+03/ThermRad2/'
pair_folder = '/Users/gchomett/Documents/Professional/NASA/PAIR-dv/PAIR-r68-laptop/PAIR-CodeFiles/'

# # Find solutions of the inverse problem
# result_df = multiple_inverse_problems(N_inverse, model_folder, target_value)
# result_np = result_df.to_numpy()

# print(result_np)

# Test purpose
test_np = np.array([[40.31, 3264.95, 386300, 0.1555, 15908, 49.93, 105.6, 0.02291, 1.79e-8, 1500],
                    [161.02, 2221.46, 5357000, 0.1284, 15456, 53.52, 80.84, 0.002974, 1.84e-8, 1500],
                    [156.96, 2468.12, 138800, 0.2932, 15637, 38.75, 121.39, 0.008511, 1.32e-8, 1500]])

# # Verify the solutions w/ PAIR
# val = validate_solution(N_inverse, result_np, pair_folder)
val = validate_solution(N_inverse, test_np, pair_folder)
print(val)