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
from sklearn import preprocessing

class Dataset:

    def __init__(self, inputs, outputs, Ntrain, inputfile):

        self.inputs = inputs
        self.outputs = outputs
        self.N_train = Ntrain
        self.N_val = int(0.1 * Ntrain)
        self.N_test = int(0.1 * Ntrain)
        self.csv_name = inputfile

    def prepare_data(self):
        # Retrieve the data
        data_inputs = pd.read_csv('./data/'+ self.csv_name, usecols=self.inputs)
        data_outputs = pd.read_csv('./data/'+ self.csv_name, usecols=self.outputs)

        # Normalize input and output
        scaler_x = preprocessing.StandardScaler().fit(data_inputs.iloc[:, :].values)
        self.x = scaler_x.transform(data_inputs.iloc[:, :].values)
        scaler_y = preprocessing.StandardScaler().fit(data_outputs.iloc[:, :].values)
        self.y = scaler_y.transform(data_outputs.iloc[:, :].values)

        # Put output in different categories
        data_outputs[self.outputs[0]].mask(data_outputs[self.outputs[0]] > 1, 1, inplace=True)
        self.y_class = data_outputs.iloc[:, :].values

        # Verify that we have enough data in our database
        assert np.shape(self.x)[0] > self.N_test + self.N_val + self.N_train, "The data base contains insufficient data"

        # Split the inputs into test, validation, train, and cross validation sets
        self.x_test = self.x[0:self.N_test, :]
        self.x_val = self.x[self.N_test:self.N_test + self.N_val, :]
        self.x_train = self.x[self.N_test + self.N_val: self.N_test + self.N_val + self.N_train, :]
        
        # Split the outputs into test, validation, train, and cross validation sets
        self.y_test = self.y[0:self.N_test, :]
        self.y_val = self.y[self.N_test:self.N_test + self.N_val, :]
        self.y_train = self.y[self.N_test + self.N_val: self.N_test + self.N_val + self.N_train, :]

        # Split the  categorized into test, validation, train, and cross validation sets
        self.y_class_test = self.y_class[0:self.N_test, :]
        self.y_class_val = self.y_class[self.N_test:self.N_test + self.N_val, :]
        self.y_class_train = self.y_class[self.N_test + self.N_val: self.N_test + self.N_val + self.N_train, :]

        # remember scaler and return it
        self.scaler_y = scaler_y
        return