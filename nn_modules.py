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
from datetime import datetime

# ML libraries
import tensorflow as tf
from tensorflow.keras import Model, Input, regularizers, layers, models
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint


'''====================== Parent class, for all types of neural networks ======================'''
class GeneralNN:

    def __init__(self, learningrate, regularizer, batchsize, 
                    epochs, patience, verbosity, outputfolder):
      
        self.lr = learningrate
        self.reg = regularizer
        self.batchsize = batchsize
        self.epochs = epochs
        self.patience = patience
        self.verbosity = verbosity
        self.outputfolder = outputfolder


'''====================== Class related to simple FCNN ======================'''
class FCNN(GeneralNN):

    def __init__(self, learningrate, regularizer, batchsize, 
                    epochs, patience, verbosity, outputfolder):
                 
        GeneralNN.__init__(self, learningrate, regularizer, batchsize, 
                            epochs, patience, verbosity, outputfolder)


    ''' 
        Creates the NN, given the design, activation function, regularization and learning rate.
    '''
    def make_nn(self, lam, lr, input_size, NNtype='shallow'):

        # Design the architecture
        inputs = tf.keras.Input((input_size,), name="input1")
        predictions = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense1")(inputs)
        predictions = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense2")(predictions)
        predictions = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense3")(predictions)

        if NNtype == 'shallow':
            predictions = layers.Dense(1, activation='linear', name="dense4")(predictions)
        elif NNtype == 'deep':
            predictions = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense31")(predictions)
            predictions = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense32")(predictions)
            predictions = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense33")(predictions)
            predictions = layers.Dense(1, activation='linear', name="dense4")(predictions)

        # Instantiate the model with labels given as input
        model = tf.keras.Model(inputs=inputs, outputs=predictions)

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=['mae'])

        return model
    
    def make_nn_for_hpc(self, lam, lr, input_size, hvd):
    
        # Design the architecture
        inputs = tf.keras.Input((input_size,), name="input1")
        predictions = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense1")(inputs)
        predictions = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense2")(predictions)
        predictions = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense3")(predictions)
        predictions = layers.Dense(1, activation='linear', name="dense4")(predictions)
        
        # Get the number of workers
        size = hvd.size()
        
        # Instantiate the model with labels given as input
        model = tf.keras.Model(inputs=inputs, outputs=predictions)
        
        # Instantiate optimizer
        opt = tf.keras.optimizers.Adam(learning_rate=lr * size)
        
        # Wrap Keras optimizer into Horovod to make it a distributed optimizer
        opt = hvd.DistributedOptimizer(opt)

        # Compile the model
        model.compile(optimizer=opt,
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=['mae'])

        return model

    def make_nn_for_inverse(self, lam, lr, input_size, target, saved_model, NNtype='shallow'):

        # Design the architecture
        inputs = tf.keras.Input((input_size,), name="input1")
        predictions = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense1")(inputs)
        predictions = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense2")(predictions)
        predictions = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense3")(predictions)

        if NNtype == 'shallow':
            predictions = layers.Dense(1, activation='linear', name="dense4")(predictions)
        elif NNtype == 'deep':
            predictions = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense31")(predictions)
            predictions = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense32")(predictions)
            predictions = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense33")(predictions)
            predictions = layers.Dense(1, activation='linear', name="dense4")(predictions)

        # Calculate the difference between the predictions and target 
        difference = tf.math.squared_difference(predictions, target)

        # Instantiate the model and assign the weights
        new_model = tf.keras.Model(inputs=inputs, outputs=difference)
        new_model.set_weights(saved_model.get_weights()) 

        return new_model

    '''
        Method to train the NN with gradient descent with labeled data
    '''
    def train_nn(self, model, x_train, y_train, x_test, y_test, show_conv=False):

        # Call to the fit object to train the NN
        history = model.fit(x=x_train,
                            y=y_train,
                            batch_size = self.batchsize, 
                            epochs = self.epochs, 
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=self.patience)], 
                            verbose = self.verbosity, 
                            validation_data = (x_test, y_test))

        # print the history of the convergence
        if (show_conv): history_plot(history, self.outputfolder)

        # Returns y_predict_train, y_predict_test
        return [model.predict(x_train), model.predict(x_test)]

    '''
        Method to train the NN with gradient descent with labeled data on a distributed system
    '''
    def train_nn_for_hpc(self, model, x_train, y_train, x_test, y_test, hvd, show_conv=False):
        
        # Call to the fit object to train the NN
        history = model.fit(x = x_train,
                            y = y_train,
                            steps_per_epoch = (x_train.shape[0] // self.batchsize ) // hvd.size(),
                            epochs = self.epochs, 
                            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=self.patience)], 
                            verbose = self.verbosity if hvd.rank() == 0 else 0, 
                            validation_data = (x_test, y_test),
                            validation_steps = (x_test.shape[0] // self.batchsize ) // hvd.size())

        # print the history of the convergence
        if (show_conv): history_plot(history, self.outputfolder)

        # Returns y_predict_train, y_predict_test
        return [model.predict(x_train), model.predict(x_test)]


    def compute_metrics(self, y_predict_test, y_test, y_scaler, show_metrics=False):

        y_predict_test_rescaled = (y_scaler.scale_ * np.reshape(np.array(y_predict_test[:,0]), y_test.shape)[:,0]) + y_scaler.mean_
        y_true_rescaled = (y_scaler.scale_ * np.array(y_test))[:,0] + y_scaler.mean_
        rel_error = np.divide(np.absolute(y_true_rescaled - y_predict_test_rescaled), 
                          np.array(y_true_rescaled), 
                          out= 1e-4 + np.zeros_like(np.absolute(y_true_rescaled - y_predict_test_rescaled)), 
                          where=np.array(y_true_rescaled)!=0)
    
        # Order the data
        pred_dict = {'indices': np.arange(len(y_true_rescaled)),
                    'rel_error': rel_error,
                    'label': y_true_rescaled}

        pred_dict_df = pd.DataFrame(data=pred_dict)
        pred_dict_filtered_df = pred_dict_df[pred_dict_df['label']>1e-6]

        # Compute the metrics
        mean_se = np.mean(np.square(y_true_rescaled - y_predict_test_rescaled))
        mean_ae = np.mean(np.absolute(y_true_rescaled - y_predict_test_rescaled))
        mean_re = pred_dict_filtered_df['rel_error'].mean()
        med_re = pred_dict_filtered_df['rel_error'].median()
            
        # Print metrics
        if show_metrics:
            print("Mean squared error = ", round(mean_se))
            print("Mean absolute error = ", round(mean_ae))
            print("Mean relative error = ", round(mean_re * 100, 1),'%')
            print("Median relative error = ", round(med_re * 100, 1), '%')

        return round(mean_se), round(mean_ae), round(mean_re * 100, 1), round(med_re * 100, 1)


'''====================== Class related to FCNN with prediction of variance ======================'''
class FCNN_with_variance(GeneralNN):

    def __init__(self, learningrate, regularizer, batchsize, 
                    epochs, patience, verbosity, outputfolder):
                 
        GeneralNN.__init__(self, learningrate, regularizer, batchsize, 
                            epochs, patience, verbosity, outputfolder)

    ''' 
        Creates the NN, given the design, activation function, regularization and learning rate.
    '''
    def make_nn(self, lam, lr, input_size, NNtype='shallow'):

        # Design the architecture
        inputs = tf.keras.Input((input_size,), name="input1")
        predictions = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense1")(inputs)
        predictions = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense2")(predictions)
        predictions = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense3")(predictions)

        if NNtype == 'shallow':
            predictions = layers.Dense(1, activation='linear', name="dense4")(predictions)
        elif NNtype == 'deep':
            predictions = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense31")(predictions)
            predictions = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense32")(predictions)
            predictions = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense33")(predictions)
            predictions = layers.Dense(1, activation='linear', name="dense4")(predictions)

        variances = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense5")(inputs)
        variances = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense6")(variances)
        variances = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense7")(variances)
        variances = layers.Dense(1, activation='softplus', name="dense8")(variances)

        # Labels for the loss function
        labels_predictions = tf.keras.Input((1,), name="input2")
        labels_variances = tf.keras.Input((1,), name="input3")

        # Instantiate the model with labels given as input
        model = tf.keras.Model(inputs=[inputs, labels_predictions, labels_variances], outputs=[predictions, variances])

        # Define a loss of interest and add it
        loss = tf.math.reduce_mean(tf.math.divide(tf.math.squared_difference(predictions, labels_predictions), variances) + tf.math.log(variances), axis=None, keepdims=False, name=None)
        model.add_loss(loss)

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        # print(model.summary())

        return model

    '''
        Method to train the NN with gradient descent with labeled data
    '''
    def train_nn(self, model, x_train, y_train, x_test, y_test, show_conv=False):

        # Call to the fit object to train the NN
        history = model.fit(x=[x_train, y_train, 0.5 * y_train],
                            y=[y_train, y_train],
                            batch_size = self.batchsize, 
                            epochs = self.epochs, 
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=self.patience)], 
                            verbose = self.verbosity, 
                            validation_data = ([x_test, y_test, 0.5 * y_test], [y_test, y_test]))

        # print the history of the convergence
        if (show_conv): history_plot(history, self.outputfolder)

        # Returns y_predict_train, y_predict_test
        return [model.predict([x_train, y_train, 0.5 * y_train]), model.predict([x_test, y_test, 0.5 * y_test])]


    def compute_metrics(self, y_predict_test, y_test, y_scaler, show_metrics=False):

        y_predict_test_rescaled = (y_scaler.scale_ * np.reshape(np.array(y_predict_test[0]), y_test.shape)[:,0]) + y_scaler.mean_
        y_true_rescaled = (y_scaler.scale_ * np.array(y_test))[:,0] + y_scaler.mean_
        rel_error = np.divide(np.absolute(y_true_rescaled - y_predict_test_rescaled), 
                          np.array(y_true_rescaled), 
                          out= 1e-4 + np.zeros_like(np.absolute(y_true_rescaled - y_predict_test_rescaled)), 
                          where=np.array(y_true_rescaled)!=0)
    
        # Order the data
        pred_dict = {'indices': np.arange(len(y_true_rescaled)),
                    'rel_error': rel_error,
                    'label': y_true_rescaled}

        pred_dict_df = pd.DataFrame(data=pred_dict)
        pred_dict_filtered_df = pred_dict_df[pred_dict_df['label']>1e-6]

        # Compute the metrics
        mean_se = np.mean(np.square(y_true_rescaled - y_predict_test_rescaled))
        mean_ae = np.mean(np.absolute(y_true_rescaled - y_predict_test_rescaled))
        mean_re = pred_dict_filtered_df['rel_error'].mean()
        med_re = pred_dict_filtered_df['rel_error'].median()
            
        # Print metrics
        if show_metrics:
            print("Mean squared error = ", round(mean_se))
            print("Mean absolute error = ", round(mean_ae))
            print("Mean relative error = ", round(mean_re * 100, 1),'%')
            print("Median relative error = ", round(med_re * 100, 1),'%')

        return round(mean_se), round(mean_ae), round(mean_re * 100, 1), round(med_re * 100, 1)

        

'''====================== Class related to FCNN with classification ======================'''
class FCNN_classification(GeneralNN):

    def __init__(self, learningrate, regularizer, batchsize, 
                    epochs, patience, verbosity, outputfolder):
                 
        GeneralNN.__init__(self, learningrate, regularizer, batchsize, 
                            epochs, patience, verbosity, outputfolder)


    ''' 
        Creates the NN, given the design, activation function, regularization and learning rate.
    '''
    def make_nn(self, lam, lr, input_size, NNtype='shallow'):

        # Design the architecture
        inputs = tf.keras.Input((input_size,), name="input1")
        predictions = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense1")(inputs)
        predictions = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense2")(predictions)
        predictions = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense3")(predictions)

        if NNtype == 'shallow':
            predictions = layers.Dense(1, activation='sigmoid', name="dense4")(predictions)
        elif NNtype == 'deep':
            predictions = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense31")(predictions)
            predictions = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense32")(predictions)
            predictions = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(lam), name="dense33")(predictions)
            predictions = layers.Dense(1, activation='sigmoid', name="dense4")(predictions)

        # Instantiate the model with labels given as input
        model = tf.keras.Model(inputs=inputs, outputs=predictions)

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

        return model

    '''
        Method to train the NN with gradient descent with labeled data
    '''
    def train_nn(self, model, x_train, y_train, x_test, y_test, show_conv=False):

        # Call to the fit object to train the NN
        history = model.fit(x=x_train,
                            y=y_train,
                            batch_size = self.batchsize, 
                            epochs = self.epochs, 
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=self.patience)], 
                            verbose = self.verbosity, 
                            validation_data = (x_test, y_test))

        # print the history of the convergence
        if (show_conv): history_plot(history, self.outputfolder)

        # Returns y_predict_train, y_predict_test
        return [model.predict(x_train), model.predict(x_test)]


    def compute_metrics(self, y_predict_test, y_test, show_metrics=False):

        # Introduce a new variable to keep the probabilities
        y_predict_test_classified = np.array(y_predict_test)

        # Transform probabilities into predictions
        y_predict_test_classified[y_predict_test_classified < 0.5] = 0
        y_predict_test_classified[y_predict_test_classified >= 0.5] = 1

        # Compute the metrics
        accuracy = (np.count_nonzero(y_predict_test_classified == y_test))/y_predict_test_classified.shape[0]
        false_pos_rate = np.count_nonzero(
            np.logical_and(y_predict_test_classified != y_test, y_test == np.zeros(y_test.shape)))/y_predict_test_classified.shape[0]
        false_neg_rate = np.count_nonzero(
            np.logical_and(y_predict_test_classified != y_test, y_test == np.ones(y_test.shape)))/y_predict_test_classified.shape[0]
        
        # Print metrics
        if show_metrics:
            print("Accuracy = " + str(round(accuracy * 100, 1)) + '%')
            print("False positive rate = " + str(round(false_pos_rate * 100, 1)) + '%')
            print("False negative rate = " + str(round(false_neg_rate * 100, 1)) + '%')

        return round(accuracy * 100, 1), round(false_pos_rate * 100, 1), round(false_neg_rate * 100, 1)