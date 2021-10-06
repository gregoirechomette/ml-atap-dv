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

def save_NN_info(dataset, module, model, outputfolder, outputfile):
    with open('./' + outputfolder + '/' + outputfile, 'w') as fh:
        fh.write(' \nNEURAL NETWORK DESIGN: \n \n')
        model.summary(print_fn=lambda x: fh.write(x + '/n'))

        fh.write('\nDATA USED: \n')
        fh.write('Number of training points:     ' + str(dataset.N_train) + '\n')
        fh.write('Number of test points:         ' + str(dataset.N_test) + '\n')
        fh.write('Number of val points:          ' + str(dataset.N_val) + '\n')

        fh.write('\nOTHER PARAMETERS: \n')
        fh.write('Learning rate:                 ' + str(module.lr) + '\n')
        fh.write('Regularizer:                   ' + str(module.reg) + '\n')
        fh.write('Batch size:                    ' + str(module.batchsize) + '\n')
        fh.write('Epochs:                        ' + str(module.epochs) + '\n')
        fh.write('Patience:                      ' + str(module.patience) + '\n')

    return

def save_classification_results(accuracy, false_pos_rate, false_neg_rate, outputfolder, outputfile):
    with open('./' + outputfolder + '/' + outputfile, 'w') as fh:

        fh.write('\nCLASSIFICATION RESULTS: \n \n')
        fh.write('Accuracy:                ' + str(accuracy) + '\n')
        fh.write('False positive rate:     ' + str(false_pos_rate) + '\n')
        fh.write('False negative rate:     ' + str(false_neg_rate) + '\n')

    return

def save_regression_results(mean_se, mean_ae, mean_re, med_re, outputfolder, outputfile):
    with open('./' + outputfolder + '/' + outputfile, 'w') as fh:

        fh.write('\nREGRESSION RESULTS: \n \n')
        fh.write('Mean squared error:      ' + str(mean_se) + '\n')
        fh.write('Mean absolute error:     ' + str(mean_ae) + '\n')
        fh.write('Mean relative error:     ' + str(mean_re) + '\n')
        fh.write('Median relative error:   ' + str(med_re) + '\n')
        
    return

def save_all_results(output_list, output_metrics, outputfolder, outputfile):
    # Flatten the list
    output_list_flat = []
    for i in range(len(output_list)):
        output_list_flat.append(output_list[i][0])

    df = pd.DataFrame({'Output': output_list_flat})
    df = df.assign(Accuracy=pd.Series(output_metrics[:,0].flatten()).values)
    df = df.assign(False_pos=pd.Series(output_metrics[:,1].flatten()).values)
    df = df.assign(False_neg=pd.Series(output_metrics[:,2].flatten()).values)
    df = df.assign(Mean_SE=pd.Series(output_metrics[:,3].flatten()).values)
    df = df.assign(Mean_AE=pd.Series(output_metrics[:,4].flatten()).values)
    df = df.assign(Mean_RE=pd.Series(output_metrics[:,5].flatten()).values)
    df = df.assign(Median_RE=pd.Series(output_metrics[:,6].flatten()).values)

    df.columns = (['Output', 'Accuracy (%)','False pos rate (%)','False neg rate(%)', 'Mean SE', 'Mean AE', 'Mean RE (%)', 'Median RE (%)'])
    df.to_csv('./' + outputfolder + '/' + outputfile)
    return

def remove_zero_class(module, inputs, y_classified_train, y_classified_test):
    
    # Order the data
    train_df = pd.DataFrame(module.x_train, columns=inputs)
    train_df = train_df.assign(Label=pd.Series(module.y_train.flatten()).values)
    train_df = train_df.assign(Category=pd.Series(module.y_class_train.flatten()).values)
    train_df = train_df.assign(Category_pred=pd.Series(y_classified_train.flatten()).values)

    test_df = pd.DataFrame(module.x_test, columns=inputs)
    test_df = test_df.assign(Label=pd.Series(module.y_test.flatten()).values)
    test_df = test_df.assign(Category=pd.Series(module.y_class_test.flatten()).values)
    test_df = test_df.assign(Category_pred=pd.Series(y_classified_test.flatten()).values)

    train_df_filtered = train_df[train_df['Category']>0.5]
    test_df_filtered = test_df[test_df['Category']>0.5]

    train_filtered = train_df_filtered.to_numpy()
    test_filtered = test_df_filtered.to_numpy()
    return train_filtered[:, :-3], train_filtered[:, -3:-2], test_filtered[:, :-3], test_filtered[:, -3:-2]


def history_plot(history, outputfolder):
    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Convergence')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Training set', 'Test set'], loc='upper left')
    plt.grid(True)
    plt.draw()
    plt.show()
    plt.close()


def plot_sorted_predictions(y_predict, y_true, y_scaler, color, dataset, output, outputfolder, show_error_bars=False, savefig=False):

    # Retrieve mu and variance
    y_predict_mu = np.reshape(np.array(y_predict[0]), y_true.shape)
    y_predict_var = np.reshape(np.array(y_predict[1]), y_true.shape)

    # Create upper and lower bounds
    y_lower = y_predict_mu - y_predict_var
    y_upper = y_predict_mu + y_predict_var

    pred_dict = {'indices': np.arange(len(y_true[:,0])),
                'mu': (y_scaler.scale_ * y_predict_mu[:,0]) + y_scaler.mean_,
                'label': (y_scaler.scale_ * y_true[:,0]) + y_scaler.mean_,
                'low': (y_scaler.scale_ * y_lower[:,0]) + y_scaler.mean_,
                'up': (y_scaler.scale_ * y_upper[:,0]) + y_scaler.mean_}
  
    pred_dict_df = pd.DataFrame(data=pred_dict)
    pred_dict_df_sorted = pred_dict_df.sort_values(by='label')

    if os.path.exists('./' + outputfolder) == False and savefig == True:
        os.mkdir('./' + outputfolder)

    fig = plt.figure()
    plt.plot(pred_dict_df_sorted['mu'].values, color + 'o', label='ML predictions', markersize=1)
    plt.plot(pred_dict_df_sorted['label'].values, 'ko', label='PAIR simulations', markersize=1)
    if show_error_bars:
        plt.fill_between(np.arange(len(y_predict_mu[:,0])), pred_dict_df_sorted['low'].values, pred_dict_df_sorted['up'].values, alpha=0.4, color=color, label='Error bars')
    plt.grid(True)
    plt.title('Predictions on the ' + dataset)
    plt.legend(loc='upper left')
    plt.xlabel('Configuration number, sorted by increasing output')
    plt.ylabel(output)
    plt.draw()
    if savefig:
        plt.show(block=False)
        fig.savefig('./' + outputfolder + "/" +"prediction_" + dataset + ".png", bbox_inches="tight")
    else: 
        plt.show()
    plt.close()

    
def plot_abs_error(y_predict, y_true, y_scaler, color, dataset, output, outputfolder, savefig=False):

    y_predict_rescaled = (y_scaler.scale_ * np.reshape(np.array(y_predict[0]), y_true.shape)[:,0]) + y_scaler.mean_
    y_true_rescaled = (y_scaler.scale_ * np.array(y_true))[:,0] + y_scaler.mean_
    abs_error = np.absolute(y_true_rescaled - y_predict_rescaled)

    # Order the data
    pred_dict = {'indices': np.arange(len(y_true_rescaled)),
                 'abs_error': abs_error,
                 'label': y_true_rescaled}

    pred_dict_df = pd.DataFrame(data=pred_dict)
    pred_dict_df_sorted = pred_dict_df.sort_values(by='label')

    fig = plt.figure()
    plt.plot(pred_dict_df_sorted['abs_error'].values, color + 'o', markersize=1)
    plt.grid(True)
    # plt.legend(loc='upper left')
    plt.xlabel('Configuration number, sorted by increasing output')
    plt.ylabel('Absolute error')
    plt.title('Absolute error on the ' + dataset + ' for ' + output)
    plt.draw()
    if savefig:
        plt.show(block=False)
        fig.savefig('./' + outputfolder + "/" +"abs_error_" + dataset + ".png", bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_rel_error(y_predict, y_true, y_scaler, color, dataset, output, outputfolder, savefig=False):

    y_predict_rescaled = (y_scaler.scale_ * np.reshape(np.array(y_predict[0]), y_true.shape)[:,0]) + y_scaler.mean_
    y_true_rescaled = (y_scaler.scale_ * np.array(y_true))[:,0] + y_scaler.mean_
    rel_error = np.divide(np.absolute(y_true_rescaled - y_predict_rescaled), 
                          np.array(y_true_rescaled), 
                          out= 1e-4 + np.zeros_like(np.absolute(y_true_rescaled - y_predict_rescaled)), 
                          where=np.array(y_true_rescaled)!=0)

    # Order the data
    pred_dict = {'indices': np.arange(len(y_true_rescaled)),
                 'rel_error': rel_error,
                 'label': y_true_rescaled}

    pred_dict_df = pd.DataFrame(data=pred_dict)
    pred_dict_df_sorted = pred_dict_df.sort_values(by='label')

    fig = plt.figure()
    plt.plot(100 * pred_dict_df_sorted['rel_error'].values, color + 'o', markersize=1)
    plt.grid(True)
    plt.yscale("log")
    # plt.legend(loc='upper left')
    plt.xlabel('Configuration number, sorted by increasing output')
    plt.ylabel('Relative error in %') 
    plt.title('Relative error on the ' + dataset + ' for ' + output)
    plt.draw()
    if savefig:
        plt.show(block=False)
        fig.savefig('./' + outputfolder + "/" +"rel_error_" + dataset + ".png", bbox_inches="tight")
    else:
        plt.show()
    plt.close()

def plot_rel_distribution(y_predict, y_true, y_scaler, color, dataset, output, outputfolder, savefig=False):

    y_predict_rescaled = (y_scaler.scale_ * np.reshape(np.array(y_predict[0]), y_true.shape)[:,0]) + y_scaler.mean_
    y_true_rescaled = (y_scaler.scale_ * np.array(y_true))[:,0] + y_scaler.mean_
    rel_error = np.divide(np.absolute(y_true_rescaled - y_predict_rescaled), 
                          np.array(y_true_rescaled), 
                          out= 1e-4 + np.zeros_like(np.absolute(y_true_rescaled - y_predict_rescaled)), 
                          where=np.array(y_true_rescaled)!=0)
    
    # Order the data
    pred_dict = {'indices': np.arange(len(y_true_rescaled)),
                 'rel_error': rel_error,
                 'label': y_true_rescaled}

    pred_dict_df = pd.DataFrame(data=pred_dict)
    pred_dict_filtered_df = pred_dict_df[pred_dict_df['label']>1e-6]


    fig = plt.figure()
    plt.hist(100 * pred_dict_filtered_df['rel_error'], bins=40, color='r',density=True, cumulative=False, range=(0,100))
    plt.title('Density distribution of the relative errors on the ' + dataset)
    plt.annotate('mean = ' + str(round(pred_dict_filtered_df['rel_error'].mean() * 100, 2)) + "%", xy=(0.7, 0.85), xycoords='axes fraction')
    plt.annotate('median = ' + str(round(pred_dict_filtered_df['rel_error'].median() * 100, 2)) + "%", xy=(0.69, 0.75), xycoords='axes fraction')
    plt.xlabel('Relative error in %')
    plt.ylabel('Approximate density function')
    if savefig:
        plt.show(block=False)
        fig.savefig('./' + outputfolder + "/" +"rel_error_distr_" + dataset + ".png", bbox_inches="tight")
    else:
        plt.show()
    plt.close()

def plot_regression_results(module, y_train, y_predict_train, y_test, y_predict_test, outputs, outputfolder, show_error_bars, savefig):
        
    plot_sorted_predictions(y_predict_train, y_train, module.scaler_y, color='b', dataset='training set', output=outputs[0], outputfolder=outputfolder, show_error_bars=show_error_bars, savefig=savefig)
    plot_sorted_predictions(y_predict_test, y_test, module.scaler_y, color='r', dataset='test set', output=outputs[0], outputfolder=outputfolder, show_error_bars=show_error_bars, savefig=savefig)
    
    # plot_abs_error(y_predict_train, y_train, module.scaler_y, color='b', dataset='training set', output=outputs[0], outputfolder=outputfolder, savefig=savefig)
    plot_abs_error(y_predict_test, y_test, module.scaler_y, color='r', dataset='test set', output=outputs[0], outputfolder=outputfolder, savefig=savefig)
    
    # plot_rel_error(y_predict_train, y_train, module.scaler_y, color='b', dataset='training set', output=outputs[0], outputfolder=outputfolder, savefig=savefig)
    plot_rel_error(y_predict_test, y_test, module.scaler_y, color='r', dataset='test set', output=outputs[0], outputfolder=outputfolder, savefig=savefig)
    
    plot_rel_distribution(y_predict_test, y_test, module.scaler_y, color='b', dataset='test set', output=outputs[0], outputfolder=outputfolder, savefig=savefig)

    return

    plot_scalability(Ntrain_list, classification_accuracies, 
                              regression_mean_abs_error, 
                              regression_mean_rel_error,
                              regression_median_rel_error,
                              savefig=False)

def plot_scalability(Ntrain_list, accuracies, mean_abs_error, mean_rel_error, median_rel_error, savefig=False):
    
    colors_linspace = np.linspace(0, 1, len(Noutput_list))
    colors = [plt.cm.seismic(x) for x in colors_linspace]

    fig = plt.figure()
    for k, color in enumerate(colors):
        plt.loglog(Ntrain_list[:], (1 - accuracies[:,k]), color=color, label=Noutput_list[k][0])
    plt.title('Classification error on the test set')
    plt.ylabel('Classification error')
    plt.xlabel('Number of training points')
    plt.legend()
    plt.grid(True, which='both')
    plt.draw()
    if savefig:
        plt.show(block=False)
        fig.savefig('./'+"scalability_accuracies" + ".png", bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    fig = plt.figure()
    for k, color in enumerate(colors):
        plt.loglog(Ntrain_list[:], mean_abs_error[:,k], color=color, label=Noutput_list[k][0])
    plt.title('Mean absolute error on the test set')
    plt.ylabel('Mean absolute error')
    plt.xlabel('Number of training points')
    plt.legend()
    plt.grid(True, which='both')
    plt.draw()
    if savefig:
        plt.show(block=False)
        fig.savefig('./'+"scalability_abs_err" + ".png", bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    fig = plt.figure()
    for k, color in enumerate(colors):
        plt.loglog(Ntrain_list[:], mean_rel_error[:,k], color=color, label=Noutput_list[k][0])
    plt.title('Mean relative error on the test set')
    plt.ylabel('Mean relative error')
    plt.xlabel('Number of training points')
    plt.legend()
    plt.grid(True, which='both')
    plt.draw()
    if savefig:
        plt.show(block=False)
        fig.savefig('./'+"scalability_mean_rel_err" + ".png", bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    fig = plt.figure()
    for k, color in enumerate(colors):
        plt.loglog(Ntrain_list[:], median_rel_error[:,k], color=color, label=Noutput_list[k][0])
    plt.title('Median relative error on the test set')
    plt.ylabel('Median relative error')
    plt.xlabel('Number of training points')
    plt.legend()
    plt.grid(True, which='both')
    plt.draw()
    if savefig:
        plt.show(block=False)
        fig.savefig('./'+"scalability_median_rel_err" + ".png", bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    return


def plot_inputs_distributions(input_samples):
    
    plt.subplot(3,3,1)
    plt.hist(input_samples[:,0], bins=40, density=True, range=(0,800))
    plt.ylabel('Norm. probability')
    plt.title('Diameter distribution', fontsize=10)

    plt.subplot(3,3,2)
    plt.hist(input_samples[:,1], bins=40, density=True)
    plt.ylabel('Norm. probability')
    plt.title('Density distribution')

    plt.subplot(3,3,3)
    plt.hist(np.log10(input_samples[:,2]), bins=40, density=True)
    plt.ylabel('Norm. probability')
    plt.title('Strength distribution of log(Strength)')

    plt.subplot(3,3,4)
    plt.hist(input_samples[:,3], bins=40, density=True)
    plt.ylabel('Norm. probability')
    plt.title('Alpha distribution')

    plt.subplot(3,3,5)
    plt.hist(input_samples[:,4], bins=40, density=True)
    plt.ylabel('Norm. probability')
    plt.title('Velocity distribution')

    plt.subplot(3,3,6)
    plt.hist(input_samples[:,5], bins=40, density=True)
    plt.ylabel('Norm. probability')
    plt.title('Angle distribution')

    plt.subplot(3,3,7)
    plt.hist(input_samples[:,6], bins=40, density=True)
    plt.ylabel('Norm. probability')
    plt.title('Azimuth distribution')

    plt.subplot(3,3,8)
    plt.hist(np.log10(input_samples[:,7]), bins=40, density=True)
    plt.ylabel('Norm. probability of log(LumEff)')
    plt.title('Luminous efficiency distribution')

    plt.subplot(3,3,9)
    plt.hist(np.log10(input_samples[:,8]), bins=40, density=True)
    plt.ylabel('Norm. probability of log(Ablation)')
    plt.title('Ablation distribution')
    plt.show()

    return
