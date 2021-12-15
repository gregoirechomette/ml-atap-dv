#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# System integration
import os
import sys
import abc
import csv

# Standard python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Name of the file to read
file_name = './full-data.txt'

# Number of data points to extract
N = 1e4

# Columns of interest
INPUTS = ['Diameter', 'Density', 'Strength', 'Velocity', 
            'Angle', 'Azimuth', 'Alpha', 'LumEff', 'Ablation', 
            'HMag', 'Albedo', 'Altitude', 'GroundE', ' Latitude', 'Longitude']
OUTPUTS = ['BlastRad1', 'BlastRad2', 'BlastRad4', 'BlastRad10', 
            'ThermRad2', 'ThermRad3', 'ThermRad4', 'ThermRad6', 
            'BlastPop1', 'BlastPop2', 'BlastPop4', 'BlastPop10', 
            'ThermPop2', 'ThermPop3', 'ThermPop4', 'ThermPop6', 
            'TsuWet', 'TsuCas', 'GlobalCas']

# Read the CSV file and save it to a CSV file
data = pd.read_csv(file_name, sep=' , ', engine ='python', 
                    usecols=INPUTS + OUTPUTS, dtype=np.float32, nrows=N)
data.to_csv('./data-' + "{:.0e}".format(N) + '.csv', index=False)