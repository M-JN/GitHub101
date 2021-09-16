#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sept  16 11:40:59 2021

@author: nunesjoao
"""
import os
import re
import pickle
import numpy as np

## Sort files within a folder the same way as the computer displays them (rather than the filesystem order) ##
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key = alphanum_key)

## Data normalization using the Z-score approach between [-1, 1] ##
def data_normalization(input_data):
    min_v = np.min(input_data)
    max_v = np.max(input_data)
    norm_data = 2*((input_data-min_v)/(max_v-min_v))-1
    
    return norm_data

## Reduces the loading time for big datasets by creating bin files ##
def bin_creation(variable, filename):
    with open(filename, 'wb') as f:
        pickle.dump(variable, f)
    print('{} has been created.'.format(filename))

## Load .bin files ##
def bin_load(filename):
    with open(filename, 'rb') as f:
        variable_name = pickle.load(f)
    return variable_name