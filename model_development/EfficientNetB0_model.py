#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  16 11:28:12 2021

@author: nunesjoao
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
   
## EfficientNetB0 implementation for image classification testing ##
def EfficientNet_B0(height, width, channel, num_classes, learning_rate = 1e-3):
    """
    Pre-trained model on ImageNet dataset
    Input images need to have the following dimensions: (heigh - 224, width- 224, RBG channels - 3)
    """
    # Define training hyperparameters based on the number of classes
    if len(num_classes) == 2:
        parameters = {'Activation': 'sigmoid',
                      'Loss': 'binary_crossentropy',
                      'Metrics': 'binary_accuracy'}
    elif len(num_classes) >= 3:
        parameters = {'Activation': 'softmax',
                      'Loss': 'categorical_crossentropy',
                      'Metrics': 'categorical_accuracy'}

    X_input = tf.keras.Input(shape = (height, width, channel))
    model = tf.keras.applications.EfficientNetB0(include_top = False, input_tensor = X_input, weights = 'imagenet')

    # False will freeze the pre-trained weights
    model.trainable = True

    # Rebuild top layers to customize the number of classes
    X = GlobalAveragePooling2D(name = 'Global_Max_pool')(model.output)
    X = Dropout(0.2, name = 'Dropout-layer')(X)
    X_output = Dense(num_classes, activation = parameters['Activation'], name = 'Prediction-layer')(X)
    
    # Create model
    custom_model = Model(X_input, X_output, name = 'EfficientNetB0')
    custom_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
                         loss = parameters['Loss'],
                         metrics = parameters['Metrics'])
    return custom_model