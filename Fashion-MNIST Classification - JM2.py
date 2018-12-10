#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 19:03:51 2018

@author: Jake
"""

'''
Using TensorFlow to classify MNIST dataset
'''

import tensorflow as tf
from keras.datasets import fashion_mnist

############################
# Utility/Helper Functions #
############################
def create_interactive_session():
    '''
    Creates interactive TensorFlow session
    '''
    return tf.InteractiveSession()

def create_placeholders():
    '''
    Creates x, y placeholders
    x: inputs
    y_: labels
    '''
    x = tf.placeholder(tf.float32, shape = [None, 28*28])
    y_ = tf.placeholder(tf.float32, shape = [None, 10])
    return x, y_

def initialize_variables(tf_session):
    '''
    Initializes variables of a session
    '''
    tf_session.run(tf.global_variables_initializer())

def convert_image_to_tensor(image):
    '''
    Converts input image to flat tensor
    '''
    return tf.reshape(image, [-1, 28, 28, 1])

#####################
# Functions for CNN #
#####################
def create_datasets():
    '''
    Creates FashionMNIST dataset
    Creates "train_dataset" and "validation_dataset"
    Currently saves to "./JM1"
    '''
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    return x_train, y_train, x_test, y_test

def create_weights_bias(dimensions):
    '''
    Creates weights and bias to use in matrix multiplication
    '''
    W = tf.Variable(tf.truncated_normal(dimensions))
    b = tf.Variable(tf.constant(0.1, shape = [dimensions[-1]]))
    return W, b

def apply_conv_layer(x_input, kernel, strides, padding = 'SAME'):
    '''
    Applies convolutional layer, to be used when training
    kernel: [height, width, channels in, channels out]
    strides:  [batch, height, width, channels]
    padding: "SAME" (default) or "VALID"
    '''
    W, b = create_weights_bias(kernel)
    return tf.nn.conv2d(x_input, W, strides, padding) + b

def apply_relu_layer(x_input):
    '''
    Apply ReLU layer, to be applied when training
    '''
    return tf.nn.relu(x_input)

def apply_max_pool_layer(x_input, kernel, strides, padding = 'SAME'):
    '''
    Applies maxpool layer, to be used when training
    kernel: [height, width, channels in, channels out]
    strides:  [batch, height, width, channels]
    padding: "SAME" (default) or "VALID"
    '''
    return tf.nn.max_pool(x_input, kernel, strides, padding)

def dropout(x_input, keep_prob = None):
    '''
    Applies dropout with "probability"
    '''
    if keep_prob == None:
        keep_prob = tf.placeholder(tf.float32)
    return tf.nn.dropout(x_input, keep_prob)

def flatten_layer(x_input):
    '''
    Flattens x_input matrix into 1-D tensor
    Uses dimensions from x_input
    '''
    return tf.reshape(x_input, [-1,
                                x_input.shape[1].value *
                                x_input.shape[2].value *
                                x_input.shape[3].value])

def fully_connected_layer(x_input, dimensions):
    '''
    Multiples matrices of flattened layer and weights, adds bias
    '''
    W, b = create_weights_bias(dimensions)
    return tf.matmul(x_input, W) + b

def apply_softmax(x_input):
    '''
    Applies softmax to x_input
    '''
    return tf.nn.softmax(x_input)
    
def loss_function(y_, y_input):
    '''
    Applies cross entropy loss function
    '''
    return tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_input),
                                         reduction_indices=[1]))

def optimizer(loss):
    '''
    Applies Adam optimizer to loss
    '''
    return tf.train.AdamOptimizer(0.0001).minimize(loss)

def accuracy(y_, y_input):
    '''
    Returns average of correct predictions
    '''
    correct = tf.equal(tf.argmax(y_input, 1), tf.argmax(y_, 1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))

###########
# Program #
###########
try:
    InteractiveSession.close()
except NameError:
    pass

sess = create_interactive_session()

x_train, y_train, x_test, y_test = create_datasets()

x, y_ = create_placeholders()
#x_tensor = convert_image_to_tensor(x)
x_tensor = convert_image_to_tensor(x_train)

conv1 = apply_conv_layer(x_tensor,
                         kernel = [5, 5, 1, 32],
                         strides = [1, 1, 1, 1])
relu1 = apply_relu_layer(conv1)
max_pool1 = apply_max_pool_layer(relu1,
                                 kernel = [1, 2, 2, 1],
                                 strides = [1, 2, 2, 1])
dropout1 = dropout(max_pool1, keep_prob = 0.5)
conv2 = apply_conv_layer(dropout1,
                         kernel = [5, 5, 32, 64],
                         strides = [1, 1, 1, 1])
relu2 = apply_relu_layer(conv2)
max_pool2 = apply_max_pool_layer(relu2,
                                 kernel = [1, 2, 2, 1],
                                 strides = [1, 2, 2, 1])
dropout2 = dropout(max_pool2, keep_prob = 0.5)
flatten2 = flatten_layer(dropout2)
fc3 = fully_connected_layer(flatten2,
                            dimensions = [flatten2.shape[1].value,
                                          10])
softmax3 = apply_softmax(fc3)
loss = loss_function(y_, softmax3)
optimize = optimizer(loss)

epochs = 5

for epoch in range(epochs):
    print('Evaluating epoch: ' + str(epoch + 1))
    train_accuracy = accuracy(y_, softmax3).eval(feed_dict =
                                                {x : x_train,
                                                 y_ : y_train})
#                                                 keep_prob : 1.0})
    print(train_accuracy)
    optimizer(loss).run(feed_dict = {x : x_train,
                                     y_ : y_train})
#                                     keep_prob : 0.5})