import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.VariableScope("layer_parameter"):
        Weight = tf.Variable(tf.random_normal([in_size, out_size]), name="Weight")
        Basies = tf.Variable(tf.random_normal([1,out_size]), name="Basies")
        Wx_plus_b = tf.matmul(inputs, Weight)+Basies
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def comute_accuracy(v_xs, y_xs):
    global prediction
