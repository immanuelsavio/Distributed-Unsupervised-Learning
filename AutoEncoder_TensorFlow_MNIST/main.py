#!/usr/bin/python
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

df = input_data.read_data_sets("MNIST_data", one_hot=True)
print(df)