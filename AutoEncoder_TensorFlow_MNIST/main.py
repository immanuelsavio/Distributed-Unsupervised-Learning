#!/usr/bin/python
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

session = tf.Session()
df = input_data.read_data_sets("MNIST_data", one_hot=True)

num_inputs = 784
num_h1 = 256
num_h2 = 128

X = tf.placeholder("float", shape = [None,num_inputs])


#Necessary inputs
batch_size = 64
num_steps = 40000
learning_rate = 5e-1
display_step = 1000

def encoder_layer(x):
    l1 = tf.matmul(x, W["w1"])
    l1 = tf.add(l1,b["b1"])
    l1 = tf.nn.sigmoid(l1)

    l1 = tf.matmul(l1, W["w2"])
    l1 = tf.add(l1,b["b2"])
    l1 = tf.nn.sigmoid(l1)

    return l1

def decoder_layer(x):
    l2 = tf.matmul(x, W["w3"])
    l2 = tf.add(l2,b["b3"])
    l2 = tf.nn.sigmoid(l2)

    l2 = tf.matmul(l2, W["w4"])
    l2 = tf.add(l2,b["b4"])
    l2 = tf.nn.sigmoid(l2)

    return l2

#defining the weight matrices
W = {"w1" : tf.Variable(tf.random_normal([num_inputs, num_h1])),
     "w2" : tf.Variable(tf.random_normal([num_h1,num_h2])),
     "w3" : tf.Variable(tf.random_normal([num_h2,num_h1])),
     "w4" : tf.Variable(tf.random_normal([num_h1, num_inputs]))}

b = { "b1" : tf.Variable(tf.random_normal([num_h1])),
      "b2" : tf.Variable(tf.random_normal([num_h2])),
      "b3" : tf.Variable(tf.random_normal([num_h1])),
      "b4" : tf.Variable(tf.random_normal([num_inputs]))} 

encoder_fun = encoder_layer(X)
decoder_fun = decoder_layer(encoder_fun)

prediction = decoder_fun
actual = X

cost_fun = tf.reduce_mean(tf.pow(actual - prediction, 2))
optim = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optim.minimize(cost_fun)

loss_batch = []
for i in range(0,num_steps):
    rand_index = np.random.choice(100, batch_size)
    x_batch = 



init = tf.global_variables_initializer()
sess.run(init)

