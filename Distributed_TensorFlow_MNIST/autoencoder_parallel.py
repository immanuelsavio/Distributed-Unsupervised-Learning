from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense
from keras.optimizers import RMSprop
from keras import backend as K
from keras.models import Model
from keras.datasets import mnist
from keras import regularizers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import tensorflow.examples.tutorials.mnist.input_data as input_data
import time

#Global variable Declarations

FLAGS = None
REPLICAS_TO_AGGREGATE = 2
def plot_traincurve(history):
    colors = {'loss':'r', 'acc':'b', 'val_loss':'m', 'val_acc':'g'}
    plt.figure(figsize=(10,6))
    plt.title("Training Curve") 
    plt.xlabel("Epoch")

    for measure in history.keys():
        color = colors[measure]
        ln = len(history[measure])
        plt.plot(range(1,ln+1), history[measure], color + '-', label=measure)  # use last 2 values to draw line

    plt.legend(loc='upper left', scatterpoints = 1, frameon=False)


def main():

    # Configure
    config=tf.ConfigProto(log_device_placement=False)

    # Server Setup
    cluster = tf.train.ClusterSpec({
            'ps':['localhost:2222'],
            'worker':['localhost:2223','localhost:2224']
            }) #allows this node know about all other nodes
    if FLAGS.job_name == 'ps': #checks if parameter server
        server = tf.train.Server(cluster,
            job_name="ps",
            task_index=FLAGS.task_index,
            config=config)
        server.join()
    else: #it must be a worker server
        is_chief = (FLAGS.task_index == 0) #checks if this is the chief node
        server = tf.train.Server(cluster,
            job_name="worker",
            task_index=FLAGS.task_index,
            config=config)
    batch_size = 128
    num_classes = 10
    epochs = 30
    img_rows, img_cols = 28, 28

    worker_device = "/job:%s/task:%d/cpu:0" % (FLAGS.job_name,FLAGS.task_index)
    with tf.device(tf.train.replica_device_setter(ps_tasks=1,
          worker_device=worker_device)):


    # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        # this is the size of our encoded representations
        encoding_dim = 32 

        # this is our input placeholder
        input_img = Input(shape=(784,))
        # "encoded" is the encoded representation of the input
        encoded = Dense(encoding_dim, activation='relu')(input_img)
        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(784, activation='sigmoid')(encoded)

        # this model maps an input to its reconstruction
        autoencoder = Model(input_img, decoded)

        # this model maps an input to its encoded representation
        encoder = Model(input_img, encoded)

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(encoding_dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = autoencoder.layers[-1]
        # create the decoder model
        decoder = Model(encoded_input, decoder_layer(encoded_input))

        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        history = autoencoder.fit(x_train, x_train,
                        epochs=30,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(x_test, x_test))
    plot_traincurve(history.history)

    encoded_imgs_train = encoder.predict(x_train)
    encoded_imgs_test = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs_test)

    encoded_imgs_train_normalized = encoded_imgs_train / np.max(encoded_imgs_train)
    encoded_imgs_test_normalized = encoded_imgs_test / np.max(encoded_imgs_test)

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(encoding_dim,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])

    history = model.fit(encoded_imgs_train_normalized, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(encoded_imgs_test_normalized, y_test))
    score = model.evaluate(encoded_imgs_test_normalized, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    plot_traincurve(history.history)

if __name__ == "__main__":
    main()