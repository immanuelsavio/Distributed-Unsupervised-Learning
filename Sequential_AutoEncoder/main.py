from __future__ import print_function, division
from keras.datasets import mnist
import math
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np 
from optimizers import Adam
from loss_function import CrossEntropy, SquareLoss
import progressbar
from layers import Dense, Dropout, Flatten, Activation, Reshape, BatchNormalization
from NeuralNetwork import NeuralNetwork

class Autoencoder():

    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.img_dim = self.img_rows * self.img_cols
        self.latent_dim = 128 # The dimension of the data embedding
        self.count = 0

        mnist = fetch_openml('mnist_784')

        self.X = mnist.data
        self.y = mnist.target

        # Rescale [-1, 1]
        self.X = (self.X.astype(np.float32) - 127.5) / 127.5
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2)

        optimizer = Adam(learning_rate=0.0002, b1=0.5)
        loss_function = SquareLoss

        self.encoder = self.build_encoder(optimizer, loss_function)
        self.decoder = self.build_decoder(optimizer, loss_function)
        self.classifier = self.build_classifier(optimizer, loss_function)

        self.autoencoder = NeuralNetwork(optimizer=optimizer, loss=loss_function)
        self.autoencoder.layers.extend(self.encoder.layers)
        self.autoencoder.layers.extend(self.decoder.layers)
        self.autoencoder2 = self.autoencoder
        self.autoencoder_final = self.autoencoder

        print ()
        self.autoencoder.summary(name="Variational Autoencoder")


    def build_encoder(self, optimizer, loss_function):

        encoder = NeuralNetwork(optimizer=optimizer, loss=loss_function)
        encoder.add(Dense(512, input_shape=(self.img_dim,)))
        encoder.add(Activation('leaky_relu'))
        encoder.add(BatchNormalization(momentum=0.8))
        encoder.add(Dense(256))
        encoder.add(Activation('leaky_relu'))
        encoder.add(BatchNormalization(momentum=0.8))
        encoder.add(Dense(self.latent_dim))

        return encoder

    def build_decoder(self, optimizer, loss_function):

        decoder = NeuralNetwork(optimizer=optimizer, loss=loss_function)
        decoder.add(Dense(256, input_shape=(self.latent_dim,)))
        decoder.add(Activation('leaky_relu'))
        decoder.add(BatchNormalization(momentum=0.8))
        decoder.add(Dense(512))
        decoder.add(Activation('leaky_relu'))
        decoder.add(BatchNormalization(momentum=0.8))
        decoder.add(Dense(self.img_dim))
        decoder.add(Activation('tanh'))

        return decoder

    def build_classifier(self, optimizer, loss_function):
           
        classifier = NeuralNetwork(optimizer=optimizer, loss=loss_function)
        classifier.add(Dense(10, input_shape=(self.img_dim,)))
        classifier.add(Activation('softmax'))

        return classifier

    def update_weights(a,b,c):
        for layer, layer1, layer2 in reversed(a.layers), reversed(b.layers), reversed(c.layers):
            w1 = layer.get_weight_layer()
            w2 = layer1.get_weight_layer()
            w3 = np.average([w1,w2])
            layer2.update_weight_final(w3)
            layer1.update_weight_final(w3)
            layer.update_weight_final(w3)


    def train(self, n_epochs, batch_size=128):
        mutex = 0
        chance = 0
        for epoch in range(n_epochs):
            if(chance == 2):
                chance = 0
            # Select a random half batch of images
            idx_train = np.random.randint(0, self.X_train.shape[0], batch_size)
            imgs_train = self.X_train[idx_train]
            lab_train = self.y_train[idx_train]


            # Train the Autoencoder
            while(mutex == 0):
                loss, acc = self.autoencoder.train_on_batch(imgs_train, imgs_train)
                mutex = 1
                chance = chance + 1
            while(mutex == 1):
                chance = chance + 1 
                loss, acc = self.autoencoder2.train_on_batch(imgs_train, imgs_train)
                mutex = 0

            if (mutex == 2):
                self.update_weights(self.autoencoder, self.autoencoder2, self.autoencoder_final)
            # Display the progress
            print ("%d [D loss: %f]" % (epoch, loss))

            

    def train_classifier (self, n_epochs=5, batch_size=32):

        classifier = self.autoencoder.set_trainable(True)
        classifier = self.autoencoder.layers.extend(self.classifier.layers)
        optimizer = Adam(learning_rate=0.0002, b1=0.5)
        loss_function = SquareLoss

        self.build_classifier(optimizer, loss_function)
        for epoch in range(n_epochs):
            idx_train = np.random.randint(0, self.X_train.shape[0], batch_size)
            imgs_train = self.X_train[idx_train]
            lab_train = self.y_train[idx_train]

            idx_test = np.random.randint(0, self.X_test.shape[0], batch_size)
            imgs_test = self.X_test[idx_test]
            lab_test = self.y_test[idx_test]

            loss, acc = self.autoencoder.train_on_batch(imgs_train, lab_train)
            print ("%d [D loss: %f] and accuracy: %f" % (epoch, loss, acc))


if __name__ == '__main__':
    ae = Autoencoder()
    ae.train(n_epochs=200000, batch_size=64)
    #ae.train_classifier(n_epochs= 5, batch_size= 32)
