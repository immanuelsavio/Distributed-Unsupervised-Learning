#imports and modules
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import keras
import matplotlib.pyplot as plt
import numpy as np
import gzip
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import fashion_mnist
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

#def data_reading():
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, x_test.shape)
#data_reading()

label_dict = {
 0: 'A',
 1: 'B',
 2: 'C',
 3: 'D',
 4: 'E',
 5: 'F',
 6: 'G',
 7: 'H',
 8: 'I',
 9: 'J',
}

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
print(x_train.shape)
print(x_test.shape)

print(np.max(x_train))
print(np.max(x_test))

''' scaling it in the range of 0 - 1 by dividing it with the maximum'''
x_train = x_train/np.max(x_train)
x_test = x_test/np.max(x_test)

print(np.max(x_test), np.max(x_train))

train_x, valid_x, train_ground, test_ground = train_test_split(x_train, x_train, test_size = 0.2, random_state = 12)
print(train_x.shape, valid_x.shape)

batch_size = 64
epochs = 200
inChannel = 1
x, y = 28, 28
input_img = Input(shape = (x, y, inChannel))
num_classes = 10
