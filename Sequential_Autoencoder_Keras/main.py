import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical


from keras.datasets import mnist



(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255
X_train = X_train.reshape(len(X_train), np.prod(X_train.shape[1:]))
X_test = X_test.reshape(len(X_test), np.prod(X_test.shape[1:]))
print(X_train.shape)
print(X_test.shape)

batch_size = 64
epochs = 100
inChannel = 1
x, y = 28, 28
input_img = Input(shape = (x, y, inChannel))
num_classes = 10



X_train = X_train/np.max(X_train)
X_test = X_test/np.max(X_test)

print(np.max(X_train))
train_X,valid_X,train_ground,valid_ground = train_test_split(X_train,
                                                             X_train,
                                                             test_size=0.2,
                                                             random_state=13)


input_img = Input(shape = (784,))
num_classes = 10

def Encoder_part(input_img):

    encoded = Dense(units=512, activation='relu')(input_img)
    encoded = Dense(units=256, activation='relu')(encoded)
    encoded = Dense(units=128, activation='relu')(encoded)
    encoded = Dense(units=64, activation='relu')(encoded)
    encoded = Dense(units=32, activation='relu')(encoded)

    return encoded

def Decoder_part(encoded):
        
    decoded = Dense(units=64, activation='relu')(encoded)
    decoded = Dense(units=128, activation='relu')(decoded)
    decoded = Dense(units=256, activation='relu')(decoded)
    decoded = Dense(units=512, activation='relu')(decoded)
    decoded = Dense(units=784, activation='relu')(decoded)
    
    return decoded

autoencoder=Model(input_img, Decoder_part(Encoder_part(input_img)))
encoder = Model(input_img, Encoder_part(input_img))

autoencoder.summary()
autoencoder.compile(loss='mean_squared_error', optimizer = SGD())
autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))

loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(100)

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

autoencoder.save_weights('autoencoder.h5')

x = autoencoder.get_weights()
print(x)

(_ , y_train), (_, y_test) = mnist.load_data()
train_Y_one_hot = to_categorical(y_train)
test_Y_one_hot = to_categorical(y_test)
print('Original label:', y_train[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

train_X,valid_X,y_train,y_valid = train_test_split(X_train,train_Y_one_hot,test_size=0.2,random_state=13)
train_X.shape,valid_X.shape,y_train.shape,y_valid.shape

def fc(enco):
    flat = enco
    den = Dense(128, activation='relu')(flat)
    out = Dense(num_classes, activation='softmax')(den)
    return out

full_model = Model(input_img,fc(Encoder_part(input_img)))

for l1,l2 in zip(full_model.layers[:6],autoencoder.layers[0:6]):
    l1.set_weights(l2.get_weights())

autoencoder.get_weights()[0][1]

full_model.get_weights()[0][1]

for layer in full_model.layers[0:6]:
    layer.trainable = False

full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

full_model.summary()

for layer in full_model.layers[0:19]:
    layer.trainable = False

classify_train = full_model.fit(train_X, y_train, batch_size=64,epochs=100,verbose=1,validation_data=(valid_X, y_valid))

full_model.save_weights('autoencoder_classification.h5')

for layer in full_model.layers[0:19]:
    layer.trainable = True

full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

classify_train = full_model.fit(train_X, y_train, batch_size=64,epochs=100,verbose=1,validation_data=(valid_X, y_valid))

full_model.save_weights('classification_complete.h5')

accuracy = classify_train.history['acc']

val_accuracy = classify_train.history['val_acc']

loss = classify_train.history['loss']

val_loss = classify_train.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

test_eval = full_model.evaluate(X_test, test_Y_one_hot, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])