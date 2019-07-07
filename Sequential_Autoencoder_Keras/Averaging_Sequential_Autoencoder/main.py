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
epochs = 50
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

autoencoder1=Model(input_img, Decoder_part(Encoder_part(input_img)))
autoencoder2=Model(input_img, Decoder_part(Encoder_part(input_img)))
autoencoder3=Model(input_img, Decoder_part(Encoder_part(input_img)))
autoencoder4=Model(input_img, Decoder_part(Encoder_part(input_img)))
autoencoder5=Model(input_img, Decoder_part(Encoder_part(input_img)))
autoencoder_final=Model(input_img, Decoder_part(Encoder_part(input_img)))

#encoder = Model(input_img, Encoder_part(input_img))

autoencoder1.compile(loss='mean_squared_error', optimizer = SGD())
autoencoder2.compile(loss='mean_squared_error', optimizer = SGD())
autoencoder3.compile(loss='mean_squared_error', optimizer = SGD())
autoencoder4.compile(loss='mean_squared_error', optimizer = SGD())
autoencoder5.compile(loss='mean_squared_error', optimizer = SGD())
autoencoder_final.compile(loss='mean_squared_error', optimizer = SGD())

autoencoder1.summary()
autoencoder2.summary()
autoencoder3.summary()
autoencoder4.summary()
autoencoder5.summary()
autoencoder_final.summary()

batch_train_X = []
batch_train_ground = []

for i in train_X:
  count = 0
  x = []
  while(count<1000):
    x.append(i)
    count = count+1  
  batch_train_X.append(x)

for i in train_ground:
  count = 0
  x = []
  while(count<1000):
    x.append(i)
    count = count+1  
  batch_train_ground.append(x)
  
  valid_X = valid_X[:1000]
  valid_ground = valid_ground[:1000]

count = 0
number = 0
for i,j in zip(batch_train_X,batch_train_ground):
  number = number + 1000
  print("First", number, "of 40000")
  _, _, v_x, v_y = train_test_split(valid_X, valid_ground, test_size=0.02,random_state=13)
  i = [i]
  j = [j]
  if count == 0:
    autoencoder_train1 = autoencoder1.fit(i,j, batch_size=1,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))
  if count == 1:
    autoencoder_train2 = autoencoder2.fit(i,j, batch_size=1,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))
  if count == 2:  
    autoencoder_train3 = autoencoder3.fit(i,j, batch_size=1,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))
  if count == 3:  
    autoencoder_train4 = autoencoder4.fit(i,j, batch_size=1,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))
  if count ==4:  
    autoencoder_train5 = autoencoder5.fit(i,j, batch_size=1,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))
  if count ==2:
    count = 0
    w1 = autoencoder1.get_weights()
    w2 = autoencoder2.get_weights()
    w3 = autoencoder3.get_weights()
    w4 = autoencoder4.get_weights()
    w5 = autoencoder5.get_weights()
    w_final = np.mean(w1,w2)

    w_cur = autoencoder_final.get_weights()
    w_final = mean(w_cur,w_final)
    w_final.set_weights(w_final)

    autoencoder_final.set_weights(w_final)
    autoencoder1.set_weights(w_final)
    autoencoder2.set_weights(w_final)
    autoencoder3.set_weights(w_final)
    autoencoder4.set_weights(w_final)
    autoencoder5.set_weights(w_final)


loss = autoencoder_train1.history['loss']
val_loss = autoencoder_train1.history['val_loss']
epochs = range(100)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

loss = autoencoder_train2.history['loss']
val_loss = autoencoder_train2.history['val_loss']
epochs = range(100)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
