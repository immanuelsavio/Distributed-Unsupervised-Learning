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

autoencoder_final.save_weights('autoencoder.h5')

x = autoencoder_final.get_weights()
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

for l1,l2 in zip(full_model.layers[:6],autoencoder_final.layers[0:6]):
    l1.set_weights(l2.get_weights())

autoencoder_final.get_weights()[0][1]

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

predicted_classes = full_model.predict(X_test)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
predicted_classes.shape, y_test.shape

correct = np.where(predicted_classes==y_test)[0]
print("Found %d correct labels" % len(correct))

incorrect = np.where(predicted_classes!=y_test)[0]
print ("Found %d incorrect labels" % len(incorrect))