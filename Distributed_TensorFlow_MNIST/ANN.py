from autoencoder import *

#Calling the function
autoencoder()
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