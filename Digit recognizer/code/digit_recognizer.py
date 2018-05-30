import json
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Activation, MaxPooling2D, Flatten
from keras.regularizers import l2
from keras.optimizers import SGD
from keras import utils
import matplotlib.pyplot as plt
import keras.backend as K
from keras.utils import np_utils
import csv

with open('../data/test.csv') as f:
    reader = csv.reader(f)
    next(reader) # skip header
    test_data = np.array([r for r in reader],dtype="float64")

with open('../data/train.csv') as f:
    reader = csv.reader(f)
    next(reader) # skip header
    train_data = np.array([r for r in reader],dtype="float64")

# The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user.
# The rest of the columns contain the pixel-values of the associated image.
# Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783, inclusive.
# To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive.
# Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero).

# The test data set, (test.csv), is the same as the training set, except that it does not contain the "label" column.

num_classes = 10

x_train = train_data[:, 1:] # Take all the columns except for the first
x_train = np.divide(x_train, 255.0) # divide by 255 to get values between 0 and 1
y_train = train_data[:, 0] # The first column contains the labels
y_train = utils.to_categorical(y_train, num_classes=num_classes)

x_test = test_data # Test data does not contain the label column
x_test = np.divide(x_test, 255.0) # divide by 255 to get values between 0 and 1

# Reshape.
# The first column is the number of samples. -1 means that we let numpy figure this out itself.
# The second and third column are the width and height of the image in pixels.
# The third column is the number of layers in the picture.
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

#Generate the model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1))) # Input shape = dimensions of the image, layers. The conv2d layer uses a 3x3 kernel and 32 hidden neurons.
model.add(Activation('relu')) # Relu means negative becomes 0
model.add(Conv2D(32,(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # Max pooling takes the highest pixel value for a kernel of 2x2, this is meant to reduce the number of inputs
model.add(Dropout(0.20)) # The dropout disables a percentage of connections, to reduce overfitting.

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))

model.add(Flatten()) # Reduces the 2D input to a single vector
model.add(Dense(512)) # Dense layer that maps the results of the conv layers to the output
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model, split into 30% validation, and 70% training
history = model.fit(x_train, y_train, epochs=25, batch_size=128, validation_split=0.20, shuffle=True)

# Make predictions on the test set. We use argmax to map the vector of probabilities to a single category.
predictions = np.argmax(model.predict(x_test), axis = 1)

# Write the results to a file (floydhub preserves the output folder)
file = open("output/output.csv","w")

file.write("ImageId,Label\n")
for (index, prediction) in enumerate(predictions):
    file.write(str(index + 1) + "," + str(prediction)+"\n")

file.close() 

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for mean error
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()