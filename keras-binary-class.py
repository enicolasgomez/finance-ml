from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import KBinsDiscretizer
import keras as K
import numpy as np
import pandas as pd

from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.preprocessing import MinMaxScaler

import os
import sys
import math

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # suppress CPU msg

# load dataset
dataframe = read_csv(os.path.join(sys.path[0], "processed.csv"), delim_whitespace=False, header=0)
dataframe = dataframe[ abs(dataframe["CLOSINGPIPS"]) > 0.002 ]

y = dataframe["CLOSINGPIPS"].to_numpy()
dataframe = dataframe.drop(["CLOSINGPIPS"], axis=1)

output_classes = 2
binner = KBinsDiscretizer(n_bins=output_classes, encode='ordinal' )
y = y.reshape(-1, 1)
y_bin = binner.fit_transform(y)
y_categorical = np_utils.to_categorical(y_bin, num_classes=output_classes)

#scaler = MinMaxScaler()
#dataframe = scaler.fit_transform(dataframe)

binner = KBinsDiscretizer(n_bins=30, encode='ordinal', strategy='uniform')
dataframe = binner.fit_transform(dataframe)

X_train, X_test, y_train, y_test = train_test_split(dataframe, y_categorical, test_size=0.3, shuffle=True)

class log(K.callbacks.Callback):
  def __init__(self, n):
    self.n = n   # print loss & acc every n epochs

  def on_epoch_end(self, epoch, logs={}):
    if epoch % self.n == 0:
      curr_loss =logs.get('loss')
      curr_acc = logs.get('accuracy') * 100
      print("epoch = %4d loss = %0.6f accuracy = %0.2f%%" % \
        (epoch, curr_loss, curr_acc))

max_epochs = 2000
batch_size = 5
my_logger = log(n=5)

my_init = K.initializers.glorot_uniform(seed=1)
model = K.models.Sequential()

model.add(K.layers.Dense(units = 54, input_dim = 54, activation='relu'))
model.add(K.layers.Dense(units = output_classes, activation='softmax'))
simple_sgd = K.optimizers.Adam()
model.compile(loss='categorical_crossentropy', optimizer=simple_sgd, metrics=['accuracy'])  

h = model.fit(X_train
            , y_train
            , epochs=max_epochs
            , batch_size=batch_size
            , verbose=0
            , callbacks=[my_logger])
print(h)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
#print('\n# Generate predictions for 20 samples')
predictions = model.predict(X_test) #evaluate_y for checks
#print('predictions shape:', predictions.shape)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

#results = model.evaluate(X_test, predictions)
#print('Test loss:', results[0])
#print('Test accuracy:', results[1])

matrix = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
print(matrix)
