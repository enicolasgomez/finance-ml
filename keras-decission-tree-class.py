from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
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
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # suppress CPU msg

# load dataset
dataframe = read_csv(os.path.join(sys.path[0], "candle_data_out_normalize_2.csv"), delim_whitespace=False, header=0)
dataframe = dataframe.drop(columns=["CLOSINGPIPS"], axis=1)


y = np_utils.to_categorical(dataframe["RESULT"], num_classes=2)
dataframe = dataframe.drop(["RESULT"], axis=1)
# demonstrate data normalization with sklearn

scaler = MinMaxScaler()
dataframe = scaler.fit_transform(dataframe)

#binner = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
#dataframe = binner.fit_transform(dataframe)
#fwd = binner.fit_transform(fwd)

X_train, X_test, y_train, y_test = train_test_split(dataframe, y, test_size=0.2)

class log(K.callbacks.Callback):
  def __init__(self, n):
    self.n = n   # print loss & acc every n epochs

  def on_epoch_end(self, epoch, logs={}):
    if epoch % self.n == 0:
      curr_loss =logs.get('loss')
      curr_acc = logs.get('binary_accuracy') * 100
      print("epoch = %4d loss = %0.6f accuracy = %0.2f%%" % \
        (epoch, curr_loss, curr_acc))

clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
h = clf.fit(X_train,y_train)

##print(h)
y_pred = clf.predict(X_test)

#print("Accuracy:",metrics.accuracy_score(y_fwd, y_pred))

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
#print('\n# Generate predictions for 20 samples')
#predictions = model.predict(fwd) #evaluate_y for checks
#print('predictions shape:', predictions.shape)

# serialize model to JSON
#model_json = model.to_json()
#with open("model.json", "w") as json_file:
    #json_file.write(model_json)
# serialize weights to HDF5
#model.save_weights("model.h5")
#print("Saved model to disk")

#results = model.evaluate(fwd, y_fwd)
#print('Test loss:', results[0])
#print('Test accuracy:', results[1])

matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(matrix)
