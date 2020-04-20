from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from sklearn import svm
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


y = dataframe["RESULT"]
#dataframe = dataframe.drop(["RESULT"], axis=1)
# demonstrate data normalization with sklearn

###scaler = MinMaxScaler()
#dataframe = scaler.fit_transform(dataframe)

#binner = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
#dataframe = binner.fit_transform(dataframe)
#fwd = binner.fit_transform(fwd)

X_train, X_test, y_train, y_test = train_test_split(dataframe, y, test_size=0.2)

clf = svm.SVC(kernel='linear', C=0.01)

# Train Decision Tree Classifer
h = clf.fit(X_train,y_train)

##print(h)
y_pred = clf.predict(X_test)

matrix = confusion_matrix(y_test, y_pred)
print(matrix)
