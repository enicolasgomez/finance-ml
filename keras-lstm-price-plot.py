#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import tensorflow as tf

import os
import sys
import keras as K
import numpy as np
import pandas as pd

from pandas import read_csv

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

#creating dataframe
df = read_csv(os.path.join(sys.path[0], "processed.csv"), delim_whitespace=False, header=0)

inputs = df.tail(100)

scaler = MinMaxScaler(feature_range=(0, 1))
inputs['Close'] = scaler.fit_transform(inputs['Close'])

y_fwd = inputs['Close']
inputs = inputs.drop('Close', axis=1, inplace=True)

closing_price = model.predict(inputs)
closing_price = scaler.inverse_transform(closing_price)

import matplotlib.pyplot as plt
plt.figure(figsize=(16,8))
plt.plot(y_fwd)
plt.plot(closing_price)
plt.show()