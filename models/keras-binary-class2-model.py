import pandas as pd
import sklearn as sklearn
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

forward = True

df = pd.read_csv("C:\\Dumps\\10-12h-100p\\train.csv", header=0)
df = shuffle(df)
print(df.head())

cols = ['WILLR_1','BB_LOW_1','BB_MID_1','BB_UP_1','OPENASKVOL','OPENBIDVOL','ATR_1']

X = df[cols].values
y = df.ATR_1.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=95)

from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
encoder = LabelEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train)

encoder = LabelEncoder()
encoder.fit(y_test)
y_test = encoder.transform(y_test)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(7, input_dim=7,  activation='relu'))
model.add(Dense(7, input_dim=7,  activation='relu'))
# model.add(Dense(15, activation='relu'))
# model.add(Dense(10, activation='relu'))
#model.add(Dense(5, activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=1)
test_loss, test_acc = model.evaluate(X_test, y_test)

print(y_train)
print(y_test)