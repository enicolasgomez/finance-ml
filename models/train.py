import pandas as pd
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

Data = pd.read_csv('MSFT.csv',header=0, usecols=['Date',
          'Close'],parse_dates=True,index_col='Date')

print(Data.info())
print(Data.head())
print(Data.describe())

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(Data)
plt.show()

DataPCh = Data.pct_change()

import numpy as np
LogReturns = np.log(1 + DataPCh)
print(LogReturns.tail(10))

plt.figure(figsize=(10,5))
plt.plot(LogReturns)
plt.show()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
DataScaled = scaler.fit_transform(Data)

TrainLen = int(len(DataScaled) * 0.70)
TestLen = len(DataScaled) - TrainLen
TrainData = DataScaled[0:TrainLen,:]
TestData = DataScaled[TrainLen:len(DataScaled),:]

print(len(TrainData), len(TestData))

def DatasetCreation(dataset, TimeStep=1):
   DataX, DataY = [], []
   for i in range(len(dataset)- TimeStep -1):
         a = dataset[i:(i+ TimeStep), 0]
         DataX.append(a)
         DataY.append(dataset[i + TimeStep, 0])
   return np.array(DataX), np.array(DataY)

TimeStep = 15
TrainX, TrainY = DatasetCreation(TrainData, TimeStep)
TestX, TestY = DatasetCreation(TestData, TimeStep)

TrainX = np.reshape(TrainX, (TrainX.shape[0], 1, TrainX.shape[1]))
TestX = np.reshape(TestX, (TestX.shape[0], 1, TestX.shape[1]))

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

model = Sequential()

model.add(LSTM(256, input_shape=(1, TimeStep)))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

model.fit(TrainX, TrainY, epochs=1, batch_size=1, verbose=1)

model.summary()

score = model.evaluate(TrainX, TrainY, verbose=0)
print('Keras Model Loss = ',score[0])
print('Keras Model Accuracy = ',score[1])

TrainPred = model.predict(TrainX)
TestPred = model.predict(TestX)

#TrainPred = scaler.inverse_transform(TrainPred)
#TrainY = scaler.inverse_transform([TrainY])

TestPred = scaler.inverse_transform(TestPred)
TestY = scaler.inverse_transform([TestY])

totalMatches = 0

for i in range(0, len(TestPred)-2):
      #if ((TestPred[i+1]-TestPred[i] > 0) and (TestY[0][i+1]-TestY[0][i] > 0)) or ((TestPred[i+1]-TestPred[i] < 0) and (TestY[0][i+1]-TestY[0][i] < 0)):
      #      totalMatches = totalMatches + 1
      print(abs(TestPred[i]))
      print(abs(TestY[0][i] ))
      print()
