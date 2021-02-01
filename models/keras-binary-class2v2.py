import pandas as pd
import sklearn as sklearn
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import datetime 

df = pd.read_csv("C:\\Dumps\\8\\train.csv", header=0)
#df = shuffle(df)
print(df.head())

df.INIT = pd.to_datetime(df.INIT)

train = df[ df.INIT < datetime.datetime(2017, 1, 1) ]
test = df[ df.INIT >= datetime.datetime(2017, 1, 1) ]

cols = ['WILLR_1','AD_1','ADOSC_1','OBV_1','MFI_1','AD_BID_1','ADOSC_BID_1','OBV_BID_1','MFI_BID_1']

X = train[cols].values
y = train.TOFILTER.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=95)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
encoder = LabelEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train)

encoder = LabelEncoder()
encoder.fit(y_test)
y_test = encoder.transform(y_test)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(9, input_dim=9, kernel_initializer='normal', activation='relu'))
# model.add(Dense(15, activation='relu'))
# model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=3, batch_size=1)
test_loss, test_acc = model.evaluate(X_test, y_test)

model_json = model.to_json()

with open("C:\\Dumps\\model\\model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("C:\\Dumps\\model\\model.h5")
print("Saved model to disk")

#forward testing 

X_fwd = test[cols].values
y_fwd = test.TOFILTER.values

encoder = LabelEncoder()
encoder.fit(y_fwd)
y_fwd = encoder.transform(y_fwd)

y_pred = model.predict(X_fwd)

encoder = LabelEncoder()
encoder.fit(y_pred)
y_pred = encoder.transform(y_pred)

from sklearn.metrics import confusion_matrix

#y_pred = np.where(y_pred > y_pred.max() * 0.05, 1, 0)
y_pred = np.where(y_pred > 100, 1, 0)
confusion_matrix = sklearn.metrics.confusion_matrix(y_fwd, y_pred)
print(confusion_matrix)

#simulation
import random 

raw_df = test 
raw_df['PREDICTED'] = y_pred

cols = ['INIT','OPEN','CLOSE','HIGH','LOW']

raw_fd = raw_df[cols]

target = 10 / 10000
filteredprofit = 0
nonfilteredprofit = 0

totalfiltered = 0
totalfilteredprofit = 0

totalnonfiltered = 0
totalnonfilteredprofit = 0

for i in range(0, len(raw_df)):    
    is_sell = bool(random.getrandbits(1)) 

    real_profit = abs(raw_df.CLOSE.values[i] - raw_df.OPEN.values[i])
    if is_sell:
        max_profit = raw_df.OPEN.values[i] - raw_df.LOW.values[i]
        if raw_df.CLOSE.values[i] > raw_df.OPEN.values[i]:
            real_profit = real_profit * -1
    else:
        max_profit = raw_df.HIGH.values[i] - raw_df.OPEN.values[i]
        if raw_df.CLOSE.values[i] < raw_df.OPEN.values[i]:
            real_profit = real_profit * -1

    if max_profit > target :
        result = target
    else:
        result = real_profit

    if y_pred[i] == 0:
        filteredresult = result
        nonfilteredresult = result
        filteredprofit = filteredprofit + result
        nonfilteredprofit = nonfilteredprofit + result
        totalfiltered = totalfiltered + 1
        if result > 0 :
            totalfilteredprofit = totalfilteredprofit + 1
    else:
        filteredresult = 0
        nonfilteredresult = result
        nonfilteredprofit = nonfilteredprofit + result
        totalnonfiltered = totalnonfiltered + 1
        if result > 0:
            totalnonfilteredprofit = totalnonfilteredprofit + 1

    raw_df.at[i, 'FILTEREDRESULT'] = filteredresult
    raw_df.at[i, 'NONFILEREDRESULT'] = nonfilteredresult

import matplotlib.pyplot as plt
import numpy as np; np.random.seed(1)
plt.rcParams["figure.figsize"] = 5,2

#x = np.linspace(-3,3)
#y = np.cumsum(np.random.randn(50))+6
#y = raw_df['FILTEREDRESULT']

#fig, ax = plt.subplots(nrows=1, sharex=True)

#extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
#ax.imshow(y[np.newaxis,:], aspect="auto")
#ax.set_yticks([])
#ax.set_xlim(extent[0], extent[1])

#plt.tight_layout()
#plt.show()    

print(filteredprofit)
print(totalfilteredprofit/totalfiltered)
print(nonfilteredprofit)
print(totalnonfilteredprofit/totalnonfiltered)

raw_df.to_csv("C:\\Dumps\\6\\predicted.csv")
