import pandas as pd
import sklearn as sklearn
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import datetime

forward = True

df = pd.read_csv("C:\\Dumps\\processed.csv", header=0)
df = shuffle(df)

# train = df[ df.INIT < datetime.datetime(2017, 1, 1) ]
# test = df[ df.INIT >= datetime.datetime(2017, 1, 1) ]

print(df.head())

#cols = ['WILLR_1','BB_LOW_1','OPENASKVOL','OPENBIDVOL','ATR_1']
#cols = ['RSI_1','WILLR_1','ADX_1']

# cols = ['MFI_CLOSEBID_1',
# 'SRSI_D2',
# 'WILLR_4',
# # 'MFI_CLOSE_ASK_2',
# # 'MFI_CLOSE_ASK_1',
# # 'MFI_CLOSE_ASK_4',
# 'WILLR_5',
# 'MFI_BID_1',
# 'MFI_BID_3',
# 'SRSI_K5',
# 'SRSI_K1',
# 'SRSI_D3',
# #'MFI_CLOSEBID_4',
# 'ADX_1',
# 'ADX_4']

cols = ['RSI_5',
'SRSI_D1',
'MFI_2',
'RSI_3',
'SRSI_K3',
'SRSI_K1',
'SRSI_D3',
'SRSI_K5',
'MFI_CLOSEBID_2',
'RSI_1',
'MFI_4',
'MFI_CLOSEBID_4',
'SRSI_D2',
'ADX_4']

X = df[cols].values
y = df.TOFILTER.values

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
model.add(Dense(9, input_dim=9, kernel_initializer='normal', activation='relu'))
model.add(Dense(25, input_dim=5, kernel_initializer='normal', activation='relu'))
model.add(Dense(5, input_dim=5, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

# Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=1)
test_loss, test_acc = model.evaluate(X_test, y_test)

from sklearn.metrics import confusion_matrix

ypred_data = model.predict(X_test)
ypred_data = np.where(ypred_data > ypred_data.max() * 0.9, 1, 0)
cm3 = confusion_matrix(ypred_data, y_test) 
print(cm3)

# test = pd.read_csv("C:\\Dumps\\processed.csv", header=0)
# test = shuffle(df)

# X_fwd = test[cols].values
# y_fwd = test.TOFILTER.values

# encoder = LabelEncoder()
# encoder.fit(y_fwd)
# y_fwd = encoder.transform(y_fwd)

# y_pred = model.predict(X_fwd)

# encoder = LabelEncoder()
# encoder.fit(y_pred)
# y_pred = encoder.transform(y_pred)

# #y_pred = np.where(y_pred > y_pred.max() * 0.05, 1, 0)
# y_pred = np.where(y_pred > 100, 1, 0)
# confusion_matrix = sklearn.metrics.confusion_matrix(y_fwd, y_pred)
# print(confusion_matrix)