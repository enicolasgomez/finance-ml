import pandas as pd
import sklearn as sklearn
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("C:\\Dumps\\train.csv", header=0)
df = shuffle(df)
print(df.head())

cols = ['ATR_1', 'WILLR_1','AD_1','ADOSC_1','OBV_1','MFI_1','AD_BID_1','ADOSC_BID_1','OBV_BID_1','MFI_BID_1']

X = df[cols].values
y = df.CLOSINGPIPS2.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=95)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# encoder = LabelEncoder()
# encoder.fit(y_train)
# y_train = encoder.transform(y_train)

# encoder = LabelEncoder()
# encoder.fit(y_test)
# y_test = encoder.transform(y_test)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=10, kernel_initializer='normal', activation='relu'))
# model.add(Dense(15, activation='relu'))
# model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='linear'))
# Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.
model.compile(loss='mse', optimizer='adam')

model.fit(X_train, y_train, epochs=20, batch_size=1)
#test_loss, test_acc = model.evaluate(X_test, y_test)

#model_json = model.to_json()

# with open("C:\\Dumps\\model\\model.json", "w") as json_file:
#     json_file.write(model_json)

# model.save_weights("C:\\Dumps\\model\\model.cont.h5")
# print("Saved model to disk")

# #forward testing 
df = pd.read_csv("C:\\Dumps\\fwd.csv", header=0)

X_fwd = df[cols].values
y_fwd = df.CLOSINGPIPS2.values

# encoder = LabelEncoder()
# encoder.fit(y_fwd)
# y_fwd = encoder.transform(y_fwd)

y_pred = model.predict(X_fwd)
df['PREDICTED'] = y_pred

df.to_csv("C:\\Dumps\\predicted.csv")
#y_pred = np.where(y_pred < ( y_pred.max() * 0.01), 1, 0)
#encoder = LabelEncoder()
#encoder.fit(y_pred)
#y_pred = encoder.transform(y_pred)

#y_pred = np.where(y_pred > 100, 1, 0)
#confusion_matrix = sklearn.metrics.confusion_matrix(y_fwd, y_pred)
#print(confusion_matrix)

#simulation
# import random 

# raw_df = pd.read_csv("C:\\Dumps\\fwd.csv", header=0)
# raw_df['PREDICTED'] = y_pred

# cols = ['INIT','OPEN','CLOSE','HIGH','LOW']

# raw_fd = raw_df[cols]

# target = 10 / 10000
# filteredprofit = 0
# nonfilteredprofit = 0

# for i in range(0, len(raw_df)):    
#     is_sell = bool(random.getrandbits(1)) 

#     real_profit = abs(raw_df.at[i, "CLOSE"] - raw_df.at[i, "OPEN"])
#     if is_sell:
#         max_profit = raw_df.at[i, "OPEN"] - raw_df.at[i, "LOW"]
#         if raw_df.at[i, "CLOSE"] > raw_df.at[i, "OPEN"]:
#             real_profit = real_profit * -1
#     else:
#         max_profit = raw_df.at[i, "HIGH"] - raw_df.at[i, "OPEN"]
#         if raw_df.at[i, "CLOSE"] < raw_df.at[i, "OPEN"]:
#             real_profit = real_profit * -1

#     if max_profit > target :
#         result = target
#     else:
#         result = real_profit

#     if y_pred[i] == 0:
#         filteredresult = result
#         nonfilteredresult = result
#         filteredprofit = filteredprofit + result
#         nonfilteredprofit = nonfilteredprofit + result
#     else:
#         filteredresult = 0
#         nonfilteredresult = result
#         nonfilteredprofit = nonfilteredprofit + result

#     raw_df.at[i, 'FILTEREDRESULT'] = filteredresult
#     raw_df.at[i, 'NONFILEREDRESULT'] = nonfilteredresult

# print(filteredprofit)
# print(nonfilteredprofit)

#raw_df.to_csv("C:\\Dumps\\predicted.csv")
