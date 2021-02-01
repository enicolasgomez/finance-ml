import pandas as pd
import sklearn as sklearn
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json
# load json and create model
json_file = open('C:\\Dumps\\model\\model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("C:\\Dumps\\model\\model.h5")
print("Loaded model from disk")
#simulation
import random 
df = pd.read_csv("C:\\Dumps\\processed.csv", header=0)

cols = ['WILLR_1','AD_1','ADOSC_1','OBV_1','MFI_1','AD_BID_1','ADOSC_BID_1','OBV_BID_1','MFI_BID_1']

X_fwd = df[cols].values
y_fwd = df.TOFILTER.values

#from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler().fit(X_fwd)
# X_fwd = scaler.transform(X_fwd)

y_pred = model.predict(X_fwd)

# encoder = LabelEncoder()
# encoder.fit(y_pred)
# y_pred = encoder.transform(y_pred)


y_pred = np.where(y_pred < ( y_pred.max() * 0.05), 1, 0)

#simulation
import random 

raw_df = pd.read_csv("C:\\Dumps\\processed.csv", header=0)
raw_df['PREDICTED'] = y_pred

cols = ['INIT','OPEN','CLOSE','HIGH','LOW']

raw_fd = raw_df[cols]

target = 3 / 10000

for i in range(0, len(raw_df)):    
    is_sell = bool(random.getrandbits(1)) 

    real_profit = abs(raw_df.at[i, "CLOSE"] - raw_df.at[i, "OPEN"])
    if is_sell:
        max_profit = raw_df.at[i, "OPEN"] - raw_df.at[i, "LOW"]
        if raw_df.at[i, "CLOSE"] > raw_df.at[i, "OPEN"]:
            real_profit = real_profit * -1
    else:
        max_profit = raw_df.at[i, "HIGH"] - raw_df.at[i, "OPEN"]
        if raw_df.at[i, "CLOSE"] < raw_df.at[i, "OPEN"]:
            real_profit = real_profit * -1

    if max_profit > target :
        result = target
    else:
        result = real_profit

    if y_pred[i] == 0:
        filteredresult = result
        nonfilteredresult = result
    else:
        filteredresult = 0
        nonfilteredresult = result

    raw_df.at[i, 'FILTEREDRESULT'] = filteredresult
    raw_df.at[i, 'NONFILEREDRESULT'] = nonfilteredresult

raw_df.to_csv("C:\\Dumps\\predicted.csv")
