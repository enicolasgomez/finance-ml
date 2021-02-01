import numpy as np #linear algebra
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import datetime
from sklearn.ensemble import RandomForestClassifier

Data=pd.read_csv("C:\\Dumps\\24.csv")


DataAll=pd.read_csv("C:\\Dumps\\24all.csv")

#Data.dropna()
#Data = shuffle(Data)
l = len(Data)

cols = ['MACD_1','SIGNAL_1','HIST_1','ATR_1','BB_LOW_1','BB_MID_1','BB_UP_1','ADX_1','RSI_1','SRSI_K1','SRSI_D1','MA_1','WILLR_1','AD_1','ADOSC_1','OBV_1','MFI_1']
DataAll = DataAll[cols]

#cols = ['ATR_1','ADX_1','WILLR_1','AD_1','ADOSC_1','OBV_1','MFI_1']

# Data.INIT = pd.to_datetime(Data.INIT)
# bins = 100

# for col in cols:
#    Data[col] = Data[col].rank(method='first')
#    Data[col] = pd.qcut(Data[col], q=bins, labels=list(range(1,bins+1))).cat.codes

X=Data[cols]
y=Data['TOFILTER']

print(X.sample(frac=0.1).head(n=5))
print(X.describe())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=123)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler().fit(DataAll)

# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# X_val = scaler.transform(X_val)

#fit xgboost on an imbalanced classification dataset
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

clf = RandomForestClassifier(max_depth=5, random_state=0)

from sklearn.metrics import confusion_matrix
clf.fit(X_train, y_train)

ypred1 = clf.predict(X_test)
cm = confusion_matrix(ypred1, y_test) 
print(cm)

# print(X.sample(frac=0.1).head(n=5))
# print(X.describe())

raw_df = pd.read_csv("C:\\Dumps\\24fwd.csv")
x_raw_df = raw_df[cols]

#x_raw_df = scaler.transform(x_raw_df)

ypred_data = clf.predict(x_raw_df)
cm3 = confusion_matrix(ypred_data, raw_df.TOFILTER) 
print(cm3)

#raw_df = raw_df[cols]

# target = 10 / 10000
# filteredprofit = 0
# nonfilteredprofit = 0

# totalfiltered = 0
# totalfilteredprofit = 0

# totalnonfiltered = 0
# totalnonfilteredprofit = 0

# import random

# for i in range(0, len(raw_df)):    
#     is_sell = bool(random.getrandbits(1)) 

#     real_profit = abs(raw_df.CLOSE.values[i] - raw_df.OPEN.values[i])
#     if is_sell:
#         max_profit = raw_df.OPEN.values[i] - raw_df.LOW.values[i]
#         if raw_df.CLOSE.values[i] > raw_df.OPEN.values[i]:
#             real_profit = real_profit * -1
#     else:
#         max_profit = raw_df.HIGH.values[i] - raw_df.OPEN.values[i]
#         if raw_df.CLOSE.values[i] < raw_df.OPEN.values[i]:
#             real_profit = real_profit * -1

#     if max_profit > target :
#         result = target
#     else:
#         result = real_profit

#     if ypred_data[i] == 0:
#         filteredresult = result
#         nonfilteredresult = result
#         filteredprofit = filteredprofit + result
#         nonfilteredprofit = nonfilteredprofit + result
#         totalfiltered = totalfiltered + 1
#         if result > 0 :
#             totalfilteredprofit = totalfilteredprofit + 1
#     else:
#         filteredresult = 0
#         nonfilteredresult = result
#         nonfilteredprofit = nonfilteredprofit + result
#         totalnonfiltered = totalnonfiltered + 1
#         if result > 0:
#             totalnonfilteredprofit = totalnonfilteredprofit + 1

#     raw_df.at[i, 'FILTEREDRESULT'] = filteredresult
#     raw_df.at[i, 'NONFILEREDRESULT'] = nonfilteredresult

# print(filteredprofit)
# print(totalfilteredprofit/totalfiltered)
# print(nonfilteredprofit)
# print(totalnonfilteredprofit/totalnonfiltered)

# raw_df.to_csv("C:\\Dumps\\predicted.csv")

