import numpy 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

data = pd.read_csv("C:\Dumps\processed_clean2.csv", header=0)
data = shuffle(data)

seed = 5
numpy.random.seed(seed)

prediction_var = ['OPENASKVOL'
                  ,'OPENBIDVOL'
                  ,'MACD_1'
                  ,'SIGNAL_1'
                  ,'HIST_1'
                  ,'ATR_1'
                  ,'BB_LOW_1'
                  ,'BB_MID_1'
                  ,'BB_UP_1'
                  ,'ADX_1'
                  ,'RSI_1'
                  ,'SRSI_K1'
                  ,'SRSI_D1'
                  ,'MA_1'
                  ,'WILLR_1'
                  ,'AVGPRICE_1'
                  ,'MEDPRICE_1'
                  ,'TYPPRICE_1'
                  ,'WCLPRICE_1']

X = data[prediction_var].values
Y = data.TOFILTER.values

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(19, input_dim=19, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))