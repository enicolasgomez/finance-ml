import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
# load dataset
dataframe = pd.read_csv(os.path.join(sys.path[0], "candle_data_out_normalize_2.csv"), delim_whitespace=False, header=0)
dataframe = dataframe.drop(columns=["CLOSINGPIPS", "INIT","CLOSETIME"], axis=1)

y = np_utils.to_categorical(dataframe["RESULT"], num_classes=2)

X_train, X_test, y_train, y_test = train_test_split(dataframe, y, test_size=0.3, shuffle=True)

colnames = dataframe.columns

import numpy as np
import matplotlib.pyplot as plt 
# from matplotlib.pyplot import matplotlib
fig,axes =plt.subplots(10,3, figsize=(12, 9)) # 3 columns each containing 10 figures, total 30 features
short_candles=dataframe[dataframe["RESULT"]==0] # define malignant
long_candles=dataframe[dataframe["RESULT"]==1] # define benign
ax=axes.ravel()# flat axes with numpy ravel
for i in range(25):
  _,bins=np.histogram(dataframe[colnames[i]],bins=40)
  ax[i].hist(short_candles[colnames[i]],bins=bins,color='r',alpha=.5)# red color for malignant class
  ax[i].hist(long_candles[colnames[i]],bins=bins,color='g',alpha=0.3)# alpha is           for transparency in the overlapped region 
  ax[i].set_title(colnames[i],fontsize=9)
  ax[i].axes.get_xaxis().set_visible(False) # the x-axis co-ordinates are not so useful, as we just want to look how well separated the histograms are
  ax[i].set_yticks(())
ax[0].legend(['short','long'],loc='best',fontsize=8)
plt.tight_layout()# let's make good plots
plt.show()