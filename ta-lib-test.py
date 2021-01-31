import pandas as pd
import talib
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split  
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

data = pd.read_csv('prices-eurusd-h1.csv')

def count_pattern(pattern, lastn):
    total_list = [0] * len(pattern)
    for i in range(lastn, len(pattern)):
        total = 0
        for j in range(1, lastn):
            if pattern[(i-j)] == 100 :
                total = total + 1
        total_list[i] = total
    return total_list
  
def get_patterns(open, high, low, close, lastn):
    CDLHARAMI       = count_pattern(talib.CDLHARAMI(open,high,low,close), lastn)
    CDLHARAMICROSS  = count_pattern(talib.CDLHARAMICROSS(open,high,low,close), lastn)
    CDLHIGHWAVE     = count_pattern(talib.CDLHIGHWAVE(open,high,low,close), lastn)
    CDLHIKKAKE      = count_pattern(talib.CDLHIKKAKE(open,high,low,close), lastn)
    CDLHIKKAKEMOD   = count_pattern(talib.CDLHIKKAKEMOD(open,high,low,close), lastn)
    CDLHOMINGPIGEON = count_pattern(talib.CDLHOMINGPIGEON(open,high,low,close), lastn)
    CDLIDENTICAL3CROWS = count_pattern(talib.CDLIDENTICAL3CROWS(open,high,low,close), lastn)
    CDLINVERTEDHAMMER   = count_pattern(talib.CDLINVERTEDHAMMER(open,high,low,close), lastn)
    CDLKICKING          = count_pattern(talib.CDLKICKING(open,high,low,close), lastn)
    CDLKICKINGBYLENGTH  = count_pattern(talib.CDLKICKINGBYLENGTH(open,high,low,close), lastn)
    CDLLADDERBOTTOM     = count_pattern(talib.CDLLADDERBOTTOM(open,high,low,close), lastn)
    patterns            = np.column_stack((CDLHARAMI,CDLHARAMICROSS,CDLHIGHWAVE,CDLHIKKAKE,CDLHIKKAKE,CDLHIKKAKEMOD,CDLHOMINGPIGEON,CDLIDENTICAL3CROWS,CDLINVERTEDHAMMER,CDLKICKING,CDLKICKINGBYLENGTH,CDLLADDERBOTTOM))
    return patterns

def get_indicators(open, high, low, close):
    ADX = talib.ADX(high, low, close, timeperiod=14)
    ADXR = talib.ADXR(high, low, close, timeperiod=14)
    APO = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
    AROONOSC = talib.AROONOSC(high, low, timeperiod=14)
    CCI = talib.CCI(high, low, close, timeperiod=14)
    CMO = talib.CMO(close, timeperiod=14)
    DX = talib.DX(high, low, close, timeperiod=14)
    MINUS_DI = talib.MINUS_DI(high, low, close, timeperiod=14)
    MINUS_DM = talib.MINUS_DM(high, low, timeperiod=14)
    MOM = talib.MOM(close, timeperiod=10)
    PLUS_DI = talib.PLUS_DI(high, low, close, timeperiod=14)
    PLUS_DM = talib.PLUS_DM(high, low, timeperiod=14)
    PPO = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
    ROC = talib.ROC(close, timeperiod=10)
    ROCP = talib.ROCP(close, timeperiod=10)
    ROCR100 = talib.ROCR100(close, timeperiod=10)
    RSI = talib.RSI(close, timeperiod=14)
    ULTOSC = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    WILLR = talib.WILLR(high, low, close, timeperiod=14)
    indicators = np.column_stack((ADX,ADXR,APO,AROONOSC,CCI,CMO,DX,MINUS_DI,ROCR100,ROC,MINUS_DM,MOM,PLUS_DI,PLUS_DM,PPO,ROCP,WILLR,ULTOSC,RSI))
    return indicators

open = np.array(data['openask'])
high = np.array(data['highask'])
low = np.array(data['lowask'])
close = np.array(data['closeask'])

indicators = get_indicators(open, high, low, close)
patterns = get_patterns(open,high, low, close, 25)
features = np.column_stack((patterns, indicators))
features = features[50:]
close = close[50:]
#data = get_patterns(data)

label = []
for i in range(len(close)-1):
    if (close[i+1]-close[i])/close[i] > 0.0005:
        label.append(1)
    else:
        label.append(0)
label.append(0)
label = np.array(label).reshape(len(features),1)
#matrix = np.column_stack((matrix,label))

print("Positive %", round(len(label[label == 1])/len(label)*100, 2))
matrix = np.nan_to_num(features)


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.20)  

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix  

# svclassifier = SVC(kernel='linear')  
# svclassifier.fit(X_train, y_train) 
# y_pred = svclassifier.predict(X_test)  

# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# print(classification_report(y_test,y_pred)) 
# print(confusion_matrix(y_test, y_pred))

from sklearn.naive_bayes import GaussianNB
 
#Create a Gaussian Classifier
gnb = GaussianNB()
 
#Train the model using the training sets
gnb.fit(X_train, y_train)
 
#Predict the response for test dataset
y_pred = gnb.predict(X_test)
 
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#print(classification_report(y_test,y_pred)) 
print(confusion_matrix(y_test, y_pred))