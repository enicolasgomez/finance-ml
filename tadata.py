import pandas as pd
import talib
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split  
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def count_pattern(pattern, lastn):
    total_list = [0] * len(pattern)
    for i in range(lastn, len(pattern)):
        total = 0
        for j in range(1, lastn):
            if pattern[(i-j)] == 100 :
                total = total + 1
        total_list[i] = total
    return total_list

def fill_column(data, column, column_name, max):
    #missing = [0]*(max-len(column))
    #column = np.hstack([column,missing])
    data[column_name] = column
  
def add_patterns(data, open, high, low, close, lastn):
    CDLHARAMI       = count_pattern(talib.CDLHARAMI(open,high,low,close), lastn)
    fill_column(data, CDLHARAMI, 'CDLHARAMI', 7000)

    CDLHARAMICROSS  = count_pattern(talib.CDLHARAMICROSS(open,high,low,close), lastn)
    fill_column(data, CDLHARAMICROSS, 'CDLHARAMICROSS', 7000)

    CDLHIGHWAVE     = count_pattern(talib.CDLHIGHWAVE(open,high,low,close), lastn)
    fill_column(data, CDLHIGHWAVE, 'CDLHIGHWAVE', 7000)

    CDLHIKKAKE      = count_pattern(talib.CDLHIKKAKE(open,high,low,close), lastn)
    fill_column(data, CDLHIKKAKE, 'CDLHIKKAKE', 7000)

    CDLHIKKAKEMOD   = count_pattern(talib.CDLHIKKAKEMOD(open,high,low,close), lastn)
    fill_column(data, CDLHIKKAKEMOD, 'CDLHIKKAKEMOD', 7000)

    CDLHOMINGPIGEON = count_pattern(talib.CDLHOMINGPIGEON(open,high,low,close), lastn)
    fill_column(data, CDLHOMINGPIGEON, 'CDLHOMINGPIGEON', 7000)

    CDLIDENTICAL3CROWS = count_pattern(talib.CDLIDENTICAL3CROWS(open,high,low,close), lastn)
    fill_column(data, CDLIDENTICAL3CROWS, 'CDLIDENTICAL3CROWS', 7000)

    CDLINVERTEDHAMMER   = count_pattern(talib.CDLINVERTEDHAMMER(open,high,low,close), lastn)
    fill_column(data, CDLINVERTEDHAMMER, 'CDLINVERTEDHAMMER', 7000)

    CDLKICKING          = count_pattern(talib.CDLKICKING(open,high,low,close), lastn)
    fill_column(data, CDLKICKING, 'CDLKICKING', 7000)

    CDLKICKINGBYLENGTH  = count_pattern(talib.CDLKICKINGBYLENGTH(open,high,low,close), lastn)
    fill_column(data, CDLKICKINGBYLENGTH, 'CDLKICKINGBYLENGTH', 7000)

    CDLLADDERBOTTOM     = count_pattern(talib.CDLLADDERBOTTOM(open,high,low,close), lastn)
    fill_column(data, CDLLADDERBOTTOM, 'CDLLADDERBOTTOM', 7000)

def add_indicators(data, open, high, low, close):
    ADX = talib.ADX(high, low, close, timeperiod=14)
    fill_column(data, ADX, 'ADX', 7000)

    ADXR = talib.ADXR(high, low, close, timeperiod=14)
    fill_column(data, ADXR, 'ADXR', 7000)
    
    APO = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
    fill_column(data, APO, 'APO', 7000)

    ATR = talib.ATR(high, low, close, timeperiod=14)
    fill_column(data, ATR, 'ATR', 7000)

    AROONOSC = talib.AROONOSC(high, low, timeperiod=14)
    fill_column(data, AROONOSC, 'AROONOSC', 7000)

    CCI = talib.CCI(high, low, close, timeperiod=14)
    fill_column(data, CCI, 'CCI', 7000)

    CMO = talib.CMO(close, timeperiod=14)
    fill_column(data, CMO, 'CMO', 7000)

    DX = talib.DX(high, low, close, timeperiod=14)
    fill_column(data, DX, 'DX', 7000)

    MINUS_DI = talib.MINUS_DI(high, low, close, timeperiod=14)
    fill_column(data, MINUS_DI, 'MINUS_DI', 7000)

    MINUS_DM = talib.MINUS_DM(high, low, timeperiod=14)
    fill_column(data, MINUS_DM, 'MINUS_DM', 7000)

    MOM = talib.MOM(close, timeperiod=10)
    fill_column(data, MOM, 'MOM', 7000)

    PLUS_DI = talib.PLUS_DI(high, low, close, timeperiod=14)
    fill_column(data, PLUS_DI, 'PLUS_DI', 7000)

    PLUS_DM = talib.PLUS_DM(high, low, timeperiod=14)
    fill_column(data, PLUS_DM, 'PLUS_DM', 7000)

    PPO = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
    fill_column(data, PPO, 'PPO', 7000)

    ROC = talib.ROC(close, timeperiod=10)
    fill_column(data, ROC, 'ROC', 7000)

    ROCP = talib.ROCP(close, timeperiod=10)
    fill_column(data, ROCP, 'ROCP', 7000)

    ROCR100 = talib.ROCR100(close, timeperiod=10)
    fill_column(data, ROCR100, 'ROCR100', 7000)

    RSI = talib.RSI(close, timeperiod=14)
    fill_column(data, RSI, 'RSI', 7000)

    ULTOSC = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    fill_column(data, ULTOSC, 'ULTOSC', 7000)

    WILLR = talib.WILLR(high, low, close, timeperiod=14)
    fill_column(data, WILLR, 'WILLR', 7000)

def get_ta_features(data, period):
    open = np.array(data['openask'])
    high = np.array(data['highask'])
    low = np.array(data['lowask'])
    close = np.array(data['closeask'])

    add_indicators(data, open, high, low, close)
    add_patterns(data, open,high, low, close, period)

    return data