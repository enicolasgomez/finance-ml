from symbols import symbols
from download import read_all_data, get_file_name
from correlation import get_correlative_pairs, principal_components_regressor
from tadata import get_ta_features, fill_column

from sklearn.linear_model import LassoCV
from matplotlib import pyplot as plt

import pandas as pd 
import numpy as np 

timeframe = 'H1'

def read_file(symbol, timeframe):
  file_name = get_file_name(symbol, timeframe)
  ds = pd.read_csv(file_name)
  return ds

def correlative_pairs(timeframe):
  cor_pairs = pd.DataFrame()
  for symbol in symbols:
    ds = read_file(symbol['name'], timeframe)
    column = ds['openask'].values
    # missing = [0]*(7000-len(column))
    # column = np.hstack([column,missing])
    #cor_pairs[symbol['name']] = column
    fill_column(cor_pairs, column, symbol['name'], 7000)
  cp = get_correlative_pairs(cor_pairs)
  print(cp)


def main():
  read_all_data(symbols, timeframe,'2019-01-01','2019-12-31', False)
  ds = read_file('EUR/USD', timeframe)
  features = get_ta_features(ds, 25)
  #remove first N as they contain NaN
  features = features[50:]
  target = 'AROONOSC'
  y = features[target]
  del features[target]
  X = features
  p = principal_components_regressor(X, y)
  print(p)

if __name__ == "__main__":
  main()


