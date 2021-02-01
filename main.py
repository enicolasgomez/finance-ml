from symbols import symbols
from download import read_all_data, get_file_name
from correlation import get_correlative_pairs
import pandas as pd 
import numpy as np 

def main():
  timeframe = 'H1'
  read_all_data(symbols, timeframe,'2019-01-01','2019-12-31', False)
  cor_pairs = pd.DataFrame()
  for symbol in symbols:
    file_name = get_file_name(symbol['name'], timeframe)
    ds = pd.read_csv(file_name)
    column = ds['openask'].values
    missing = [0]*(7000-len(column))
    column = np.hstack([column,missing])
    cor_pairs[symbol['name']] = column
  cp = get_correlative_pairs(cor_pairs)
  print(cp)
if __name__ == "__main__":
  main()


