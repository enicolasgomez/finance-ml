import pandas_datareader.data as web
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import tensorflow as tf 
import quandl
import os.path

import seaborn as sn
import matplotlib.pyplot as plt

symbols = [{'name':'AUD/CAD','desc':'Australian Dollar/Canadian Dollar','date':'2001-10-22'},
{'name':'AUD/CHF','desc':'Australian Dollar/Swiss Franc','date':'2001-11-29'},
{'name':'AUD/JPY','desc':'Australian Dollar/Japanese Yen','date':'2001-10-22'},
{'name':'AUD/NZD','desc':'Australian Dollar/New Zealand Dollar','date':'2001-11-29'},
{'name':'AUD/USD','desc':'Australian Dollar/US Dollar','date':'2001-10-22'},
{'name':'CAD/CHF','desc':'Canadian Dollar/Swiss Franc','date':'2008-04-10'},
{'name':'CAD/JPY','desc':'Canadian Dollar/Japanese Yen','date':'2001-11-29'},
{'name':'CHF/JPY','desc':'Swiss Franc/Japanese Yen','date':'2001-10-22'},
{'name':'EUR/AUD','desc':'Euro/Australian Dollar','date':'2001-10-22'},
{'name':'EUR/CAD','desc':'Euro/Canadian Dollar','date':'2001-10-22'},
{'name':'EUR/CHF','desc':'Euro/Swiss Franc','date':'2001-10-22'},
{'name':'EUR/GBP','desc':'Euro/British Pound','date':'2001-10-22'},
{'name':'EUR/JPY','desc':'Euro/Japanese Yen','date':'2001-10-22'},
{'name':'EUR/NOK','desc':'Euro/Norwegian Krone','date':'2007-09-24'},
{'name':'EUR/NZD','desc':'Euro/New Zealand Dollar','date':'2001-11-29'},
{'name':'EUR/SEK','desc':'Euro/Swedish Krona','date':'2007-09-24'},
{'name':'EUR/TRY','desc':'Euro/Turkey Lira','date':'2008-04-11'},
{'name':'EUR/USD','desc':'Euro/US Dollar','date':'2001-10-22'},
{'name':'GBP/AUD','desc':'British Pound/Australian Dollar','date':'2001-11-29'},
{'name':'GBP/CAD','desc':'British Pound/Canadian Dollar','date':'2001-11-29'},
{'name':'GBP/CHF','desc':'British Pound/Swiss Franc','date':'2001-10-22'},
{'name':'GBP/JPY','desc':'British Pound/Japanese Yen','date':'2001-10-22'},
{'name':'GBP/NZD','desc':'British Pound/New Zealand Dollar','date':'2001-11-29'},
{'name':'GBP/USD','desc':'British Pound/US Dollar','date':'2001-10-22'},
{'name':'NZD/CAD','desc':'New Zealand Dollar/Canadian Dollar','date':'2008-09-15'},
{'name':'NZD/CHF','desc':'New Zealand Dollar/Swiss Franc','date':'2008-09-15'},
{'name':'NZD/JPY','desc':'New Zealand Dollar/Japanese Yen','date':'2001-11-29'},
{'name':'NZD/USD','desc':'New Zealand Dollar/US Dollar','date':'2001-10-22'},
{'name':'TRY/JPY','desc':'Turkey Lira/Japanese Yen','date':'2010-05-09'},
{'name':'USD/CAD','desc':'US Dollar/Canadian Dollar','date':'2001-10-22'},
{'name':'USD/CHF','desc':'US Dollar/Swiss Franc','date':'2001-10-22'},
{'name':'USD/CNH','desc':'US Dollar/Chinese Yuan','date':'2012-02-18'},
{'name':'USD/HKD','desc':'US Dollar/Hong Kong Dollar','date':'2007-01-22'},
{'name':'USD/JPY','desc':'US Dollar/Japanese Yen','date':'2001-10-22'},
{'name':'USD/MXN','desc':'US Dollar/Mexican Peso','date':'2008-04-11'},
{'name':'USD/NOK','desc':'US Dollar/Norwegian Krone','date':'2006-10-06'},
{'name':'USD/SEK','desc':'US Dollar/Swedish Krona','date':'2006-10-06'},
{'name':'USD/TRY','desc':'US Dollar/Turkey Lira','date':'2008-05-31'},
{'name':'USD/ZAR','desc':'US Dollar/South African Rand','date':'2007-01-22'}]

import unicodedata
import re

class POSITION_TYPE:
  SHORT = -1
  LONG = 1
  NONE = 0

class Position:
  def __init__(self, ask, bid, type, time, symbol):
    self.type = type
    self.openask = ask
    self.openbid = bid
    self.symbol = symbol
    self.closed = False

  def close(self, ask, bid, time):
    self.closeask = ask
    self.closebid = bid
    self.equity = (self.openbid - ask) if self.type == POSITION_TYPE.SHORT else (bid - self.openask)
    self.closed = True
  
  def set_stop_loss(self, value):
    self.stop_loss = value

  def set_take_profit(self, value):
    self.take_profit = value 

  def tick(self, ask, bid, time):
    if ( ( self.type == POSITION_TYPE.SHORT ) and ( ( bid > self.stop_loss) or ( bid < self.take_profit ) ) ) or ( ( self.type == POSITION_TYPE.LONG ) and ( ( ask < self.stop_loss) or ( ask > self.take_profit ) ) ):
      self.close(ask, bid, time)

  def get_equity(self):
    return self.equity

  
def slugify(value, allow_unicode=False):
  """
  Taken from https://github.com/django/django/blob/master/django/utils/text.py
  Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
  dashes to single dashes. Remove characters that aren't alphanumerics,
  underscores, or hyphens. Convert to lowercase. Also strip leading and
  trailing whitespace, dashes, and underscores.
  """
  value = str(value)
  if allow_unicode:
      value = unicodedata.normalize('NFKC', value)
  else:
      value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
  value = re.sub(r'[^\w\s-]', '', value.lower())
  return re.sub(r'[-\s]+', '-', value).strip('-_')

def read_symbol_data(symbol, timeframe, start, end, force):
  ds = pd.DataFrame()
  file_name = 'prices-%s.%s' % ( slugify('%s-%s' % (symbol, timeframe)), 'csv' )
  api = 'FXCM/%s'%(timeframe)
  if os.path.isfile(file_name) and not force:
    ds = pd.read_csv(file_name)
  else:
    quandl.ApiConfig.api_key = "LdbnZe-fWfuHrv3_kRR7"
    ds = quandl.get_table(api, qopts = { 'columns': ['closeask','openask','highask','lowask'] }, date = { 'gte': start, 'lte': end }, symbol=symbol)
    total = len(ds)
    print('%s returned %i records'%(symbol,total))
    if total > 0:
      get_ta_data(ds)
      ds.to_csv(file_name)
  return ds

def read_all_data(timeframe, start, end):
  data = pd.DataFrame()
  for symbol in symbols:
    data = read_symbol_data(symbol['name'], timeframe, start, end, True)
    total = len(data.openask)
    print(total)
    if total > 0:
      data = get_ta_data(data)
      file_name = 'prices-ta-%s.%s' % ( slugify('%s-%s' % (symbol['name'], timeframe)), 'csv' )
      data.to_csv(file_name)

def get_ta_data(dataset):
  open = dataset['openask'].dropna().values
  up, mid, low = talib.BBANDS(open, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
  rsi = talib.RSI(open, timeperiod=14)
  dataset['UpBand'] = up
  dataset['MidBand'] = mid
  dataset['LowBand'] = low
  dataset['RSI'] = rsi
  return dataset 
def get_correlative_pairs(dataset):
  corr_matrix = dataset.corr().abs()
  sn.heatmap(corr_matrix, annot=True)
  plt.show()
  corr_pairs = dataset.corr().unstack().sort_values().drop_duplicates()
  corr_list = [i[0] for i in corr_pairs.items()]
  return corr_list

read_all_data('H1','2019-01-01','2019-12-31')

#read_symbol_data('EUR/USD', 'H1', '', '', True)