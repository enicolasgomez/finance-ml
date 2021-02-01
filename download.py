import pandas_datareader.data as web
import pandas as pd
import numpy as np
import quandl
import os.path
import unicodedata
import re

def get_file_name(symbol, timeframe):
  return './data/prices-%s.%s' % ( slugify('%s-%s' % (symbol, timeframe)), 'csv' )

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
  file_name = get_file_name(symbol, timeframe)
  api = 'FXCM/%s'%(timeframe)
  if os.path.isfile(file_name) and not force:
    ds = pd.read_csv(file_name)
  else:
    quandl.ApiConfig.api_key = "LdbnZe-fWfuHrv3_kRR7"
    ds = quandl.get_table(api, qopts = { 'columns': ['closeask','openask','highask','lowask'] }, date = { 'gte': start, 'lte': end }, symbol=symbol)
    total = len(ds)
    print('%s returned %i records'%(symbol,total))

  return ds

def read_all_data(symbols, timeframe, start, end, force):
  data = pd.DataFrame()
  for symbol in symbols:
    data = read_symbol_data(symbol['name'], timeframe, start, end, force)
    total = len(data.openask)
    print(total)
    if total > 0:
      #data = get_ta_data(data)
      file_name = get_file_name(symbol['name'], timeframe)
      data.to_csv(file_name)