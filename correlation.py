import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.linear_model import LassoCV
from matplotlib import pyplot as plt

def get_correlative_pairs(dataset):
  corr_matrix = dataset.corr().abs()
  sn.heatmap(corr_matrix, annot=True)
  plt.show()
  corr_pairs = dataset.corr().unstack().sort_values().drop_duplicates()
  corr_list = [i[0] for i in corr_pairs.items()]
  return corr_list

def principal_components_regressor(X, y):
  reg = LassoCV()
  reg.fit(X, y)
  print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
  print("Best score using built-in LassoCV: %f" %reg.score(X,y))
  coef = pd.Series(reg.coef_, index = X.columns)

  print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

  imp_coef = coef.sort_values()
  imp_coef = imp_coef[imp_coef != 0]
  import matplotlib
  matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
  imp_coef.plot(kind = "barh")
  plt.title("Feature importance using Lasso Model")
  plt.show()

  return imp_coef

def principal_components_pca(X, y):
  from sklearn.model_selection import train_test_split

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  from sklearn.preprocessing import StandardScaler

  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  from sklearn.decomposition import PCA

  pca = PCA()
  X_train = pca.fit_transform(X_train)
  X_test = pca.transform(X_test)

  explained_variance = pca.explained_variance_ratio_

  return explained_variance