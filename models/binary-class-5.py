import pandas as pd
from sklearn.preprocessing import StandardScaler
# print version number
RawData=pd.read_csv("C:\\Dumps\\processed.csv")

#features = ['OPENASKVOL','OPENBIDVOL','MACD_1','SIGNAL_1','HIST_1','ATR_1','BB_LOW_1','BB_MID_1','BB_UP_1','ADX_1','RSI_1','SRSI_K1','SRSI_D1','MA_1','WILLR_1','AD_1','ADOSC_1','OBV_1','MFI_1']
#features = ['RSI_1','WILLR_1','ADX_1']
X = RawData.drop(['TOFILTERAVG','TOFILTER2','TOFILTER1','TOFILTER','INIT','OPEN','CLOSE','HIGH','LOW','OPENASKVOL','OPENBIDVOL','CLOSEASKVOL','CLOSEBIDVOL','CLOSETIME','CLOSINGPIPS','TYPE','PREDICTED'], 1)
#X = X[features]

y = RawData['TOFILTER1']

from sklearn.linear_model import LassoCV
from matplotlib import pyplot as plt

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

from sklearn.decomposition import PCA

pca = PCA(n_components=3)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)