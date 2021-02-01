import numpy as np #linear algebra
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

Data=pd.read_csv("C:\\Dumps\\processed.csv")
Data.dropna()
l = len(Data)

cols = ['MFI_CLOSEBID_1',
'SRSI_D2',
'WILLR_4',
# 'MFI_CLOSE_ASK_2',
# 'MFI_CLOSE_ASK_1',
# 'MFI_CLOSE_ASK_4',
'WILLR_5',
'MFI_BID_1',
'MFI_BID_3',
'SRSI_K5',
'SRSI_K1',
'SRSI_D3',
#'MFI_CLOSEBID_4',
'ADX_1',
'ADX_4']

Negatives=[Data['TOFILTER']==0]
Positives=[Data['TOFILTER']==1]

Train_Data=Data[cols]
Target=Data['TOFILTER']

for col in cols:
   Train_Data[col] = Train_Data[col].rank(method='first')
   Train_Data[col] = pd.qcut(Train_Data[col], q=10, labels=[1,2,3,4,5,6,7,8,9,10])


print((len(Positives)/l)*100,"%")


print(Train_Data.sample(frac=0.1).head(n=5))
print(Train_Data.describe())
#Train_Data.drop('TOFILTER',axis=1,inplace=True)

x_train,x_test,y_train,y_test=train_test_split(Train_Data,Target,test_size=0.25,random_state=0)

clf_l=svm.SVC(kernel='linear')
clf_l.fit(x_train,y_train)
print(classification_report(y_test,clf_l.predict(x_test)))

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(x_train, y_train)
print(classification_report(y_test,clf.predict(x_test)))

from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

def Train_Accuracy(Mat):
   Sum=0
   for i in Mat:    
        if(i==1):        
           Sum+=1.0   
   return(Sum/len(Mat)*100)

def Test_Accuracy(Mat):
   Sum=0
   for i in Mat:
        if(i==-1):
           Sum+=1.0
   return(Sum/len(Mat)*100)

clf_AD = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf_AD.fit(Negatives)

clf_AD_L = svm.OneClassSVM(nu=0.1, kernel="linear", gamma=0.1)
clf_AD_L.fit(Negatives)

IFA=IsolationForest()
IFA.fit(Negatives)

train_AD_L=clf_AD_L.predict(Negatives)
test_AD_L=clf_AD_L.predict(Positives)

train_IFA=IFA.predict(Negatives)
test_IFA=IFA.predict(Positives)

train_AD=clf_AD.predict(Negatives)
test_AD=clf_AD.predict(Positives)

print("Training: One CLASS SVM (RBF) : ",(Train_Accuracy(train_AD)),"%")
print("Test: One CLASS SVM (RBF) : ",(Test_Accuracy(test_AD)),"%")

print("Training: Isolation Forest: ",(Train_Accuracy(train_IFA)),"%")
print("Test: Isolation Forest: ",(Test_Accuracy(test_IFA)),"%")

print("Training: One CLASS SVM (Linear) : ",(Train_Accuracy(train_AD_L)),"%")
print("Test: One CLASS SVM (Linear) : ",(Test_Accuracy(test_AD_L)),"%")


# plt.figure(figsize=(20,18))
# Corr=Data[Data.columns].corr()
# sns.heatmap(Corr,annot=True)
# plt.show()
# from imblearn.over_sampling import SMOTE

# W_Data=pd.read_csv("C:\\Dumps\\processed.csv")
# W_Data.dropna(thresh=284315)

# sm = SMOTE(random_state=42)
# X_res, y_res = sm.fit_sample(W_Data, W_Data['TOFILTER'])

# W_Data = W_Data.values

# S_Positives=[]
# S_Negatives=[]

# for i in range(0,len(X_res)):
#     if(y_res[i]==0):
#         S_Negatives.append(X_res[i])
#     else:
#         S_Positives.append(X_res[i])

# IFA=IsolationForest()
# IFA.fit(S_Negatives)
# S_train_IFA=IFA.predict(S_Negatives)
# S_test_IFA=IFA.predict(S_Positives)

# print("Training: Isolation Forest: ",(Train_Accuracy(S_train_IFA)),"%")
# print("Test: Isolation Forest: ",(Test_Accuracy(S_test_IFA)),"%")

# Outcome=Data['TOFILTER']
# X_res, y_res = sm.fit_sample(Data,Outcome)
# x_train_E,x_test_E,y_train_E,y_test_E=train_test_split(X_res,y_res,test_size=0.5,random_state=0)
# x_train_O,x_test_O,y_train_O,y_test_O=train_test_split(Data,Outcome,test_size=0.5,random_state=0)

# clf = RandomForestClassifier(max_depth=2, random_state=0)
# clf.fit(x_train_E, y_train_E)
# print(classification_report(y_test_O,clf.predict(x_test_O)))

