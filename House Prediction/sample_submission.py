import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import datasset
train = pd.read_csv("DATASET/train_ss.csv")
test = pd.read_csv("DATASET/test_ss.csv")

X_train = train.iloc[:,1:80]
Y_train = train.iloc[:,80]
X_test = test.iloc[:,1:80]

#Deleting Columns
#X_train = X_train.drop(X_train.columns[[4,5,8,10]], axis=1)
X_train = X_train.drop(['Street','Alley','Utilities','LandSlope','RoofMatl','PoolQC','Fence','MiscFeature'],axis = 1)
X_test = X_test.drop(['Street','Alley','Utilities','LandSlope','RoofMatl','PoolQC','Fence','MiscFeature'],axis = 1)

#Label Encoder
'''
from sklearn.preprocessing import LabelEncoder
X_train1 = X_train.columns([1,4])
X_train.apply(LabelEncoder().fit_transform)
'''
X_train2 = X_train.iloc[:,1]
X_train2.add(X_train.iloc[:,4:12])
X_train2.add(X_train.iloc[:,16:20])
X_train2.append(X_train.iloc[:,21:28])
X_train2.append(X_train.iloc[:,29])
X_train2.append(X_train.iloc[:,34:37])
X_train2.append(X_train.iloc[:,46:48])