import numpy as np
import pandas as pd 
#import matplotlib as plt

#importing the dataset
Train = pd.read_csv("DATASET/train.csv")
Test = pd.read_csv("DATASET/test.csv")

#Deleting the unnecessary columns 
X_train = Train.iloc[:,2:13]
Y_train = Train.iloc[:,1]
X_test = Test.iloc[:,1:13]
#Deleting unneccesary columns from X_train
del X_train['Name']
del X_train['Cabin']
del X_train['Ticket']
#fillna
X_train['Embarked'] = X_train['Embarked'].fillna(0)

#convert dataframe to array object
X_train = X_train.values
Y_train = Y_train.values


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:, 2:3])
X_train[:, 2:3] = imputer.transform(X_train[:, 2:3])





#encoding categorical data(Sex category)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:,1] = labelencoder_X.fit_transform(X_train[:,1])
#encoding Categorical Data (Embarked Category)
X_train[:,6] = X_train[:,6].astype(str)
labelencoder_X = LabelEncoder()
X_train[:, 6] = labelencoder_X.fit_transform(X_train[:,6])
onehotencoder = OneHotEncoder(categorical_features=[6])
onehotencoder.fit(X_train)
X_train = onehotencoder.transform(X_train).toarray()
X_train = X_train[:,1:10]

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train= sc.fit_transform(X_train)


#deleting columns from the dataframe
del X_test['Cabin']
del X_test['Name']
del X_test['Ticket']
X_test['Fare'] = X_test['Fare'].fillna(0)


#convert dataframe to float object
X_test = X_test.values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_test[:, 2:3])
X_test[:, 2:3] = imputer.transform(X_test[:, 2:3])



#using LabelEncoder and oneHotencoder
#encoding categorical data(Sex category)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X_test[:,1] = labelencoder_X.fit_transform(X_test[:,1])
#encoding Categorical Data (Embarked Category)
X_test[:,6] = X_test[:,6].astype(str)
labelencoder_X = LabelEncoder()
X_test[:,6] = labelencoder_X.fit_transform(X_test[:,6])
onehotencoder = OneHotEncoder(categorical_features=[6])
onehotencoder.fit(X_test)
X_test = onehotencoder.transform(X_test).toarray()

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_test= sc.fit_transform(X_test)


#fitting the random forest classifier on the Xtest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10,criterion = 'entropy',random_state = 0)
classifier.fit(X_train,Y_train)


#Predicting
y_pred = classifier.predict(X_test)
