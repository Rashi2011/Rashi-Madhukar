#importing libraries
import pandas as pd

#load the dataset
dataset=pd.read_csv("Dataset/train_titanic.csv")

#Feature Engineering
#Deleting Cabin info as lot of information of cabin is not provided
#ticket and PassengerId do not contribute to the survival of the person
del dataset['Cabin']
del dataset['Ticket']
del dataset['PassengerId']

#Divide the data into Dependent and Independent variables(Y and X resp)
X=dataset.iloc[:,1:8].values
Y=dataset.iloc[:,0].values

#ratio of positive and negative class in Y
pos=0;neg=0
for i in Y:
    if i==1:
        pos+=1
    else:
        neg+=1
print("No of people that survived: ",pos)
print("No of people not able to survive: ",neg)

'''Feature Engineering on remaining features'''
#Two string Inputs in sex column and Embarked column ---use Label Encoding for these

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
X[:,1]=lb.fit_transform(X[:,1])  #column sex

#impute the Embarkek ,pclass and sex columns with most frequent data
from sklearn.impute import SimpleImputer
sc=SimpleImputer( strategy='most_frequent')
X[:,0:2]=sc.fit_transform(X[:,0:2])

#As embarked column contain nan values therfore we have to first impute values in missing places
#Method we use here is most frequent data ie mode as Embarked can be replaced with common port 
#that people come from and most frequent sex that visited
#Embarked(1. impute nan from Embarked columns then use Label Encoding then )
f=X[:,6].reshape(-1,1) #1d data to 2d(becoz for Label encoding 2d data is required)
f=sc.fit_transform(f)
lb1=LabelEncoder()
f=lb1.fit_transform(f)
X[:,6]=f.reshape(1,-1)#2d to 1

#Imputing nan data in age,sibsep ,parch by mean strategy
sc1 =SimpleImputer( strategy='mean')
X[:,2:6]=sc1.fit_transform(X[:,2:6])

#oneHotEncoder
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features=[6])
X=ohe.fit_transform(X).toarray()
#remove the Dummy Variable
X=X[:,1:]
#Merge these two one Hot Encoder with X by replacing Embarked

'''Applying Algorithm to train the training set'''#
#Approach used here is KNN - (1.)as Dataset is small,(2.) We can find similiarity using Euclidean Distance"
#KNN ALGORITHM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
l=[1,3,5,7,9,11,13,15,17,19]
for i in l:
    knn=KNeighborsClassifier(n_neighbors=i,metric='minkowski',p=2)
    knn.fit(X,Y)
    #Applying Cros Validation to know what value of k works best
    accuracies=cross_val_score(estimator=knn,X=X,y=Y,cv=10)
    print("for k = ",i,"accuracy is ",accuracies.mean())

