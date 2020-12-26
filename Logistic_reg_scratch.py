#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import data
dataset = pd.read_csv("DATASET/Social_Network_Ads.csv")
dataset = dataset.iloc[:,1:]
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values


#Gender columns
l = []
for i in X[:,0]:
    if i =='Male':
        l.append(1)
    else:
        l.append(0)
X[:,0] = l
        
#Standarisation
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sign_f(y_pred):
    y=[]
    for i in y_pred:
        if i <0.5:
            y.append(0)
        else:
            y.append(1)
    return y
            
#Logistic Regression
class LogisticRegression:
    def __init__(self,lr = 0.1,itr = 1000):
        self.lr = lr
        self.itr = itr
        self.weight = None
        self.bias = None
        
    def fit(self, X,Y):
        n_samples,n_features = X.shape
        self.weight = np.zeros((n_features))
        self.bias = 0
        linear = np.dot(X, self.weight) + self.bias
        y_pred = sigmoid(linear)
        dw = -2/n_samples*np.sum(np.dot(X.T,(Y-y_pred)))
        db = -2/n_samples*(np.sum(Y-y_pred))
        
        self.weight-=self.lr*dw
        self.bias-= self.lr*db
        
    def predict(self,X):
        y_pred = sigmoid(np.dot(X,self.weight) + self.bias)
        return y_pred
    
#Finding accuracy
def accuracy(Y,y):
    i=0;correct_classified = 0
    n=len(y)
    while i<len(y):
        if Y[i]==y[i]:
            correct_classified+=1
        i+=1
    return correct_classified/n 
    
regressor = LogisticRegression()
regressor.fit(X,Y)
y_pred = regressor.predict(X)

y = sign_f(y_pred)

#Accuracy
print(accuracy(Y,y))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y, y)







