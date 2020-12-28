#Libraries
import numpy as np
import pandas as pd


#import data
dataset = pd.read_csv("DATASET/Iris.csv")
X = dataset.iloc[:,1:5].values
Y = dataset.iloc[:,5].values


#Standarisation

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

class NaiveBayes:
    def fit(self,X,Y):
        n_samples,n_features = X.shape
        self.classes = np.unique(Y)
        n_classes= len(self.classes)
        #initialize mean,var,priors
        self._mean = np.zeros((n_classes,n_features),dtype = np.float64)
        self._var = np.zeros((n_classes,n_features),dtype = np.float64)
        self.prior = np.zeros((n_classes),dtype = np.float64)
        
        for c in self.classes:
            X_c = X[c==Y]    #All the sample of positive class, all the samples of negative class
            #Calculate mean, var, and priors
            self._mean[c,:] = X_c.mean(axis = 0)
            self._var[c,:] = X_c.var(axis = 0)
            self.prior[c] = X_c.shape[0]/float(n_samples)
            
    def predict(self,X):
        #Big matrix of x
        y_pred = [self._predict(x) for x in X]
        return y_pred
    
    def _predict(self,x):
        posteriors=[]
        for idx,c in enumerate(self.classes):
            prior = np.log(self.prior[idx])
            condition = np.sum(np.log(self._pdf(idx,x)))
            posterior = prior+condition
            posteriors.append(posterior)
        print(posteriors)
        return self.classes[np.argmax(posteriors)]
            
    def _pdf(self,idx,x):
        mean = self._mean[idx]
        var = self._var[idx]
        num = np.exp(-(x-mean)**2/(2*var))
        denom = np.sqrt(2*np.pi*var)
        return num/denom+0.00001
            
#Call
nb = NaiveBayes()
nb.fit(X,Y)
y_pred = nb.predict(X)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y, y_pred)