# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import data
dataset=pd.read_csv("DATASET/Salary_Data.csv")
X = dataset.iloc[:,0].values
Y = dataset.iloc[:,1].values
X = X.reshape(-1,1)

#To know the Distribution of dataset plot it
'''
plt.scatter(X,Y)
plt.xlabel("Years of Experience")
plt.ylabel("Experience")
'''
#Functions
class LinearRegression:
    def __init__(self, lr=0.01, itr= 1000):
        self.lr=lr
        self.itr=itr
        self.weight = None
        self.bias = None
        
        
    def fit(self, X,Y):
        
        n_samples,n_feature=X.shape
        self.weight = np.zeros((n_feature))
        #print(self.weight.shape)
        #print(self.weight)
        self.bias = 0
        for i in range(self.itr):
            y_pred = np.dot(X,self.weight) + self.bias
            #print("y_pred",y_pred)
            dw = -(2/n_samples)*(np.sum(np.dot(X.T,(Y - y_pred))))
            db = -(2/n_samples) *(np.sum(Y-y_pred))
            loss = (1/n_samples)*(np.sum(np.abs(Y-y_pred)))
            if i%100 == 0:
                print("Loss after every 100 iteration",loss)
            self.weight -= self.lr *(dw) 
            self.bias -= self.lr *(db)

    def predict (self,X):
        y_pred = np.dot(X,self.weight)+ self.bias
        return y_pred

        
#Calculating R2
def r_square(y_pred,Y):
    y_mean = np.sum(Y,axis = 0)//len(Y)
    ss_res = np.sum((Y-y_pred)**2)
    ss_tot = np.sum((Y-y_mean)**2)
    print(ss_res,ss_tot)
    return 1- (ss_res/ss_tot)
    

#through sklearn
lr=LinearRegression()
lr.fit(X,Y)
 
y_pred = lr.predict(X)
accuracy = r_square(y_pred, Y)
print("R2 accuracy: {:.2f}".format(accuracy))
#Prediction

#plot
plt.scatter(X,Y)
plt.plot(X,lr.predict(X))



