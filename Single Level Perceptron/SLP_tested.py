#Added the bias Internally in the Dataset
def stepf(x,A0):
    #y = input("The threshold value:")
    #A0 = y
    if x>= A0:
        return 1
    else:
        return 0

#importing libraries
import pandas as pd
import numpy as np
from numpy.random import random_sample

#importing dataset
df = pd.read_csv("DATASET/sonar.all-data.csv")
#Shuffling Dataset
from sklearn.utils import shuffle
Dataset = shuffle(df)

X = Dataset.iloc[:,:-1]
'''
bias = np.ones((207,1))
X['bias'] = bias'''
X = X.values
Y = Dataset.iloc[:,60].values

#Categorical Encoding
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
Y = lb.fit_transform(Y)

class Perceptron:
   
    def __init__(self,no_of_inputs):
        print("\nno_of_inputs\n",no_of_inputs) #no. of inputs = 60
        #self.w = np.random.rand(60,1)
        self.w = np.random.randint(3,size = (60,1))
        print("\n Initial Weight: \n",self.w)
        self.c = 0.1 #learning rate
        
        
    def weight_adjustment(self,del_w):
        #print("\ndel_w\n",del_w)
        #Updating the weight
        for i in range(0,60): #self.w is 60x1 matrix
                self.w[i] = self.w[i] +del_w[i]
        #print("Updation in Weight\n",self.w)
        return self.w
         
        
    def Learning_rule(self,w):
        no_of_Observations = 207
        y = []
        error = []
        #Using Perceptron Learning Rule
        #Calculating the Pred_op
        net = np.dot(X,w)
        mean_o = np.mean(net)
        #print(mean_o)
        for i in net:
            y.append(stepf(i,mean_o))
        #print("\nPredicted Output: \n",y)
       
        #Performance measure(Confusion matrix)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(Y,y)
        #print("\ncm\n",cm)
        
        from sklearn.metrics import accuracy_score
        print("\naccuracy\n",accuracy_score(Y, y))
        
        #Calculating the Error Loss
        for i in range(no_of_Observations):
            error.append(Y[i] - y[i])
        #print("\nError\n",error)
        
        #Calculating del_w
        del_w = self.c*(np.dot(X.T,error))
        return del_w
    
    
    
del_w = np.zeros((60))
epoch = 0 
p = Perceptron(60)   
while epoch<5:
    
    w = p.weight_adjustment(del_w)
    del_w = p.Learning_rule(w)
    epoch +=1
    
