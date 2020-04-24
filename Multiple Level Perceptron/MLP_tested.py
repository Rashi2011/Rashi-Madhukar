#Added the bias Internally in the Dataset
#sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Step function
def stepf(x,A0):
    #y = input("The threshold value:")
    #A0 = y
    if x>= A0:
        return 1
    else:
        return 0
    
#import Libraries
import numpy as np
import pandas as pd


#import Dataset
dataset = pd.read_csv("DATASET/Social_Network_Ads.csv")
bias = np.ones((400,1))
X = dataset.iloc[:,:-1].values

#Categorical Encoding
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
X[:,1] = lb.fit_transform(X[:,1])

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#continue
X = pd.DataFrame(X)
X['bias']= bias
X = X.values
Y = dataset.iloc[:,4]

class MLP:
    def __init__(self):
        # 4 input nodes 3 hidden nodes and one output node
        #self.w = np.random.randint(5,size = (5,4)) 
        #Adding the bias in the i/p as well asin the hidden_layer as a new node itself
        self.w = [[3,2,0,4],[3,4,3,5],[1,4,2,6],[3,4,2,1],[1,3,4,2]]
        print("Initial Weight\n", self.w)
        #self.w1 = np.random.randint(5,size= (4,1))
        self.w1 = [[1],[1],[4],[5]]
        print("Initialized 2nd layer weight\n",self.w1)
        self.c = 0.1
    
        
    def hidden_1(self,X,epoch,del_w):
        layer1_op = np.zeros((400,4))
        self.w = np.add(self.w,del_w)
        #print("\nUpdated Weight Between i/p to hidden_layer\n\n ",self.w)
        
        X2 =  np.dot(X,self.w) 
        #print(X2[0][0])
        for i in range(len(layer1_op)):
            # iterate through columns
            for j in range(len(layer1_op[0])):
               layer1_op[i][j] = layer1_op[i][j] +round(sigmoid(X2[i][j]),3)
        #print("\no/p of Hidden_layer 1\n\n",layer1_op)
        return layer1_op
    
        
    def outPut_layer(self,layer1_op,epoch,del_w1):
        pred_y = np.zeros((400,1))
        #print("Indirect layer1_op\n",layer1_op)
        if epoch > 0:
            self.w1 = np.add(self.w1,del_w1) 
           
        X3 = np.dot(layer1_op,self.w1)
        for i in range(len(pred_y)):
            # iterate through columns
            for j in range(len(pred_y[0])):
                pred_y[i][j] = pred_y[i][j] + round(sigmoid(X3[i][j]),3)
        #print("\n\n Predicted output\n",pred_y)
              
        mean_o = np.mean(pred_y)
        print("mean\n",mean_o)
        y = np.zeros((400,1))
        for i in range(len(y)):
            # iterate through columns
            for j in range(len(y[0])):
                y[i][j] = y[i][j] + stepf(pred_y[i][j],mean_o)
        #print("\n\n Predicted output\n",y)
        
        
        from sklearn.metrics import f1_score
        f1_score = f1_score(Y, y)
        print("F1_Score =",f1_score)
        
        #Other Performance Measures
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(Y,y)
        #print("\ncm\n",cm)
        return pred_y
    
        
    def Bp(self,pred_y,layer1_op):
        e = []
        del_w1 = np.zeros((4,1))
        del_w = np.zeros((5,4))
        #print("Indirect y",pred_y)
        
        #Error in Output layer e
        for i in range(0,len(Y)):
            e.append(self.c*(Y[i]-pred_y[i]))
        X4= np.dot(layer1_op.T,e)
        #print("X4\n",X4)
        for i in range(len(del_w1)):
            for j in range(len(del_w1[0])):
                del_w1[i][j] =del_w1[i][j] + round(X4[i][j],3)
        #print("\n\n del_w\n\n",del_w1)
        
        #Error in the Hidden layer e_h1
        e_h1 =self.c*( np.dot(e,np.transpose(self.w1)))
        X5 = np.dot(X.T,e_h1)
        for i in range(len(del_w)):
            for j in range(len(del_w[0])):
                del_w[i][j] =del_w[i][j] + round(X5[i][j],3)
        #print("\n\n del_w\n\n",del_w)
        return del_w,del_w1


w = np.zeros((5,4))
w1 = np.zeros((4,1))
epoch  = 0 
mlp = MLP()      
while(epoch < 15):   
    
    layer1_op = mlp.hidden_1(X,epoch,w)
    pred_y = mlp.outPut_layer(layer1_op,epoch,w1)
    w,w1 = mlp.Bp(pred_y,layer1_op)
    epoch+=1

