#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
Dataset = pd.read_csv("DATASET/Sports dataset/Sports.csv")
X = Dataset.iloc[:,1:9].values
Y = Dataset.iloc[:,0].values

#Categorically Encoding
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
X[:,3] = lb.fit_transform(X[:,3])
X[:,4] = lb.fit_transform(X[:,4])
X[:,6] = lb.fit_transform(X[:,6])
X[:,7] = lb.fit_transform(X[:,7])

#Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test  = train_test_split(X,Y,test_size = 1/3,random_state = 0)

#Applying PCA
from sklearn.decomposition import PCA
pca= PCA(n_components = 2)
X_train =pca.fit_transform(X_train)
X_test = pca.transform(X_test)

#Applying Knn
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) 
classifier.fit(X_train,Y_train)

#Prediction
y_pred = classifier.predict(X_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)

#Other accuracy measure
from sklearn.metrics import average_precision_score
print("\naverage_precision_score:",average_precision_score(Y_test,y_pred))

#Visualising the training set result
from matplotlib.colors import ListedColormap
X_set,Y_set = X_train,Y_train
X1,X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() -1,stop=X_set[:, 0].max() +1,step=0.01),
                    np.arange(start=X_set[:, 1].min() -1,stop=X_set[:, 1].max() +1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
              alpha = 0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(Y_set)):
     plt.scatter(X_set[Y_set == j, 0],X_set[Y_set == j, 1],
                 c=ListedColormap(('red','green'))(i) ,label=j)
plt.title('Sports Analysis(Training set)')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.legend()
plt.show()

#Visualising the test set result
from matplotlib.colors import ListedColormap
X_set,Y_set = X_test,Y_test
X1,X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() -1,stop=X_set[:, 0].max() +1,step=0.01),
                    np.arange(start=X_set[:, 1].min() -1,stop=X_set[:, 1].max() +1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
              alpha = 0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(Y_set)):
     plt.scatter(X_set[Y_set == j, 0],X_set[Y_set == j, 1],
                 c=ListedColormap(('red','green'))(i) ,label=j)
plt.title('Sports Analysis(Test set)')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.legend()
plt.show()
