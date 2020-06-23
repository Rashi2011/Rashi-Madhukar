#IMPROVEMENT 66% to 82.83% accuracy
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv("DATASET/Heart.csv")
dataset = dataset.dropna()
X = dataset.iloc[:,:14].values
Y = dataset.iloc[:,14].values

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lb = LabelEncoder()
X[:,3] = lb.fit_transform(X[:,3])
X[:,13] = lb.fit_transform(X[:,13])
#imputer(It takes 2d array)
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
X = imp.fit_transform(X)
Y = lb.fit_transform(Y)
#OneHotEncoder
'''
ohe = OneHotEncoder(categorical_features = [3,13])
X = ohe.fit_transform(X).toarray()
'''

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#Splitting Dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.33,random_state = 0)

#Applying Dimensional Reduction
'''
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
'''
#Classification model
#1.Logistic Regression(Best amoung all)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,Y_train)

#2. KNN Classifier
'''
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski',p = 2)
classifier.fit(X_train,Y_train)
'''
#3. Decision Tree Classification
'''
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'gini',random_state = 0)
classifier.fit(X_train,Y_train)
'''

#4.Random Forest Classification
'''
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300, random_state =0)
classifier.fit(X_train,Y_train)
'''

#Prediction
Y_pred = classifier.predict(X_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)

#Other Performance measure
'''from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_pred))
print(classifier.score(X_train,Y_train))'''

#Only accuracy measure
from sklearn.metrics import f1_score,accuracy_score
#print("F1_Score   : {:.2f}%".format(f1_score(Y_test,Y_pred)*100))
print("accuracy_score = {:.2f}%".format(accuracy_score(Y_test,Y_pred)*100))


'''
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
plt.title('Heart Disease(Training set)')
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
plt.title('Heart Disease(Test set)')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.legend()
plt.show()
'''
