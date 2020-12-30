#IMPROVEMENT 66% to 82.83% accuracy
#import libraries
import numpy as np
import pandas as pd


#importing the dataset
dataset = pd.read_csv("DATASET/Heart.csv")
dataset = dataset.dropna()
X = dataset.iloc[:,1:14].values
Y = dataset.iloc[:,14].values
    
#plt.bar(Y)
#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lb = LabelEncoder()
X[:,2] = lb.fit_transform(X[:,2])       #chest pain(typical,asymptomatic,non_typical,non-anginal)
f=X[:,12].tolist()

for i in range(len(f)):
    if f[i]=='normal':
        f[i]=3
    if f[i]=='fixed':
        f[i]=6
    if f[i]=='reversable':
        f[i]=7

X[:,12]=f
#use one hot encoder for chest pain column
ohe=OneHotEncoder(categorical_features=[2])
X=ohe.fit_transform(X).toarray()
X=X[:,1:]              #remove dummy variable

#imputer(It takes 2d array)
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
X = imp.fit_transform(X)
Y = lb.fit_transform(Y)

#Plotting the distribution of class labels
count_p=0;count_n=0
for i in Y:
    if i==0:
        count_n+=1
    else:
        count_p+=1
print(count_p,count_n)

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

#Classification model
#1.Logistic Regression(82.5)(Precision =87.8(Less data ,less dimensions, less affected  by outliers)
'''
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,Y_train)
'''

#2. KNN Classifier(79.5) precision= 85
'''
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski',p = 2)
classifier.fit(X_train,Y_train)
'''

#3. SVM(83.5)(precision=92.11)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(X_train,Y_train)


#4. Decision Tree Classification(73.7) Precision = 72
'''
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'gini',random_state = 0)
classifier.fit(X_train,Y_train)
'''

#5.Random Forest Classification(78)  Precision = 81.4
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

from sklearn.metrics import classification_report,roc_auc_score
'''
print(classification_report(Y_test,Y_pred))
print(classifier.score(X_train,Y_train))'''


ROC = roc_auc_score(Y_test, Y_pred)
print(ROC)

#Only accuracy measure
from sklearn.metrics import precision_score
#print("F1_Score   : {:.2f}%".format(f1_score(Y_test,Y_pred)*100))
#print("accuracy_score = {:.2f}%".format(accuracy_score(Y_test,Y_pred)*100))
print("Precison_Score   : {:.2f}%".format(precision_score(Y_test,Y_pred)*100))
