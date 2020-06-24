#(ACCURACY 60.7% to 72.07% )
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
df = pd.read_csv("DATASET/Sports dataset/Sports.csv")

#Categorically Encoding
'''
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lb = LabelEncoder()
X[:,3] = lb.fit_transform(X[:,3])
X[:,4] = lb.fit_transform(X[:,4])
X[:,6] = lb.fit_transform(X[:,6])
X[:,7] = lb.fit_transform(X[:,7])
'''
#Feature Engineering
team_value = []
playing_style = []
no_of_injured = []
coach_experience = []
for i in df.iloc[:,4].values:
    if i == "Less_Than_Four_Billion":
        team_value.append(0)
    elif i == "Above_Four_Billion":
        team_value.append(1)
    else:
        team_value.append(0)

for i in df.iloc[:,5].values:
    if i == "Relaxed":
        playing_style.append(0)
    elif i == "Aggrresive_Offense" or i == "Agressive_Defence":
        playing_style.append(1)
    elif i == "Balanced":
        playing_style.append(2)
    else:
        playing_style.append(0)
        
for i in df.iloc[:,7].values:
    if i == "eight":
        no_of_injured.append(8)
    elif i == "seven":
        no_of_injured.append(7)
    elif i == "six":
        no_of_injured.append(6)
    elif i == "five":
        no_of_injured.append(5)
    elif i == "four":
        no_of_injured.append(4)
    elif i == "three":
        no_of_injured.append(3)
    else:
        no_of_injured.append(0)
    

for i in df.iloc[:,8].values:
    if i == "Beginner":
        coach_experience.append(0)
    elif i == "Intermediate":
        coach_experience.append(1)
    elif i == "Advanced":
        coach_experience.append(2)
    else:
        coach_experience.append(0)

#Deleting the unwanted columns
df.drop(["Team_Value", "Playing_Style", "Number_Of_Injured_Players", 
                            "Coach_Experience_Level"], axis = 1, inplace = True)
#Adding the new revised columns
df['Team_value'] = team_value
df['Playing_Style'] = playing_style 
df['no_of_injured'] = no_of_injured 
df['coach_experience'] = coach_experience  

X = df.iloc[:,1:10].values
Y = df.iloc[:,0].values

#Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test  = train_test_split(X,Y,test_size = 1/3,random_state = 0)


#Applying PCA
'''
from sklearn.decomposition import PCA
pca= PCA(n_components = 2)
X_train =pca.fit_transform(X_train)
X_test = pca.transform(X_test)
'''
#Classification model
#1.Logistic Regression
'''
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,Y_train)
'''
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

#4.Random Forest Classification(works best than all above)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300, random_state =0)
classifier.fit(X_train,Y_train)


#Prediction
y_pred = classifier.predict(X_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
#Other accuracy measure
from sklearn.metrics import accuracy_score
print("\naccuracy_score: {:.2f}%".format(accuracy_score(Y_test,y_pred)*100))

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
'''