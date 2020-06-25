#PREDICTING WHICH GROUP SHOULD BE ASSIGNED ACCORDING TO THE PERFORMANCE OF EMPLOYEES
#(Improvement of v_accuracy .06  to 0.127 )
#Bad Dataset(119x71) More data required
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the Dataset
Dataset = pd.read_csv("DATASET/Business dataset/customerTargeting.csv")
X = Dataset.iloc[:,:-1].values
Y = Dataset.iloc[:,70].values

#replace nan with zero and inf with finite numbers
#np.nan_to_num(X)

#for missing data
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
X = imp.fit_transform(X)

#Feature Scaling 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#Splitting into Train And Test 
'''
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 1/4,random_state = 0)
'''
#Applying PCA
'''
from sklearn.decomposition import PCA
pca= PCA(n_components = 2)
X =pca.fit_transform(X)
#X_test = pca.transform(X_test)
'''

#K-mean Clustering Technique

from sklearn.cluster import KMeans
categ = KMeans(n_clusters=3,init ='k-means++',max_iter=300,n_init=10,random_state = 0)
km = categ.fit_predict(X)

#Agglomerative Clustering Technique
'''
from sklearn.cluster import AgglomerativeClustering
categ = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
km = categ.fit_predict(X)
'''
#Confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y,km)

#accuracy Score
from sklearn.metrics import v_measure_score
print("v_measure_score",v_measure_score(Y,km))
#print("accuracy_score  = {:.2f}%".format(accuracy_score(Y,km)*100))

#Visualising the cluster
'''
plt.scatter(X[Y == 0,0], X[Y == 0,1], s=100, c='red', label = 'cluster 1')
plt.scatter(X[Y == 1,0], X[Y == 1,1], s=100, c='green', label = 'cluster 2')
plt.scatter(X[Y == 2,0], X[Y == 2,1], s=100, c='black', label = 'cluster 3')
#plt.scatter(y_pred.cluster_centers_[:,0], y_pred.cluster_centers_[:,1], s = 300, c='yellow')
plt.title('K-mean_cluster(real)')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.show()

#Visualising the cluster
plt.scatter(X[km == 0,0], X[km == 0,1], s=100, c='red', label = 'cluster 1')
plt.scatter(X[km == 1,0], X[km == 1,1], s=100, c='green', label = 'cluster 2')
plt.scatter(X[km == 2,0], X[km == 2,1], s=100, c='black', label = 'cluster 3')

#plt.scatter(y_pred.cluster_centers_[:,0], y_pred.cluster_centers_[:,1], s = 300, c='yellow')
plt.title('K-mean_cluster')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.show()
'''