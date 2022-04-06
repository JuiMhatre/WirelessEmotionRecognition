from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
from matplotlib import pyplot
# define dataset
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler

from scr.ReadData import ReadData


import  numpy as np

readData = ReadData()
X, Y = readData.readVideoDataHRV()
print(np.unique(Y,return_counts=True))
scaler = StandardScaler()
X = scaler.fit_transform(X)# define the model
for i in range (7,16):
	clustering = AgglomerativeClustering(n_clusters = i).fit(X)
	# print(clustering.labels_)
	print(accuracy_score(Y,clustering.labels_))
	print(confusion_matrix(Y,clustering.labels_))