import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# matplotlib inline
from sklearn.metrics import f1_score, accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from scr.ReadData import ReadData
from scr.utils import Utils
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
import pickle
resultsdir ="C:/Users/16786/PycharmProjects/EmotionRecognition/Results/TrialResults/"
readData = ReadData()
utils = Utils()

X, Y = readData.readVideoDataHRV()

input =[]
print(np.unique(Y,return_counts=True))
scaler = StandardScaler()
X = scaler.fit_transform(X)
X, Y = utils.performSMOTETomek(X,Y)
X_train, X_test, y_train, y_test = train_test_split(X,Y,
                                                     test_size=0.30,
                                                       random_state=1)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(f1_score(y_test,y_pred,average='micro'))
print(confusion_matrix(y_test,y_pred))
filename = 'finalized_model.sav'
pickle.dump(knn, open(filename, 'wb'))
