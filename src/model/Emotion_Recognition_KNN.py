import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# matplotlib inline
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from scr.ReadData import ReadData
from scr.utils import Utils
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier

resultsdir ="C:/Users/16786/PycharmProjects/EmotionRecognition/Results/TrialResults/"
readData = ReadData()
utils = Utils()

X, Y = readData.readVideoDataHRV()
scaler = StandardScaler()
X = scaler.fit_transform(X)
# X, Y = utils.performSMOTETomek(X,Y)
X_train , X_test, y_train, y_test = train_test_split(X,Y ,test_size=0.3,random_state=1)
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
figname = "C:/Users/16786/PycharmProjects/EmotionRecognition/Results/KNN_confusionmatrix/KNN_mat_withoutSMOTEtomek_500"
y_pred = knn.predict(X_test)
print(accuracy_score(y_test,y_pred))
# utils.plot_confusion_matrix(y_pred,y_test, np.unique(y_test), figname)
print(f1_score(y_test,y_pred,average ='micro'))









# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# X, Y = utils.performSMOTETomek(X,Y)
# Y = label_binarize(Y, classes=[0,1,2,3,4,5])
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import tree
# clf = DecisionTreeClassifier(random_state=1234)
# model = clf.fit(X, Y)
# text_representation = tree.export_text(clf)
# print(text_representation)












# df = readData.getDataframe()
# # print(df.isnull().sum())
# # print(df.head())
# # sns.countplot(x=11, data = df)
# # plt.show()
#
# #Imbalance is observered hence SMOTETomek is performed
# X, Y = readData.readSwellWesadData()
# X, Y = utils.performSMOTETomek(X,Y)
# Y_binarize = label_binarize(Y, classes=[0,1,2,3,4,5])
# dataset = np.concatenate((X,np.transpose([Y])),axis=1)
# df = pd.DataFrame(dataset)
# # print(df.head())
# # sns.countplot(x=11, data = df)
# # plt.savefig(resultsdir+"Data_Distribution")
#
# df.describe()            # Target Variable
# X = df.drop(11,axis=1)  # Independent Variables
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# df = pd.DataFrame(X)
#
# # plt.figure(figsize=(20,20))
# # sns.heatmap(df.corr(), annot=True,cmap='viridis')
# # plt.savefig(resultsdir+"correlation_map")
#
# X = df.iloc[:,:-1]
#
# y = Y_binarize
#
#
# rf = RandomForestClassifier(n_estimators=5000,random_state=11)
# rf.fit(X,y)
# feat_imp = pd.DataFrame(rf.feature_importances_)
# feat_imp.index = pd.Series(df.iloc[:,:-1].columns)
# feat_imp = (feat_imp*100).copy().sort_values(by=0,ascending=False)
# feat_imp = feat_imp.reset_index()
# print(feat_imp)
# for var in np.arange(feat_imp.shape[0],9,-1):
#     X_new = X[feat_imp.iloc[:var,0]].copy()
#     X_train, X_test, y_train,y_test = train_test_split(X_new,y,test_size=0.2,random_state=11)
#     final_rf = RandomForestClassifier(random_state=11)
#     gscv = GridSearchCV(estimator=final_rf,param_grid={
#         "n_estimators":[100,500,1000,5000],
#         "criterion":["gini","entropy"]
#     },cv=5,n_jobs=-1,scoring="f1_weighted")
#
#     model = gscv.fit(X_train,y_train)
#     print("SMOTE Model Created using the top {} variables".format(var))
#     print("F1 Score: {}".format(model.best_score_))
#     print("Best Model {}".format(model.best_estimator_))
#     print("-"*30)