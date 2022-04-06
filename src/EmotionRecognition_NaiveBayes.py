import numpy as np
from sklearn.metrics import  f1_score,accuracy_score

# loading the iris dataset
from sklearn.preprocessing import StandardScaler

from scr.ReadData import ReadData
from scr.utils import Utils
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB

readData = ReadData()
utils = Utils()
X, y = readData.readVideoDataHRV()
scaler = StandardScaler()
X = scaler.fit_transform(X)
# X, y = utils.getBalancedData(X,y)
X, y = utils.performSMOTE(X,y)
# dividing X, y into train and test data
k=5
gnb = GaussianNB()
kf = KFold(n_splits=k, random_state=None)
i=1
acc_sum =0
f1_sum =0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = y[train_index], y[test_index]

# training a Naive Bayes classifier
    gnb.fit(X_train, y_train)
    gnb_predictions = gnb.predict(X_test)
    acc_sum += accuracy_score(y[test_index], gnb_predictions)
    f1_sum += f1_score(y[test_index], gnb_predictions, average='micro')
    # print('F1 score is {}'.format(f1_score(y[test_index], gnb_predictions, average='micro')))
    # print('accuracy score is {}'.format(accuracy(y[test_index], gnb_predictions, average='micro')))

    figname = "C:/Users/16786/PycharmProjects/EmotionRecognition/Results/NaiveBayes/nb_confmat"+str(i)
# creating a confusion matrix
    utils.plot_confusion_matrix(gnb_predictions,y_test,np.unique((y_test)),figname)
    i=i+1
print('F1 score is {}'.format(f1_sum/k))
print('accuracy score is {}'.format(acc_sum/k))

