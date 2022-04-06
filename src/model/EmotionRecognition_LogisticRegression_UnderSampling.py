# Importing libraries
# -------------------
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from ReadData import ReadData
import pickle
# Importing the dataset
# ---------------------
from scr.utils import Utils

readData = ReadData()
utils = Utils()

X, Y = readData.readSwellWesadData()


X,Y = utils.getBalancedData(X,Y)


# classes_emots =['amusement' 'baseline' 'interruption' 'no stress' 'stress', 'time pressure']
Y_binarize = label_binarize(Y, classes=[0,1,2,3,4,5])
n_classes = 6

k = 5
kf = KFold(n_splits=k, random_state=None)
model = LogisticRegression(solver='lbfgs',penalty='l1')

acc_score = []

split = 1
fpr = dict()
tpr = dict()
roc_auc = dict()
roc_store = np.zeros(shape=(k * n_classes, 4), dtype=object)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = Y_binarize[train_index], Y_binarize[test_index]

    pred_values = model.fit(X_train, Y[train_index]).predict(X_test)
    figname = "C:/Users/16786/PycharmProjects/EmotionRecognition/Results/LogisticRegression_confusionmatrix/LinearRegression_split" + str(
        split)
    utils.plot_confusion_matrix(pred_values, Y[test_index], np.unique(Y[test_index]), figname)
    acc = accuracy_score(pred_values, Y[test_index])

    acc_score.append(acc)
    for i in range(n_classes):
        y_score = model.fit(X_train, y_train[:, i]).decision_function(X_test)
        fpr[i], tpr[i], t = roc_curve(y_test[:, i], y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # fpr, tpr, t = roc_curve(y_test, pred_values[:, 1])
    colors = cycle(['blue', 'red', 'green'])
    for i, color in zip(range(n_classes), colors):
        roc_store[n_classes * (split - 1) + i][0] = fpr[i]
        roc_store[n_classes * (split - 1) + i][1] = tpr[i]
        roc_store[n_classes * (split - 1) + i][2] = color
        roc_store[n_classes * (split - 1) + i][3] = 'ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i])
        # plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
        #          label='ROC curve of class {0} (area = {1:0.2f})'
        #                ''.format(i, roc_auc[i]))
    split += 1
    print('F1 score is {}'.format(f1_score(Y[test_index], pred_values,average='micro')))
avg_acc_score = sum(acc_score) / k

print('accuracy of each fold - {}'.format(acc_score))

plt.figure()
for i in range(k * n_classes):
    plt.plot(roc_store[i][0], roc_store[i][1], color=roc_store[i][2], lw=1.5,
             label=roc_store[i][3])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
figname = "C:/Users/16786/PycharmProjects/EmotionRecognition/Results/LogisticRegression_ROC/LinearRegression_roc_auc"
plt.savefig(figname)

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
phys ={'bpm': 112.41903315229554, 'ibi': 533.7174526195863, 'sdnn': 150.31270570613165, 'sdsd': 139.86691628853757, 'rmssd': 201.55572498292025, 'pnn20': 1.0, 'pnn50': 0.6363636363636364, 'hr_mad': 99.7750727449079, 'sd1': 125.02767881446891, 'sd2': 162.08823571716675, 's': 63665.99579363563, 'sd1/sd2': 0.7713556647789908, 'breathingrate': 0.1786192730195588, 'MEAN_RR': 533.7174526195863, 'MEDIAN_RR': 482.2461849337213, 'SDRR': 150.31270570613165, 'SDRR_RMSSD': 0.7457625215997664, 'HR': 11.241903315229555, 'pnn25': 1.0}
input = [[phys.get('MEAN_RR'),phys.get('MEDIAN_RR'),phys.get('SDRR'),phys.get('rmssd'),phys.get('sdsd'),phys.get('SDRR_RMSSD'),phys.get('HR'),phys.get('pnn25'),phys.get('pnn50'),phys.get('sd1'),phys.get('sd2')]]
y_pred = model.predict(input)
print(y_pred)
