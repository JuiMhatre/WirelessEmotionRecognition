from itertools import cycle

from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, f1_score,auc
from sklearn.preprocessing import label_binarize, StandardScaler
from imblearn.pipeline import Pipeline
from scr.ReadData import ReadData
from scr.utils import Utils
import numpy as np
import matplotlib as plt
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, cross_validate, cross_val_predict, TimeSeriesSplit, \
    train_test_split

readData = ReadData()
utils = Utils()

X, Y = readData.readSwellWesadData()
scaler = StandardScaler()
X = scaler.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X,Y ,test_size=0.3,random_state=1)
# X,Y = X_train,y_train


# classes_emots =['amusement' 'baseline' 'interruption' 'no stress' 'stress', 'time pressure']
Y_binarize = label_binarize(Y, classes=[0,1,2,3,4,5])
n_classes = 6


# model=RandomForestClassifier(criterion='entropy')
# # Define SMOTE-Tomek Links
# resample=SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
# # Define pipeline
# pipeline=Pipeline(steps=[('r', resample), ('m', model)])
# # Define evaluation procedure (here we use Repeated Stratified K-Fold CV)
# cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# # features = list(readData.names)
# # cv = TimeSeriesSplit(n_splits = 5).split(features)
# # Evaluate model
# scoring=['accuracy','precision_macro','recall_macro']
# scores = cross_validate(pipeline, X, Y, scoring=scoring, cv=cv, n_jobs=-1)
# # ypred = cross_val_predict(model,X,Y,cv=cv)
# # figname = "C:/Users/16786/PycharmProjects/EmotionRecognition/Results/RandomForest/RandomForestConfMat"
# # utils.plot_confusion_matrix(ypred,Y,np.unique(Y))
#
# # summarize performance
# print('Mean Accuracy: %.4f' % np.mean(scores['test_accuracy']))
# print('Mean Precision: %.4f' % np.mean(scores['test_precision_macro']))
# print('Mean Recall: %.4f' % np.mean(scores['test_recall_macro']))
#
# ypred = model.predict(X_test)
# figname = "C:/Users/16786/PycharmProjects/EmotionRecognition/Results/RandomForest/RandomForestConfMat"
# utils.plot_confusion_matrix(ypred,y_test,np.unique(Y))



k = 5
kf = KFold(n_splits=k, random_state=None)
model = RandomForestClassifier(criterion='entropy')

acc_score = []

split = 1
mean_fpr = np.linspace(0, 1, 100)
fpr = dict()
tpr = dict()
roc_auc = dict()
roc_store = np.zeros(shape=(k * n_classes, 4), dtype=object)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = Y_binarize[train_index], Y_binarize[test_index]

    X_train, y_tr_ind = utils.performSMOTE(X_train, Y[train_index])
    y_train = label_binarize(y_tr_ind, classes=[0,1,2,3,4,5])

    pred_values = model.fit(X_train, y_tr_ind).predict(X_test)
    figname = "C:/Users/16786/PycharmProjects/EmotionRecognition/Results/RandomForest/RandomForest_split" + str(
        split)
    utils.plot_confusion_matrix(pred_values, Y[test_index], np.unique(Y[test_index]),figname)
    acc = accuracy_score(pred_values, Y[test_index])

    acc_score.append(acc)
    # for i in range(n_classes):
    #     y_score = model.fit(X_train, y_train[:, i])
    #     fpr[i], tpr[i], t = roc_curve(y_test[:, i], y_score)
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #
    # # fpr, tpr, t = roc_curve(y_test, pred_values[:, 1])
    # colors = cycle(['blue', 'red', 'green'])
    # for i, color in zip(range(n_classes), colors):
    #     roc_store[n_classes * (split - 1) + i][0] = fpr[i]
    #     roc_store[n_classes * (split - 1) + i][1] = tpr[i]
    #     roc_store[n_classes * (split - 1) + i][2] = color
    #     roc_store[n_classes * (split - 1) + i][3] = 'ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i])
    # split += 1
    print('F1 score is {}'.format(f1_score(Y[test_index], pred_values,average='micro')))
avg_acc_score = sum(acc_score) / k

print('accuracy of each fold - {}'.format(acc_score))
# plt.figure()
# for i in range(k * n_classes):
#     plt.plot(roc_store[i][0], roc_store[i][1], color=roc_store[i][2], lw=1.5,
#              label=roc_store[i][3])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic for multi-class data')
# plt.legend(loc="lower right")
# figname = "C:/Users/16786/PycharmProjects/EmotionRecognition/Results/RandomForest/RandomForest_roc_auc"
# plt.savefig(figname)

