import itertools

import matplotlib.pyplot as plt
import numpy as np
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.metrics import confusion_matrix


class Utils:
    def __init__(self):
        self.X = 0
        self.Y = 0

    def plot_confusion_matrix(self, predicted_labels_list, y_test_list, class_names, figname):
        cnf_matrix = confusion_matrix(y_test_list, predicted_labels_list)  # (tn, fp, fn, tp)
        np.set_printoptions(precision=2)

        plt.figure()
        self.generate_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                                       title='Normalized confusion matrix')
        plt.savefig(figname)

    def generate_confusion_matrix(self, cnf_matrix, classes, normalize=False, title='Confusion matrix'):
        if normalize:
            cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

        else:
            print('Confusion matrix, without normalization')

        plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
        plt.title(title)
        plt.colorbar()

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cnf_matrix.max() / 2.

        for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
            plt.text(j, i, format(cnf_matrix[i, j], fmt), horizontalalignment="center",
                     color="white" if cnf_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        return cnf_matrix

    def getBalancedData(self,X,Y):
        print(np.unique(Y,return_counts=True))
        #(array([0., 1., 2., 3., 4., 5.]), array([ 71640,  23064,  40946, 212400, 110943,  68295], dtype=int64))
        res_0 = [x for x in range(len(Y)) if Y[x] == 0]
        res_1 = [x for x in range(len(Y)) if Y[x] == 1]
        res_2 = [x for x in range(len(Y)) if Y[x] == 2]
        res_3 = [x for x in range(len(Y)) if Y[x] == 3]
        res_4 = [x for x in range(len(Y)) if Y[x] == 4]
        res_5 = [x for x in range(len(Y)) if Y[x] == 5]
        # cap_3_4_5 = 45217
        cap_3_4_5 = 23064
        res_3 = res_3[cap_3_4_5:]
        res_4 = res_4[cap_3_4_5:]
        res_5 = res_5[cap_3_4_5:]
        res_0 = res_0[cap_3_4_5:]
        res_1 = res_1[cap_3_4_5:]
        res_2 = res_2[cap_3_4_5:]
        res =np.append(np.append(np.append(np.append(res_3,res_4),res_5),res_0),res_2)
        Y=np.delete(Y,res, axis =0)
        X= np.delete(X,res,axis=0)
        dataset = np.column_stack((X, Y))
        np.random.shuffle(dataset)
        X = dataset[:, 0:11]
        Y = dataset[:, 11]
        print(np.unique(Y,return_counts=True))
        return X,Y

    def performSMOTE(self, X, Y):
        sm = SMOTE(random_state=42)
        X,Y = sm.fit_resample(X, Y)
        return X,Y
    def performSMOTETomek(self, X, Y):
        sm = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
        # sm = SMOTE(random_state=42)
        X,Y = sm.fit_resample(X, Y)
        return X,Y


