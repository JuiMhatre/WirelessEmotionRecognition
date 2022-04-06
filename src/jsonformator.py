import numpy as np
import pandas as pd
import random
import matplotlib.pyplot
from sklearn import svm
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
#import the breast cancer dataset
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from scr.ReadData import ReadData
from scr.utils import Utils

readData = ReadData()
utils = Utils()
X, Y = readData.readVideoDataHRV()
print(np.unique(Y,return_counts=True))
scaler = StandardScaler()
X = scaler.fit_transform(X)
# X, Y = utils.performSMOTE(X,Y)
from sklearn.decomposition import PCA
# pca = PCA(n_components=6)
# principalComponents = pca.fit_transform(X)
# principalDf = pd.DataFrame(data = principalComponents
#              , columns = ['pc1', 'pc2','pc3','pc4','pc5','pc6'])
# X= np.array(principalDf)
print(np.unique(Y,return_counts=True))

#splitting the model into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X,Y,
                                                     test_size=0.30,
                                                    random_state=101)
#training a logistics regression model
logmodel = KNeighborsClassifier(n_neighbors=2)




#defining various steps required for the genetic algorithm
def initilization_of_population(size,n_feat):
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat,dtype=np.bool)
        chromosome[:int(0.3*n_feat)]=False
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population

def fitness_score(population,model):
    scores = []
    for chromosome in population:
        model.fit(X_train[:,chromosome],y_train)
        predictions = model.predict(X_test[:,chromosome])
        # scores.append(f1_score(y_test,predictions,average='micro'))
        scores.append(accuracy_score(y_test,predictions))
    scores, population = np.array(scores), np.array(population)
    inds = np.argsort(scores)
    return list(scores[inds][::-1]), list(population[inds,:][::-1])

def selection(pop_after_fit,n_parents):
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen

def crossover(pop_after_sel):
    population_nextgen=pop_after_sel
    for i in range(len(pop_after_sel)):
        child=pop_after_sel[i]
        child[3:7]=pop_after_sel[(i+1)%len(pop_after_sel)][3:7]
        population_nextgen.append(child)
    return population_nextgen

def mutation(pop_after_cross,mutation_rate):
    population_nextgen = []
    for i in range(0,len(pop_after_cross)):
        chromosome = pop_after_cross[i]
        for j in range(len(chromosome)):
            if random.random() < mutation_rate:
                chromosome[j]= not chromosome[j]
        population_nextgen.append(chromosome)
    #print(population_nextgen)
    return population_nextgen

def generations(size,n_feat,n_parents,mutation_rate,n_gen,model):
    best_chromo= []
    best_score= []
    population_nextgen=initilization_of_population(size,n_feat)
    for i in range(n_gen):
        scores, pop_after_fit = fitness_score(population_nextgen,model)
        print(scores[:2])
        pop_after_sel = selection(pop_after_fit,n_parents)
        pop_after_cross = crossover(pop_after_sel)
        population_nextgen = mutation(pop_after_cross,mutation_rate)
        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])
    return best_chromo,best_score

chromo,score=generations(size=X.shape[0],n_feat=X.shape[1],n_parents=100,mutation_rate=0.10,
                     n_gen=38,model=logmodel)


logmodel.fit(X_train[:,chromo[-1]],y_train)
predictions_knn = logmodel.predict(X_test[:,chromo[-1]])


# model = RandomForestClassifier(n_estimators=1800,max_depth=10,max_features=6,bootstrap=True)
# # evaluate the model
# # chromo,score=generations(size=X.shape[0],n_feat=X.shape[1],n_parents=100,mutation_rate=0.10,
# #                      n_gen=38,model=model)
#
# model.fit(X_train[:,chromo[-1]],y_train)
# predictions_rf1 = model.predict(X_test[:,chromo[-1]])
#
# model = RandomForestClassifier(n_estimators=600,max_depth=10,max_features=6,bootstrap=True)
# # chromo,score=generations(size=X.shape[0],n_feat=X.shape[1],n_parents=100,mutation_rate=0.10,
# #                      n_gen=38,model=model)
#
# model.fit(X_train[:,chromo[-1]],y_train)
# predictions_rf2 = model.predict(X_test[:,chromo[-1]])
#
# poly = svm.SVC(kernel='poly', degree=3, C=1)
# # chromo,score=generations(size=X.shape[0],n_feat=X.shape[1],n_parents=100,mutation_rate=0.10,
# #                      n_gen=38,model=model)
#
# poly.fit(X_train, y_train)
# poly_pred = poly.predict(X_test)
#
#
#
# def comparepredictions(knn, rf, svm):
#     final =[]
#     for i in range(len(knn)):
#         k =knn[i]
#         r = rf[i]
#         s = svm[i]
#         if r==s:
#             final.append(r)
#         else:
#             final.append(k)
#     return final
#
#
# predictions = comparepredictions(predictions_knn,predictions_rf1,poly_pred)
print("Accuracy score after knn algorithm is= "+str(accuracy_score(y_test,predictions_knn)))
print("Accuracy score after knn algorithm is= "+str(confusion_matrix(y_test,predictions_knn)))
# print("Accuracy score after rf algorithm is= "+str(accuracy_score(y_test,predictions_rf1)))
# print("Accuracy score after rf2 algorithm is= "+str(accuracy_score(y_test,predictions_rf2)))
# print("Accuracy score after svm algorithm is= "+str(accuracy_score(y_test,poly_pred)))
# print("Accuracy score after genetic algorithm is= "+str(accuracy_score(y_test,predictions)))
# print (confusion_matrix(y_test, predictions))


