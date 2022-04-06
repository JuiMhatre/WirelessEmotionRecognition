import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split # Import train_test_split function
import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions
# baseline	3
# amusement	4
# stress		5
# time pressure 	2
# no stress	0
# interruption	1

class ReadData:
    def __init__(self):
        self.X=0
        self.Y=0
        self.names=[]

    def readSwellWesadData(self):
        dataset_wesad = pd.read_csv(
            'C:/Users/16786/PycharmProjects/EmotionRecognition/dataset/hrv/wesad/combined/classification/wesad-chest-combined-classification-hrv.csv')
        X_wesad = dataset_wesad.iloc[:, 0:11].values
        self.names = dataset_wesad.iloc[:,0:11].columns
        y_wesad = dataset_wesad.iloc[:, 66].values  # wesad
        y_wesad = y_wesad[np.newaxis, :]
        dataset_wesad = np.hstack((X_wesad,np.transpose(y_wesad)))
        dataset_swell = pd.read_csv(
            'C:/Users/16786/PycharmProjects/EmotionRecognition/dataset/hrv/swell/combined/classification/combined-swell-classification-hrv-dataset.csv')
        X_swell = dataset_swell.iloc[:, 0:11].values
        y_swell = dataset_swell.iloc[:, 67].values  # swell
        y_swell = y_swell[np.newaxis,:]+3 #no stress :3,interruption:4 time pressure:5
        dataset_swell = np.hstack((X_swell,np.transpose(y_swell)))
        dataset = np.concatenate((dataset_swell,dataset_wesad),axis=0)
        np.random.shuffle(dataset)
        self.X= dataset[:,0:11]
        self.Y = dataset[:,11]
        X_train, X_test_inter, y_train, y_test_inter = train_test_split(dataset[:,0:11],dataset[:,11] ,test_size=0.3,
                                                                                    random_state=1)
        X_val,X_test,y_val,y_test = train_test_split(X_test_inter,y_test_inter ,test_size=0.5,
                                                                                    random_state=1)
        # X_wesad = dataset_wesad.iloc[:, 0:11].values
        # y_wesad = dataset_wesad.iloc[:, 63].values  # wesad
        # X_train_wesad, X_test_wesad, y_train_wesad, y_test_wesad = train_test_split(X_wesad, y_wesad, test_size=0.3,
        #                                                                             random_state=1)
        # dataset_swell = pd.read_csv(
        #     'C:/Users/16786/PycharmProjects/EmotionRecognition/dataset/hrv/swell/combined/classification/combined-swell-classification-hrv-dataset.csv')
        # X_swell = dataset_swell.iloc[:, 0:11].values
        # y_swell = dataset_swell.iloc[:, 34].values  # swell
        # X_train_swell, X_test_swell, y_train_swell, y_test_swell = train_test_split(X_swell, y_swell, test_size=0.3,
        #                                                                             random_state=1)
        # X = np.concatenate((X_wesad, X_swell), axis=0)
        # Y = np.concatenate((y_wesad, y_swell), axis=0)
        #
        # # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
        # X_train = np.concatenate((X_train_wesad, X_train_swell), axis=0)
        # X_test = np.concatenate((X_test_wesad, X_test_swell), axis=0)
        # y_train = np.concatenate((y_train_wesad, y_train_swell), axis=0)
        # y_test = np.concatenate((y_test_wesad, y_test_swell), axis=0)
        return self.X, self.Y
        # return X_train, X_test, X_val, y_train,y_test,y_val
    def getXY(self):
        return self.X, self.Y
    def getSwellXY(self):
        dataset_swell = pd.read_csv(
            'C:/Users/16786/PycharmProjects/EmotionRecognition/dataset/hrv/swell/combined/classification/combined-swell-classification-hrv-dataset.csv')
        X_swell = dataset_swell.iloc[:, 0:11].values
        y_swell = dataset_swell.iloc[:, 67].values  # swell
        return X_swell,y_swell
    def getDataframe(self):
        dataset_wesad = pd.read_csv(
            'C:/Users/16786/PycharmProjects/EmotionRecognition/dataset/hrv/wesad/combined/classification/wesad-chest-combined-classification-hrv.csv')
        X_wesad = dataset_wesad.iloc[:, 0:11].values
        y_wesad = dataset_wesad.iloc[:, 66].values  # wesad
        y_wesad = y_wesad[np.newaxis, :]
        dataset_wesad = np.hstack((X_wesad, np.transpose(y_wesad)))
        dataset_swell = pd.read_csv(
            'C:/Users/16786/PycharmProjects/EmotionRecognition/dataset/hrv/swell/combined/classification/combined-swell-classification-hrv-dataset.csv')
        X_swell = dataset_swell.iloc[:, 0:11].values
        y_swell = dataset_swell.iloc[:, 67].values  # swell
        y_swell = y_swell[np.newaxis, :] + 3
        dataset_swell = np.hstack((X_swell, np.transpose(y_swell)))
        dataset = np.concatenate((dataset_swell, dataset_wesad), axis=0)
        np.random.shuffle(dataset)
        return pd.DataFrame(dataset)
    def readVideoDataHRV(self):
        dataset = pd.read_csv('C:/Users/16786/PycharmProjects/EmotionRecognition/dataset/clean.csv',
                                index_col=None,
                                usecols=['mean_rr', 'median_rr', 'sdrr', 'rmssd', 'sdsd', 'hr', 'pnn25',
                                         'pnn50', 'sd1', 'sd2', 'emotion'])
        dataset = dataset.sample(frac=1)
        dataset = np.array(dataset)
        self.X = dataset[0:,0:-1]
        self.Y = dataset[0:,-1]
        # X_train, X_test, y_train, y_test = train_test_split(dataset[:, 0:11], dataset[:, 11], test_size=0.3,
        #                                                                 random_state=1)
        return self.X,self.Y

