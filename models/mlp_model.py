#モジュールの読み込み
from __future__ import print_function

import pandas as pd
from pandas import Series,DataFrame

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
# from keras.optimizers import RMSprop
# from keras.optimizers import Adam
# from keras.utils import plot_model

import scipy.stats
from keras.utils import np_utils #Numpyユーティリティのインポート
from tensorflow.keras import datasets, layers, models, optimizers, losses




class MyNN:
    def __init__(self, setumei_size):
        #ニューラルネットワークの実装①
        model = Sequential()

        model.add(Dense(50, activation='relu', input_shape=(setumei_size,)))
        model.add(Dropout(0.2))

        model.add(Dense(50, activation='relu', input_shape=(setumei_size,)))
        model.add(Dropout(0.2))

        model.add(Dense(50, activation='relu', input_shape=(setumei_size,)))
        model.add(Dropout(0.2))

        # model.add(Dense(10, activation='softmax'))
        model.add(Dense(3, activation='softmax')) # for 3class

        model.summary()
        print("\n")

        #ニューラルネットワークの実装②
        model.compile(loss='mean_squared_error',
                        optimizer=RMSprop(),
                        metrics=['accuracy'])
        #勾配法には、Adam(lr=1e-3)という方法もある（らしい）。

        self.model = model
    
    def getModel(self):
        return self.model