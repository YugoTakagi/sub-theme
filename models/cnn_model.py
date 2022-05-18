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


class MyCNN:
    def __init__(self, input_size):
        #@brief Criate fundamental model of CNN
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 1), activation='relu', input_shape=(input_size, 1, 1)))
        self.model.add(layers.MaxPooling2D((2, 1)))
        self.model.add(layers.Conv2D(64, (3, 1), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 1)))
        self.model.add(layers.Conv2D(64, (3, 1), activation='relu'))

        self.model.add(layers.MaxPooling2D((2, 1)))
        self.model.add(layers.Conv2D(64, (3, 1), activation='relu'))

        # By using layers.Flatten(), Convert tensor to scoler
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(Dropout(0.2))

        self.model.add(layers.Dense(10, activation='relu'))
        self.model.add(Dropout(0.2))

        self.model.add(layers.Dense(3, activation='softmax'))

        self.model.summary()
        print("\n")

        #@brier Compile model and Learning
        self.model.compile(optimizer='adam',
                    #   loss='sparse_categorical_crossentropy',
                    # loss='mean_squared_error',
                    #   loss='mean_absolute_error',
                    #   loss='mean_absolute_percentage_error',
                    #   loss='mean_squared_logarithmic_error',
                    #   loss='kullback_leibler_divergence',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    
    def getModel(self):
        return self.model