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


class OkadaDataSet:
    def __init__(self,file_lst, target_label, drop_labels):
        print("-----Loading Okada dataset-------")
        okada_data_set = pd.DataFrame()
        okada_data_set = pd.concat([pd.read_csv(i) for i in file_lst], ignore_index=True)

        # 目的変数の設定．
        self.mokuteki = DataFrame(okada_data_set[target_label])
        print("-----目的変数-------")
        print('shape:', self.mokuteki.shape)
        print(self.mokuteki.head())

        # 説明変数の設定．
        self.setumei = DataFrame(okada_data_set.drop(drop_labels, axis=1))
        ### 説明変数ごとにz化（標準化：平均0，分散1）
        self.setumei = scipy.stats.zscore(self.setumei)
        ### NaNがある列を削除．
        self.setumei = self.setumei.dropna(axis='columns')
        print("-----説明変数--------")
        print('shape:', self.setumei.shape)
        print(self.setumei.head())

        ## 各ファイルのインデックスを取得する．
        sub_top = 0
        sub_bottom = -1
        self.im_list = []
        for file_nema in file_lst:
            sub_df = pd.read_csv(file_nema)
            sub_top = sub_bottom + 1
            sub_bottom += len(sub_df)
            self.im_list.append([sub_top, sub_bottom])
        
    def get_testdatas(self, i):
        ''' テストデータを取得する．なお，i番目のファイルをテストデータとする．
            (top行目からbottom行目を取り出す．)
        '''
        top = self.im_list[i][0]
        bottom = self.im_list[i][1]

        test_setumei = self.setumei[top:bottom].values
        test_mokuteki = self.mokuteki[top:bottom].values

        print('-----テストデータ-----:')
        print('test_setumei.shape: {}, test_mokuteki.shape: {}'.format(test_setumei.shape, test_mokuteki.shape))

        return test_setumei, test_mokuteki
    
    def get_traindatas(self, i):
        ''' トレーニングデータを取得する．なお，i番目のファイルをテストデータとする．
            (top行目からbottom行目を取り出す．)
        '''
        top = self.im_list[i][0]
        bottom = self.im_list[i][1]

        # train_setumei = self.setumei.drop([top,bottom]).values
        # train_mokuteki = self.mokuteki.drop([top,bottom]).values
        train_setumei = self.setumei.drop(range(top,bottom)).values
        train_mokuteki = self.mokuteki.drop(range(top,bottom)).values

        print('-----トレーニングデータ-----:')
        print('train_setumei.shape: {}, train_mokuteki.shape: {}'.format(train_setumei.shape, train_mokuteki.shape))

        return train_setumei, train_mokuteki


