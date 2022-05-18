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



def main():
    okada_data_set = OkadaDataSet(lst, target_label, drop_labels)
    # i = 0
    sum_acc = 0
    accs = []
    for i in range(len(lst)):
        print('% ({}/{})'.format(i, len(lst)))
        train_setumei, train_mokuteki = okada_data_set.get_traindatas(i)
        test_setumei, test_mokuteki = okada_data_set.get_testdatas(i)

        # 3値分類．
        # train_mokuteki = keras.utils.to_categorical(train_mokuteki, 3)
        # test_mokuteki = keras.utils.to_categorical(test_mokuteki, 3)
        train_mokuteki = np_utils.to_categorical(train_mokuteki, 3)
        test_mokuteki = np_utils.to_categorical(test_mokuteki, 3)

        # print('setumei_size:', train_setumei.shape[1])
        setumei_size = train_setumei.shape[1]

        # > MLP
        mynn = MyNN(setumei_size)
        model = mynn.getModel()

        # > CNN
        # mycnn = MyCNN(setumei_size)
        # model = mycnn.getModel()
        # # data[0].size: 1415, data.shape[1]: 1415
        # # data.size: 2596525, data.shape[0]: 1835
        # train_setumei = train_setumei.reshape(train_setumei.shape[0], train_setumei.shape[1], 1, 1)
        # test_setumei = test_setumei.reshape(test_setumei.shape[0], test_setumei.shape[1], 1, 1)


        #ニューラルネットワークの学習
        # history = model.fit(x_train, y_train, batch_size=200,epochs=1000,verbose=1,validation_data=(x_test, y_test))
        history = model.fit(train_setumei, train_mokuteki, batch_size=200, epochs=1000, verbose=1, validation_data=(test_setumei, test_mokuteki))
        # history = model.fit(train_setumei, train_mokuteki, batch_size=200, epochs=1000)
        # history = model.fit(train_setumei, train_mokuteki, batch_size=200, epochs=200)

        #ニューラルネットワークの推論
        # score = model.evaluate(x_test,y_test,verbose=1)1
        # score = model.evaluate(test_setumei, test_mokuteki, verbose=1)
        score = model.evaluate(test_setumei, test_mokuteki)
        print("\n")
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

        sum_acc += score[1]
        accs.append(score[1])
    
    print('total test_acc =', (sum_acc/len(lst)))

    for i in range(len(lst)):
        print('({}) when test file is \'{}\', accuracy : {}'.format(i, lst[i], accs[i]))
    


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





# データセットのラベルを設定．
target_label = 'SS_ternary'
# target_label = 'TC_ternary'
# target_label = 'TS_ternary'
drop_labels = ['start(exchange)[ms]', 'end(system)[ms]', 'end(exchange)[ms]', \
                'kinectstart(exchange)[ms]', 'kinectend(system)[ms]', 'kinectend(exchange)[ms]', \
                'SS_ternary', \
                'TC_ternary', \
                'TS_ternary', \
                'SS', 
                'TC1', 'TC2', 'TC3', 'TC4', 'TC5', \
                'TS1', 'TS2', 'TS3', 'TS4', 'TS5']

# データセットに読み込むファイル名．
lst = ["../ws/Hazumi1902-master/dumpfiles/1902F2001.csv",
       "../ws/Hazumi1902-master/dumpfiles/1902F2002.csv",
       "../ws/Hazumi1902-master/dumpfiles/1902F3001.csv",
       "../ws/Hazumi1902-master/dumpfiles/1902F3002.csv",
       "../ws/Hazumi1902-master/dumpfiles/1902F4001.csv",
       "../ws/Hazumi1902-master/dumpfiles/1902F4002.csv",
       "../ws/Hazumi1902-master/dumpfiles/1902F4003.csv",
       "../ws/Hazumi1902-master/dumpfiles/1902F4004.csv",
       "../ws/Hazumi1902-master/dumpfiles/1902F4005.csv",
       "../ws/Hazumi1902-master/dumpfiles/1902F4006.csv",
       "../ws/Hazumi1902-master/dumpfiles/1902F4008.csv",
       "../ws/Hazumi1902-master/dumpfiles/1902F4009.csv",
       "../ws/Hazumi1902-master/dumpfiles/1902F4010.csv",
       "../ws/Hazumi1902-master/dumpfiles/1902F4011.csv",
       
       "../ws/Hazumi1902-master/dumpfiles/1902M2001.csv",
       "../ws/Hazumi1902-master/dumpfiles/1902M3001.csv",
       "../ws/Hazumi1902-master/dumpfiles/1902M4001.csv",
       "../ws/Hazumi1902-master/dumpfiles/1902M4002.csv",
       "../ws/Hazumi1902-master/dumpfiles/1902M5001.csv",
       "../ws/Hazumi1902-master/dumpfiles/1902M5002.csv",
       "../ws/Hazumi1902-master/dumpfiles/1902M5003.csv",
       "../ws/Hazumi1902-master/dumpfiles/1902M7001.csv",
       "../ws/Hazumi1902-master/dumpfiles/1902M7002.csv"]


if __name__ == '__main__':
    main()