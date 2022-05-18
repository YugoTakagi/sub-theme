#モジュールの読み込み
# from __future__ import print_function

# import pandas as pd
# from pandas import Series,DataFrame

# from sklearn import svm
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

import numpy as np
# import matplotlib.pyplot as plt

# import keras
# from keras.datasets import fashion_mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import RMSprop
# from keras.optimizers import RMSprop
# from keras.optimizers import Adam
# from keras.utils import plot_model

import scipy.stats
from keras.utils import np_utils #Numpyユーティリティのインポート
# from tensorflow.keras import datasets, layers, models, optimizers, losses

from models.mlp_model import MyNN
from models.cnn_model import MyCNN
from data.okada_dataset import OkadaDataSet

import glob

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

def main():
    dataset_dir = '../ws/Hazumi1902-master/dumpfiles/*'
    lst = sorted( glob.glob(dataset_dir) )
    names = [r.split('/')[-1] for r in lst]
    print('-> Load csv:\n', lst)

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
        print('({}) when test file is \'{}\', accuracy : {}'.format(i, names[i], accs[i]))
    







if __name__ == '__main__':
    main()