import pandas as pd
import numpy as np

import scipy.stats

import tensorflow as tf
from tensorflow.keras import datasets, layers, models


def main():
    ''' Load dataset
        - テストは，一人データ抜き交差検定を行う．
        (テストデータに一人分のデータを使い，全ての人のデータがテストデータになるようにする．)
        - 人数分の回数実験を行い，平均精度を報告．
    '''

    # データセットで使う要素を指定．
    target_label = 'SS_ternary'
    # target_label = 'TC_ternary'
    # target_label = 'TS_ternary'
    # target_label = ['SS_ternary', 'TC_ternary', 'TS_ternary']
    drop_labels = ['start(exchange)[ms]', 'end(system)[ms]', 'end(exchange)[ms]', \
                        'kinectstart(exchange)[ms]', 'kinectend(exchange)[ms]', \
                        # 'SS_ternary', \
                        'TC_ternary', \
                        'TS_ternary', \
                        'SS', 'TC1', 'TC2', 'TC3', 'TC4', 'TC5', \
                        'TS1', 'TS2', 'TS3', 'TS4', 'TS5']


    # 全ての人のデータを読み込む．
    df = pd.DataFrame()
    # df = pd.concat([pd.read_csv(i) for i in lst])
    df = pd.concat([pd.read_csv(i) for i in lst], ignore_index=True)
    # print(df)

    # ターゲットラベルを分離．
    df_labels = df[target_label]

    # 不要な要素を削除．
    df = df.drop(columns=drop_labels) 
    df = df.drop(columns=target_label) 

    # 読み込んだデータを列ごとにz化（標準化：平均0，分散1）
    df = scipy.stats.zscore(df)
    # print(df)

    # NaNがある列を削除．
    # df.isna().all()
    df = df.dropna(axis='columns')
    # print(df)

    # テストデータとトレーニングデータを分ける．
    ## 各ファイルのインデックスを取得する．
    sub_top = 0
    sub_bottom = -1
    im_list = []
    for file_nema in lst:
        sub_df = pd.read_csv(file_nema)
        sub_top = sub_bottom + 1
        sub_bottom += len(sub_df)
        im_list.append([sub_top, sub_bottom])



    # top行目からbottom行目を取り出す．
    sum_acc = 0
    for i in range(len(lst)):
        # i = 2
        top = im_list[i][0]
        bottom = im_list[i][1]

        test_datas = df[top:bottom].values
        test_datas = test_datas.reshape(int(test_datas.size/test_datas[0].size), test_datas[0].size, 1, 1)
        test_labels = df_labels[top:bottom].values

        train_datas = df.drop([top,bottom]).values
        train_datas = train_datas.reshape(int(train_datas.size/train_datas[0].size), train_datas[0].size, 1, 1)
        train_labels = df_labels.drop([top,bottom]).values


        # print(train_labels)
        # print(train_datas)
        print('train_datas.shape =', train_datas.shape)
        print('train_labels.shape =', train_labels.shape)


        input_size = len(df.columns)
        # input_size = len(df.index)
        print(input_size)


        #@brief Criate fundamental model of CNN
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 1), activation='relu', input_shape=(input_size, 1, 1)))
        model.add(layers.MaxPooling2D((2, 1)))
        model.add(layers.Conv2D(64, (3, 1), activation='relu'))
        model.add(layers.MaxPooling2D((2, 1)))
        model.add(layers.Conv2D(64, (3, 1), activation='relu'))

        model.add(layers.MaxPooling2D((2, 1)))
        model.add(layers.Conv2D(64, (3, 1), activation='relu'))

        # By using layers.Flatten(), Convert tensor to scoler
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(3, activation='softmax'))
        model.add(layers.Dense(1, activation='sigmoid'))
        # model.add(layers.Dense(3, activation='sigmoid'))

        model.summary()


        #@brier Compile model and Learning
        model.compile(optimizer='adam',
                    #   loss='sparse_categorical_crossentropy',
                    #   loss='mean_squared_error',
                    #   loss='mean_absolute_error',
                    #   loss='mean_absolute_percentage_error',
                    #   loss='mean_squared_logarithmic_error',
                    #   loss='kullback_leibler_divergence',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])


        model.fit(train_datas, train_labels, epochs=15)


        # test_csv_file = "./Hazumi1902-master/dumpfiles/1902F7001.csv"
        # test_datas, test_labels, test_col_size, test_row_size = LoadDataSet(test_csv_file)

        test_loss, test_acc = model.evaluate(test_datas,  test_labels, verbose=2)

        print("test_acc =", test_acc)
        sum_acc += test_acc

    print('sum_acc =', sum_acc)
    print('sum_acc =', (sum_acc/len(lst)))


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