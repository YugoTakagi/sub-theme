import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch

import glob

import sys
sys.path.append('~/Nextcloud/AIR-JAIST/00_M1-ワーク/M1_副テーマ/岡田研/sub-theme')
from data.okada_dataset import OkadaDataSet

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
    path = '../results.csv'
    df = pd.read_csv(path)

    status = df['age'].values

    fig = plt.figure()
    axs = []

    for j in range(len(df.columns)): # columnsごとのグラフが欲しい．
        if df.columns[j] == 'age':
            continue
        else:
            axs.append( fig.add_subplot(2, 3, j) ) # 行，列，場所．

            # ages = []
            # sexs = []
            # values = []
            for i in range(len(status)): # ファイル分のデータにアクセス．
                ages = status[i][1:3]
                sexs = status[i][0]
                values = df[df.columns[df.columns != 'age']].iloc[i].values
                # print(i, 'values', values)

                # 描画処理．
                if sexs == 'M':
                    # print('label: {}, ages: {}, value: {}'.format(df.columns[j], ages, values[j-1]))
                    # axs[j-1].plot(ages, values[j-1], '.', label=df.columns[j], color='skyblue')
                    axs[j-1].plot(ages, values[j-1], '.', color='#1f77b4') # color := tab:blue
                    # axs[j-1].plot(ages, values[j-1], '.', color='dodgerblue')
                elif sexs == 'F':
                    # print('label: {}, ages: {}, value: {}'.format(df.columns[j], ages, values[j-1]))
                    # axs[j-1].plot(ages, values[j-1], '.', label=df.columns[j], color='tomato')
                    axs[j-1].plot(ages, values[j-1], '.', color='#ff7f0e') # color := tab:orange
                    # axs[j-1].plot(ages, values[j-1], '.', color='tomato')
                else:
                    print('err')
        
            axs[j-1].set_title(str(df.columns[j]))
            axs[j-1].set_ylim(0, 1)
            # axs[j-1].legend()
            axs[j-1].grid(':')
    
    plt.show()





from inspect import currentframe
def print_(*args):
    names = {id(v): k for k, v in currentframe().f_back.f_locals.items()}
    print('\n'.join([names.get(id(arg), '???') + ' = ' + repr(arg) for arg in args]))



if __name__ == '__main__':
    main()