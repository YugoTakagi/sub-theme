import sys
sys.path.append('~/Nextcloud/AIR-JAIST/00_M1-ワーク/M1_副テーマ/岡田研/sub-theme')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import glob

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
    dataset_dir = '../ws/Hazumi1902-master/dumpfiles/*'
    lst = sorted( glob.glob(dataset_dir) )
    names = [r.split('/')[-1] for r in lst]
    num_peoples = len(names)
    # print('-> Load csv:\n', names)

    okada_data_set = OkadaDataSet(lst, target_label, drop_labels)

    t_age = 60

    t_setumeis = []
    for l in range(num_peoples):
        # print('>>>', names[l][5:7])
        _age = int( names[l][5:7] )
        if t_age == _age:
            _setumei, _mokuteki = okada_data_set.get_testdatas(l)
            # ndarray -> torch.Tensor
            t_setumeis.append( torch.from_numpy(_setumei.astype(np.float32)).clone() )

    # 各要素のL1損失を計算．
    device = torch.device('cpu') 
    l1_loss = torch.nn.L1Loss().to(device)
    
    lhs_setumei = t_setumeis[0]
    rhs_setumei = t_setumeis[1]

    num_setumei = lhs_setumei.shape[1]
    # num_row = min(lhs_setumei[:, i].shape[0], rhs_setumei[:, i].shape[0]) - 1
    num_row = min(lhs_setumei[:, 0].shape[0], rhs_setumei[:, 0].shape[0]) - 1
    l1s = []
    for i in range(num_setumei):
        l1s.append( l1_loss(lhs_setumei[0:num_row, i], rhs_setumei[0:num_row, i]) )
    
    # plt.plot(l1s, '.')
    # plt.show()

    print('max index:', l1s.index(max(l1s)))
    l1s_val = list(map(np.float32, l1s))
    # print(type(l1s_val[0]))
    # print(l1s_val[0])
    # print(okada_data_set.setumei.columns[l1s.index(max(l1s))])
    # l1s_val.sort(reverse=True)
    index = np.argsort(l1s_val)[::-1]
    # print( l1s_val[index[0]] )

    for i in range(5):
        print(okada_data_set.setumei.columns[ index[i] ])






    

    # print_(lhs_setumei[:, 0]) : 0番目の要素列が取れる．
    # x print_(lhs_setumei[0, :]) : 0番目の要素列が取れる．
    # print(lhs_setumei.shape)
    # print(lhs_setumei[:, 0].shape)
    # print(lhs_setumei[0, :].shape)

    # l1s = []
    # for i in range(num_setumei):
    #     num_row = min(lhs_setumei[:, i].shape[0], rhs_setumei[:, i].shape[0]) - 1
    #     # print(num_row)
    #     # print(lhs_setumei[0:num_row, i].shape)
    #     # print(rhs_setumei[0:num_row, i].shape)

    #     l1s.append( l1_loss(lhs_setumei[0:num_row, i], rhs_setumei[0:num_row, i]) )

    # plt.plot(l1s)
    # plt.show()

    # print('max index:', l1s.index(max(l1s)))
    # print(okada_data_set.setumei.columns[l1s.index(max(l1s))])




from inspect import currentframe
def print_(*args):
    names = {id(v): k for k, v in currentframe().f_back.f_locals.items()}
    print('\n'.join([names.get(id(arg), '???') + ' = ' + repr(arg) for arg in args]))



if __name__ == '__main__':
    main()